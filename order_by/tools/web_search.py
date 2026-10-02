import asyncio
import json
import logging
import os
import re
import time
import urllib.parse
from datetime import timezone
from email.utils import parsedate_to_datetime

log = logging.getLogger(__name__)

import httpx
from ddgs import DDGS
from diskcache import Cache
from pydantic import BaseModel

from ..utils import count_tokens, hash_prompt, create_numbered_passages, create_numbered_SQLs, create_numbered_reviews
from prompts.all_prompts import web_search_system_prompt as _WEB_POINTWISE_SYSTEM_PROMPT

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
cache = Cache(os.path.join(PROJECT_ROOT, "sort_cache"), size_limit=50 * 1024**3, eviction_policy='least-recently-used')
_wiki_cache = Cache(os.path.join(PROJECT_ROOT, "wiki_cache"), size_limit=2 * 1024**3, eviction_policy='least-recently-used')

_CACHE_VERSION = "v5"
_WIKI_CACHE_VERSION = "v2"

_USER_AGENT = "OrderByLLM-research/1.0 (https://github.com/fahmi-ramadhan/OrderByLLM)"

_WIKI_MAX_CONCURRENT = int(os.getenv("WIKI_MAX_CONCURRENT", "3"))
_WIKI_MAX_ATTEMPTS = int(os.getenv("WIKI_MAX_ATTEMPTS", "5"))
_WIKI_BACKOFF_BASE = float(os.getenv("WIKI_BACKOFF_BASE", "2.0"))
_WIKI_BACKOFF_CAP = float(os.getenv("WIKI_BACKOFF_CAP", "60.0"))
_WIKI_TIMEOUT = float(os.getenv("WIKI_TIMEOUT", "10.0"))

_RETRY_STATUSES = frozenset({403, 429, 500, 502, 503, 504})
_wiki_semaphore = asyncio.Semaphore(_WIKI_MAX_CONCURRENT)
_wiki_client: httpx.AsyncClient | None = None
_wiki_client_loop = None


class WikiRateLimited(Exception):
    """Raised when Wikipedia keeps answering 429/5xx after every retry, so
    callers can treat the entity as unresolved instead of caching a bogus miss."""


def _get_wiki_client() -> httpx.AsyncClient:
    """Lazily build one AsyncClient per event loop, reusing its connection pool."""
    global _wiki_client, _wiki_client_loop
    loop = asyncio.get_running_loop()
    if _wiki_client is None or _wiki_client.is_closed or _wiki_client_loop is not loop:
        _wiki_client = httpx.AsyncClient(
            timeout=httpx.Timeout(_WIKI_TIMEOUT),
            follow_redirects=True,
            headers={"User-Agent": _USER_AGENT, "Accept-Encoding": "gzip"},
        )
        _wiki_client_loop = loop
    return _wiki_client


def _retry_after_seconds(headers) -> float | None:
    """Read a Retry-After header, which may be delta-seconds or an HTTP-date."""
    raw = headers.get("Retry-After") if headers else None
    if not raw:
        return None
    try:
        return max(0.0, float(raw))
    except (TypeError, ValueError):
        pass
    try:
        when = parsedate_to_datetime(raw)
    except (TypeError, ValueError):
        return None
    if when is None:
        return None
    if when.tzinfo is None:
        when = when.replace(tzinfo=timezone.utc)
    return max(0.0, when.timestamp() - time.time())


def _is_definitive(resp: httpx.Response | None) -> bool:
    """True when the server gave an answer we may cache.

    200 means "here is the page"; 404 means "no such page". Anything else — 403
    refusals, gateway errors, or a request that never completed — tells us
    nothing about the entity, so it must never become a cached miss.
    """
    return resp is not None and resp.status_code in (200, 404)


async def _wiki_get(url: str, params: dict | None = None) -> httpx.Response | None:
    """GET a Wikimedia endpoint under the concurrency cap with backoff retries.

    Returns the response for any status that is not worth retrying (including
    404) so callers can interpret it. Raises :class:`WikiRateLimited` when the
    last attempt still hit 429/5xx or the network failed, so a transient
    problem is never recorded as a permanent miss.
    """
    client = _get_wiki_client()
    reason = ""
    for attempt in range(1, _WIKI_MAX_ATTEMPTS + 1):
        retry_after = None
        async with _wiki_semaphore:
            try:
                resp = await client.get(url, params=params)
            except (httpx.TimeoutException, httpx.TransportError) as e:
                reason = f"network error: {type(e).__name__}"
                log.warning("wiki_get %s: %s (attempt %d/%d)", url, reason,
                            attempt, _WIKI_MAX_ATTEMPTS)
            else:
                if resp.status_code not in _RETRY_STATUSES:
                    return resp
                reason = f"HTTP {resp.status_code}"
                retry_after = _retry_after_seconds(resp.headers)
                log.warning("wiki_get %s: %s (attempt %d/%d)", url, reason,
                            attempt, _WIKI_MAX_ATTEMPTS)

        if attempt == _WIKI_MAX_ATTEMPTS:
            raise WikiRateLimited(f"{url}: {reason} after {_WIKI_MAX_ATTEMPTS} attempts")
        wait = retry_after if retry_after is not None else min(
            _WIKI_BACKOFF_BASE * (2 ** (attempt - 1)), _WIKI_BACKOFF_CAP
        )
        await asyncio.sleep(wait)

    return None


class WebSearchPointwiseResult(BaseModel):
    explanation: str
    value: float


class WebSearchExternalResult(BaseModel):
    explanation: str
    values: list[float]


# ── Wikipedia helpers ─────────────────────────────────────────────────────────

async def _fetch_page_info(entity: str) -> tuple[str, str] | None:
    """Return (title, extract) for the best-matching Wikipedia page.

    Tries a direct title lookup first (fast, avoids rate-limited search API),
    then falls back to the search API. Returns None when the entity genuinely
    has no usable page. Raises WikiRateLimited when retries are exhausted, so
    throttling is never mistaken for a missing page.
    """
    title_slug = urllib.parse.quote(entity.replace(" ", "_"))

    async def _from_summary(slug: str):
        resp = await _wiki_get(f"https://en.wikipedia.org/api/rest_v1/page/summary/{slug}")
        if not _is_definitive(resp):
            raise WikiRateLimited(f"unexpected summary HTTP {getattr(resp, 'status_code', None)}")
        if resp.status_code == 404:
            return None
        page = resp.json()
        if page.get("type") == "disambiguation":
            return None
        return page.get("title", ""), page.get("extract", "")

    result = await _from_summary(title_slug)
    if result:
        return result

    resp = await _wiki_get("https://en.wikipedia.org/w/api.php", params={
        "action": "query", "list": "search",
        "srsearch": entity, "format": "json", "srlimit": 1,
    })
    if not _is_definitive(resp):
        raise WikiRateLimited(f"unexpected search HTTP {getattr(resp, 'status_code', None)}")
    if resp.status_code == 404:
        return None
    hits = resp.json().get("query", {}).get("search", [])
    if not hits:
        return None
    slug = urllib.parse.quote(hits[0]["title"].replace(" ", "_"))
    return await _from_summary(slug)


async def _fetch_infobox_field(title: str, field: str) -> str | None:
    """Fetch section 0 for a page and pull one infobox field out of the HTML.

    Fetches section 0 HTML via the MediaWiki parse API and delegates to
    :func:`_parse_infobox_field`. Raises WikiRateLimited when retries are
    exhausted.
    """
    resp = await _wiki_get("https://en.wikipedia.org/w/api.php", params={
        "action": "parse", "page": title.replace(" ", "_"),
        "prop": "text", "section": "0", "format": "json",
    })
    if not _is_definitive(resp):
        raise WikiRateLimited(f"unexpected parse HTTP {getattr(resp, 'status_code', None)}")
    if resp.status_code == 404:
        return None
    return _parse_infobox_field(resp.json().get("parse", {}).get("text", {}).get("*", ""), field)


def _parse_infobox_field(html: str, field: str) -> str | None:
    """Extract a field value from a Wikipedia infobox HTML using regex.

    Tries two strategies:

    1. Direct match — the field label sits in a ``<th>`` immediately followed by
       a ``<td>`` with the value.
    2. Section match — the field is a section header in a ``mergedtoprow`` ``<tr>``
       and the values live in subsequent ``mergedrow`` ``<tr>`` sub-rows.  Iteration 
       stops at the first ``<tr>`` whose class is *not* ``mergedrow`` (i.e. a new
       ``mergedtoprow``, a ``mergedbottomrow``, or a classless ``<tr>``).

    In both cases HTML tags and citation references (``[1]``, ``&#91;2&#93;``)
    are stripped from the returned text.  For direct matches that contain a
    parenthesised metric value like "(2.01 m)", only that metric part is
    returned for a clean, unambiguous context string.

    Returns None if the field is not present in the infobox.
    """
    pat = rf'{re.escape(field)}</th>\s*<td[^>]*>(.*?)</td>'
    m = re.search(pat, html, re.DOTALL | re.IGNORECASE)
    if not m:
        NB = r'((?:(?!</tr>)[\s\S])*?)'
        header_pat = rf'<tr\s+class="mergedtoprow">\s*<th[^>]*>{NB}{re.escape(field)}{NB}</th>{NB}</tr>'
        hm = re.search(header_pat, html, re.DOTALL | re.IGNORECASE)
        if hm:
            rest = html[hm.end():]
            parts = []
            for rm in re.finditer(r'<tr([^>]*)>', rest, re.IGNORECASE):
                cls = re.search(r'class="([^"]+)"', rm.group(1))
                if not cls or cls.group(1) != "mergedrow":
                    break
                chunk = rest[rm.start():]
                td_m = re.search(r'<th[^>]*>(.*?)</th>\s*<td[^>]*>(.*?)</td>', chunk, re.DOTALL | re.IGNORECASE)
                if not td_m:
                    break
                label = re.sub(r"<[^>]+>", "", td_m.group(1))
                label = label.replace("&#160;", " ").replace("&nbsp;", " ")
                label = re.split(r"&#91;|\[", label)[0].strip().lstrip("\u2022").strip()
                value = re.sub(r"<[^>]+>", "", td_m.group(2))
                value = value.replace("&#160;", " ").replace("&nbsp;", " ")
                value = re.split(r"&#91;|\[", value)[0].strip()
                if label and value:
                    parts.append(f"{label}: {value}")
            if parts:
                return "; ".join(parts)
        return None

    clean = re.sub(r"<[^>]+>", "", m.group(1))
    clean = clean.replace("&#160;", " ").replace("&nbsp;", " ")
    clean = re.split(r"&#91;|\[", clean)[0].strip()
    if not clean:
        return None

    # Prefer the parenthesised metric value when present (e.g. "6 ft 7 in (2.01 m)" → "2.01 m")
    metric = re.search(r"\(([\d.]+)\s*m\)", clean)
    return f"{metric.group(1)} m" if metric else clean


async def wiki_status(entity: str, wiki_field: str | None = None) -> dict:
    """Resolve an entity to its Wikipedia context."""
    page_info = await _fetch_page_info(entity)
    if not page_info:
        return {"status": "not_found", "reason": "no_page", "title": "",
                "header": "", "extract": ""}
    title, extract = page_info

    field_value = None
    if wiki_field:
        try:
            field_value = await _fetch_infobox_field(title, wiki_field)
        except WikiRateLimited:
            raise
        except Exception:
            field_value = None
        if field_value is None:
            return {"status": "not_found", "reason": f"field_absent:{wiki_field}",
                    "title": title, "header": "", "extract": ""}

    header = f"Wikipedia — {title}"
    if wiki_field and field_value:
        header += f" [{wiki_field}: {field_value}]"
    header += ":"
    return {"status": "ok", "reason": "", "title": title,
            "header": header, "extract": extract}


def _wiki_cache_key(entity: str, wiki_field: str | None) -> str:
    return f"wiki_{_WIKI_CACHE_VERSION}:{entity}|field:{wiki_field or ''}"


async def wiki_search(entity: str, wiki_field: str | None = None, max_chars: int = 500) -> str:
    cache_key = _wiki_cache_key(entity, wiki_field)
    chars = 0 if wiki_field else max_chars
    if cache_key in _wiki_cache:
        cached = _wiki_cache[cache_key]
        if cached is None:
            return ""
        return f"{cached['header']} {cached['extract'][:chars]}".strip()

    try:
        info = await wiki_status(entity, wiki_field)
    except WikiRateLimited:
        log.warning("wiki_search %r: rate limited through %d attempts, leaving uncached",
                    entity, _WIKI_MAX_ATTEMPTS)
        return ""
    except Exception:
        return ""

    if info["status"] != "ok":
        _wiki_cache[cache_key] = None
        return ""

    _wiki_cache[cache_key] = {"header": info["header"], "extract": info["extract"]}
    return f"{info['header']} {info['extract'][:chars]}".strip()


# ── Pointwise value scoring ───────────────────────────────────────────────────

async def web_search_pointwise_value(
    client,
    modelname: str,
    prompt: str,
    wiki_entity: str,
    wiki_field: str | None = None,
):
    """Return a float score for a single item, optionally grounded by Wikipedia.

    Parameters
    ----------
    wiki_entity:
        Bare entity name used for the Wikipedia lookup (e.g. "LeBron James").
    wiki_field:
        Infobox label to extract (e.g. "Listed height").
        If found: the prompt is augmented with that fact as context.
        If not found or omitted: the raw prompt is sent so the LLM uses its
        own knowledge.
    """
    field_tag = f"[field={wiki_field}]" if wiki_field else ""
    wiki_context = await wiki_search(wiki_entity, wiki_field=wiki_field)
    ctx_tag = f"[ctx={hash_prompt(wiki_context, 'wiki')[:16]}]"
    cache_key = f"[web_search_{_CACHE_VERSION}][wiki={wiki_entity}]{field_tag}{ctx_tag}{prompt}"
    key_hash = hash_prompt(cache_key, modelname)

    if key_hash in cache:
        try:
            cached = cache[key_hash]
            parsed = WebSearchPointwiseResult(**cached["parsed"])
            input_tokens = cached.get("input_tokens", count_tokens(prompt))
            output_tokens = cached["tokens"] - input_tokens
            return parsed.value, 0, input_tokens, output_tokens
        except Exception:
            del cache[key_hash]

    augmented_prompt = (
        f"Use the following context.\n\n"
        f"Context:\n{wiki_context or 'No data found.'}\n\n"
        f"{prompt}"
    )

    for attempt in range(1, 11):
        try:
            response = await client.beta.chat.completions.parse(
                model=modelname,
                messages=[
                    {"role": "system", "content": _WEB_POINTWISE_SYSTEM_PROMPT},
                    {"role": "user", "content": augmented_prompt},
                ],
                temperature=0.0,
                response_format=WebSearchPointwiseResult,
                max_completion_tokens=8192,
            )
            break
        except Exception as e:
            msg = str(e).lower()
            if "429" in msg or "rate limit" in msg:
                wait = 30.0 * attempt
                log.warning("web_search_pointwise_value %s: rate limited (attempt %d/10), sleeping %.0fs",
                            modelname, attempt, wait)
                await asyncio.sleep(wait)
            else:
                log.error("web_search_pointwise_value %s [ERROR] attempt %d: %s", modelname, attempt, e)
                return 0.0, 1, 0, 0
    else:
        log.error("web_search_pointwise_value %s: all 10 retries exhausted", modelname)
        return 0.0, 1, 0, 0

    parsed = response.choices[0].message.parsed
    input_tokens = (
        getattr(response.usage, "input_tokens", None)
        or getattr(response.usage, "prompt_tokens", None)
        or count_tokens(augmented_prompt)
    )
    output_tokens = response.usage.total_tokens - input_tokens
    cache[key_hash] = {
        "parsed": parsed.model_dump(),
        "tokens": response.usage.total_tokens,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
    }
    return float(parsed.value), 1, input_tokens, output_tokens


# ── External pointwise scoring ───────────────────────────────────────────────

def _ddg_search(query: str, max_results: int = 5, max_chars_per_result: int = 300) -> str:
    """Return DuckDuckGo search results as a text snippet."""
    try:
        results = list(DDGS().text(query, max_results=max_results))
    except Exception:
        return "No results found."
    if not results:
        return "No results found."
    return "\n".join(f"- {r['title']}: {r['body'][:max_chars_per_result]}" for r in results)


def ddg_search_cached(query: str, max_results: int = 5, max_chars_per_result: int = 300) -> str:
    """Cached DuckDuckGo search — same query always returns the same result."""
    cache_key = f"ddg_v1:{query}:{max_results}:{max_chars_per_result}"
    if cache_key in _wiki_cache:
        return _wiki_cache[cache_key]
    result = _ddg_search(query, max_results=max_results, max_chars_per_result=max_chars_per_result)
    _wiki_cache[cache_key] = result
    return result


def _extract_vals(parsed, schema):
    for attr in ("values", "relevance_scores", "correctness_scores", "review_scores"):
        if hasattr(parsed, attr):
            return getattr(parsed, attr)
    return []


async def wiki_search_external_values(data, client, prompt_template, modelname, output_type, schema, wiki_field=None):
    """Score a batch of items using Wikipedia infobox-augmented prompts.

    For each item in the batch, fetches the ``wiki_field`` value from Wikipedia
    (e.g. "Listed height"). Found values are prepended as context; items with no
    data get a "Not found" note so the LLM falls back to its own knowledge.
    """
    base_prompt = prompt_template.format(keys=str(data))

    contexts = await asyncio.gather(*[
        wiki_search(str(item), wiki_field=wiki_field) for item in data
    ])
    context_lines = [
        f"[Item {i+1}] {ctx}" if ctx else f"[Item {i+1}] No data found."
        for i, ctx in enumerate(contexts)
    ]
    combined_context = "\n".join(context_lines)

    augmented_prompt = (
        f"Use the following Wikipedia context for each item.\n\n"
        f"Context per item:\n{combined_context}\n\n"
        f"{base_prompt}"
    )

    field_tag = f"[field={wiki_field}]" if wiki_field else ""
    cache_key = f"[wiki_search_ext_{_CACHE_VERSION}]{field_tag}{augmented_prompt}"
    key_hash = hash_prompt(cache_key, modelname)
    if key_hash in cache:
        try:
            cached = cache[key_hash]
            parsed = schema(**cached["parsed"])
            vals = _extract_vals(parsed, schema)
            input_tokens = cached.get("input_tokens", count_tokens(augmented_prompt))
            output_tokens = cached["tokens"] - input_tokens
            if len(vals) == len(data):
                return [output_type(v) for v in vals], 0, input_tokens, output_tokens
            del cache[key_hash]
        except Exception:
            del cache[key_hash]

    for attempt in range(1, 11):
        # From attempt 2 onward, remind the model of the required output length.
        suffix = (
            f"\nRespond in JSON. The values list must contain exactly {len(data)} floats, one per item.\n"
            if attempt > 1 else ""
        )
        temperature = 0.0 if attempt <= 5 else 0.3
        try:
            response = await client.beta.chat.completions.parse(
                model=modelname,
                messages=[
                    {"role": "system", "content": _WEB_POINTWISE_SYSTEM_PROMPT},
                    {"role": "user", "content": augmented_prompt + suffix},
                ],
                temperature=temperature,
                response_format=schema,
                max_completion_tokens=8192,
            )
            parsed = response.choices[0].message.parsed
            vals = _extract_vals(parsed, schema)
            input_tokens = (
                getattr(response.usage, "input_tokens", None)
                or getattr(response.usage, "prompt_tokens", None)
                or count_tokens(augmented_prompt)
            )
            output_tokens = response.usage.total_tokens - input_tokens

            if len(vals) == len(data):
                cache[key_hash] = {
                    "parsed": parsed.model_dump(),
                    "tokens": response.usage.total_tokens,
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
                }
                return [output_type(v) for v in vals], 1, input_tokens, output_tokens

            log.warning("wiki_search_external_values %s: length mismatch attempt %d/10, got %d expected %d",
                        modelname, attempt, len(vals), len(data))
        except Exception as e:
            msg = str(e).lower()
            if "429" in msg or "rate limit" in msg:
                wait = 30.0 * attempt
                log.warning("wiki_search_external_values %s: rate limited (attempt %d/10), sleeping %.0fs",
                            modelname, attempt, wait)
                await asyncio.sleep(wait)
            else:
                log.error("wiki_search_external_values %s [ERROR] attempt %d: %s", modelname, attempt, e)

    log.error("wiki_search_external_values %s: all 10 retries exhausted, returning zeros", modelname)
    return [output_type(0) for _ in data], 1, 0, 0


async def web_search_external_values(data, client, prompt_template, modelname, output_type, schema):
    """Score a batch of items using DuckDuckGo-augmented prompts via chat completions."""
    from ..pointwise import (
        PassageExternalPointwiseReasoning,
        SQLExternalPointwiseReasoning,
        ReviewExternalPointwiseReasoning,
    )

    if schema == PassageExternalPointwiseReasoning:
        base_prompt = prompt_template.format(keys=str(create_numbered_passages(data)))
    elif schema == SQLExternalPointwiseReasoning:
        base_prompt = prompt_template.format(keys=str(create_numbered_SQLs(data)))
    elif schema == ReviewExternalPointwiseReasoning:
        base_prompt = prompt_template.format(keys=str(create_numbered_reviews(data)))
    else:
        base_prompt = prompt_template.format(keys=str(data))

    contexts = [_ddg_search(str(item)[:150], max_results=3) for item in data]
    combined_context = "\n\n".join(f"[Item {i+1}] {ctx}" for i, ctx in enumerate(contexts))

    augmented_prompt = (
        f"Use the following web search results as context for each item.\n\n"
        f"Web context per item:\n{combined_context}\n\n"
        f"{base_prompt}"
    )

    cache_key = f"[web_search_ext]{augmented_prompt}"
    key_hash = hash_prompt(cache_key, modelname)
    if key_hash in cache:
        try:
            cached = cache[key_hash]
            parsed = schema(**cached["parsed"])
            vals = _extract_vals(parsed, schema)
            input_tokens = cached.get("input_tokens", count_tokens(augmented_prompt))
            output_tokens = cached["tokens"] - input_tokens
            if len(vals) == len(data):
                return [output_type(v) for v in vals], 0, input_tokens, output_tokens
            del cache[key_hash]
        except Exception:
            del cache[key_hash]

    response = await client.beta.chat.completions.parse(
        model=modelname,
        messages=[
            {"role": "system", "content": _WEB_POINTWISE_SYSTEM_PROMPT},
            {"role": "user", "content": augmented_prompt},
        ],
        temperature=0.0,
        response_format=schema,
        max_completion_tokens=8192,
    )
    parsed = response.choices[0].message.parsed
    vals = _extract_vals(parsed, schema)
    input_tokens = getattr(response.usage, "prompt_tokens", None) or count_tokens(augmented_prompt)
    output_tokens = response.usage.total_tokens - input_tokens

    if len(vals) == len(data):
        cache[key_hash] = {
            "parsed": parsed.model_dump(),
            "tokens": response.usage.total_tokens,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
        }
    else:
        print(f"web_search_external_values: length mismatch, got {len(vals)}, expected {len(data)}")
        vals = [0.0] * len(data)

    return [output_type(v) for v in vals], 1, input_tokens, output_tokens
