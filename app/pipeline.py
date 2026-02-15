"""Input preprocessing pipeline for article generation.

This module normalizes raw user parameters, cleans SEO keywords, and optionally
extracts contextual text from a source URL.
"""

import re
from dataclasses import dataclass
from typing import List, Optional

import requests
from bs4 import BeautifulSoup


STOPWORDS = {
    "the",
    "and",
    "for",
    "with",
    "your",
    "from",
    "that",
    "this",
    "are",
    "into",
    "about",
}


@dataclass
class PreparedInput:
    """Normalized payload consumed by prompt building and generation."""

    topic_category: str
    industry: str
    target_audience: str
    seo_keywords: List[str]
    source_context: str
    article_length: str
    tone: str


def _normalize_text(text: str) -> str:
    """Normalize whitespace and remove hidden zero-width characters."""
    text = re.sub(r"\s+", " ", text or "").strip()
    text = text.replace("\u200b", "")
    return text


def _clean_keywords(keywords: List[str]) -> List[str]:
    """Filter invalid/common keywords, deduplicate, and cap keyword count."""
    cleaned = []
    seen = set()
    for keyword in keywords or []:
        key = _normalize_text(keyword).lower()
        if not key or key in STOPWORDS or key in seen:
            continue
        seen.add(key)
        cleaned.append(keyword.strip())
    return cleaned[:12]


def _extract_web_text(url: str, timeout: int = 10) -> str:
    """Fetch and sanitize visible webpage text for optional context enrichment."""
    try:
        resp = requests.get(url, timeout=timeout)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")
        for bad in soup(["script", "style", "noscript"]):
            bad.extract()
        text = " ".join(soup.stripped_strings)
        # Hard-cap context size to keep prompt input within practical limits.
        return _normalize_text(text)[:6000]
    except Exception:
        # Fail softly to keep generation available even when URL retrieval fails.
        return ""


def prepare_input(
    topic_category: str,
    industry: str,
    target_audience: str,
    seo_keywords: Optional[List[str]] = None,
    source_url: Optional[str] = None,
    source_text: Optional[str] = None,
    article_length: str = "medium",
    tone: str = "insightful, practical, strategic",
) -> PreparedInput:
    """Transform API payload fields into a validated `PreparedInput` object."""
    normalized_length = article_length.lower().strip()
    if normalized_length not in {"short", "medium", "long"}:
        # Fallback prevents invalid generation settings from breaking the flow.
        normalized_length = "medium"

    cleaned_keywords = _clean_keywords(seo_keywords or [])

    context_parts = []
    if source_text:
        context_parts.append(_normalize_text(source_text)[:3000])
    if source_url:
        extracted = _extract_web_text(str(source_url))
        if extracted:
            context_parts.append(extracted)

    # Merge all available context blocks in a deterministic order.
    source_context = "\n".join([part for part in context_parts if part]).strip()

    return PreparedInput(
        topic_category=_normalize_text(topic_category),
        industry=_normalize_text(industry),
        target_audience=_normalize_text(target_audience),
        seo_keywords=cleaned_keywords,
        source_context=source_context,
        article_length=normalized_length,
        tone=_normalize_text(tone),
    )
