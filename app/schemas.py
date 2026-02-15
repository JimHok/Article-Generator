"""Pydantic schemas for notebook-aligned article generation API."""

from typing import List, Literal, Optional

from pydantic import BaseModel, Field, HttpUrl


class GenerateRequest(BaseModel):
    """Inbound payload used to build a structured prompt for generation."""

    topic_category: str = Field(
        ..., description="Main topic category (e.g., Marketing, Technology)."
    )
    industry: str = Field(..., description="Target industry (e.g., FMCG, Banking).")
    seo_keywords: List[str] = Field(
        default_factory=list, description="SEO keywords to include naturally."
    )
    source_url: Optional[HttpUrl] = Field(
        default=None,
        description="Optional URL to extract supporting context from.",
    )
    source_text: Optional[str] = Field(
        default=None,
        description="Optional direct source notes or document text.",
    )
    article_length: Literal["short", "medium", "long"] = Field(
        default="medium",
        description="short | medium | long",
    )
    tone: str = Field(
        default="insightful, practical, strategic",
        description="Writing tone guideline.",
    )


class GenerateResponse(BaseModel):
    """Structured response with title/article parsed from model output."""

    title: str
    article: str
    used_keywords: List[str]
    model_name: str
