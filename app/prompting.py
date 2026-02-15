"""Prompt engineering utilities for style-aligned article generation."""

from app.pipeline import PreparedInput


def build_generation_prompt(item: PreparedInput) -> str:
    """
    Prompt template tuned to match observed style patterns from the reference article:
    - Starts with a strong misconception/tension and immediate business relevance
    - Uses clear section headings with numbered insights
    - Includes practical examples and a decision matrix/framework
    - Ends with strategic recommendations and a concise CTA
    """
    # Join SEO keywords into a human-readable line for prompt injection.
    keyword_line = ", ".join(item.seo_keywords) if item.seo_keywords else "(none)"
    # Centralized target lengths improve consistency across requests.
    length_map = {
        "short": "500-700 words",
        "medium": "900-1200 words",
        "long": "1400-1800 words",
    }

    return f"""
You are a senior strategy writer for Ideas.
Write an original, insight-driven article in English.

Input parameters:
- Topic category: {item.topic_category}
- Industry: {item.industry}
- Target audience: {item.target_audience}
- SEO keywords: {keyword_line}
- Tone: {item.tone}
- Target length: {length_map[item.article_length]}

Style requirements:
1) Start with a compelling headline and a 2-3 sentence hook.
2) Explain why the topic matters now using market/consumer context.
3) Present 4-6 clear insights or misconceptions with short examples.
4) Include a practical framework or matrix for decision making.
5) End with strategic recommendations and a concise call-to-action.
6) Keep language clear, analytical, and business-friendly.
7) Integrate SEO keywords naturally (no keyword stuffing).
8) Do not mention any specific company brand name in the output.

Optional source context (use only as supporting context, never copy):
{item.source_context if item.source_context else "No external source text provided."}

Return format:
- Title: <title>
- Article:
<full article markdown with headings and bullet points>
""".strip()
