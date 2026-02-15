"""Prompt engineering utilities aligned to notebook training template."""

from app.pipeline import PreparedInput


def build_generation_prompt(item: PreparedInput) -> str:
    """Build an instruction prompt that asks the model to return JSON keys."""
    keyword_line = (
        ", ".join(item.seo_keywords) if item.seo_keywords else "business strategy"
    )

    context_suffix = ""
    if item.source_context:
        context_suffix = (
            "\nOptional source context (use as supporting context only):\n"
            f"{item.source_context}"
        )

    return (
        "Instruction: Write an insight-driven business article body based on the given title, category, industry, and tags.\n"
        f"Title: {item.title}\n"
        "Input: "
        f"Topic Category: {item.topic_category}; "
        f"Industry: {item.industry}; "
        f"Target length: {item.target_length_label}; "
        f"Tags: {keyword_line}; "
        f"Keywords: {keyword_line}; "
        f"Tone: {item.tone}"
        f"{context_suffix}\n"
        "Output (JSON):"
    )
