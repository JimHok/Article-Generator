"""FastAPI application entrypoint for content generation."""

import os

from fastapi import FastAPI

from app.generator import get_generator
from app.pipeline import prepare_input
from app.schemas import GenerateRequest, GenerateResponse

app = FastAPI(title="Article Generator Prototype", version="1.0.0")

# Model path can point to a Hugging Face hub ID or local fine-tuned artifacts.
MODEL_PATH = os.getenv("MODEL_PATH", "artifacts/article-generator-flan-t5-lora")


@app.get("/health")
def health_check():
    """Return service liveness and currently configured model path."""
    return {"status": "ok", "model": MODEL_PATH}


@app.post("/generate", response_model=GenerateResponse)
def generate_article(payload: GenerateRequest):
    """Generate an article from structured business parameters."""
    # Convert inbound request into normalized model-ready input.
    prepared = prepare_input(
        topic_category=payload.topic_category,
        industry=payload.industry,
        seo_keywords=payload.seo_keywords,
        source_url=str(payload.source_url) if payload.source_url else None,
        source_text=payload.source_text,
        article_length=payload.article_length,
        tone=payload.tone,
    )

    # Lazily initialize (and cache) model resources for efficient API usage.
    generator = get_generator(MODEL_PATH)
    title, article = generator.generate(prepared)

    return GenerateResponse(
        title=title,
        article=article,
        used_keywords=prepared.seo_keywords,
        model_name=MODEL_PATH,
    )
