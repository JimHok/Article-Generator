"""Model loading and article generation utilities.

This module wraps Hugging Face model/tokenizer initialization and exposes a
single generator class used by the API layer.
"""

from functools import lru_cache
import re
from typing import Dict, Tuple

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from app.pipeline import PreparedInput
from app.prompting import build_generation_prompt


class ArticleGenerator:
    """Generates insight articles from structured business inputs."""

    def __init__(self, model_name_or_path: str = "google/flan-t5-base"):
        # Load tokenizer/model once per process for stable runtime performance.
        self.model_name_or_path = model_name_or_path
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name_or_path)
        # Prefer GPU when available to reduce generation latency.
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def _build_params(self, article_length: str) -> Dict:
        """Map logical length choices to generation token budgets."""
        if article_length == "short":
            return {"max_new_tokens": 512, "min_new_tokens": 220}
        if article_length == "long":
            return {"max_new_tokens": 1400, "min_new_tokens": 700}
        return {"max_new_tokens": 900, "min_new_tokens": 400}

    def generate(self, item: PreparedInput) -> Tuple[str, str]:
        """Generate and parse an article response.

        Returns a tuple: (title, article_markdown).
        """
        prompt = build_generation_prompt(item)
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=2048,
        ).to(self.device)

        length_params = self._build_params(item.article_length)

        # Disable gradients for inference to save memory and speed up execution.
        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                do_sample=True,
                temperature=0.7,
                top_p=0.92,
                repetition_penalty=1.1,
                num_beams=4,
                **length_params,
            )

        decoded = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)

        # Fallbacks protect API response shape if model output misses markers.
        title = "Generated Insight Article"
        article = decoded
        if "Title:" in decoded and "Article:" in decoded:
            lines = decoded.splitlines()
            for line in lines:
                if line.lower().startswith("title:"):
                    title = line.split(":", 1)[1].strip() or title
                    break
            article = decoded.split("Article:", 1)[-1].strip()

        # Normalize extra whitespace from model generations.
        title = title.strip()
        article = article.strip()
        title = re.sub(r"\s{2,}", " ", title)
        article = re.sub(r"\s{2,}", " ", article)
        if not title:
            title = "Generated Insight Article"

        return title, article


@lru_cache(maxsize=1)
def get_generator(model_name_or_path: str) -> ArticleGenerator:
    """Return a cached generator instance to avoid repeated model loading."""
    return ArticleGenerator(model_name_or_path=model_name_or_path)
