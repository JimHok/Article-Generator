"""Model loading and article generation utilities.

This module wraps Hugging Face model/tokenizer initialization and exposes a
single generator class used by the API layer.
"""

from functools import lru_cache
import json
from pathlib import Path
import re
from typing import Dict, Tuple

from peft import PeftConfig, PeftModel
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from app.pipeline import PreparedInput
from app.prompting import build_generation_prompt


class ArticleGenerator:
    """Generates insight articles from structured business inputs."""

    def __init__(self, model_name_or_path: str = "google/flan-t5-base"):
        # Load tokenizer/model once per process for stable runtime performance.
        self.model_name_or_path = model_name_or_path
        self.tokenizer, self.model = self._load_tokenizer_and_model(model_name_or_path)
        # Prefer GPU when available to reduce generation latency.
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def _load_tokenizer_and_model(self, model_name_or_path: str):
        """Load either a full model or a PEFT adapter output from training."""
        model_path = Path(model_name_or_path)
        is_local_adapter = (
            model_path.exists() and (model_path / "adapter_config.json").exists()
        )

        if is_local_adapter:
            peft_config = PeftConfig.from_pretrained(model_name_or_path)
            base_model_name = peft_config.base_model_name_or_path
            if not base_model_name:
                raise ValueError("Adapter config is missing `base_model_name_or_path`.")
            base_model = AutoModelForSeq2SeqLM.from_pretrained(base_model_name)
            model = PeftModel.from_pretrained(base_model, model_name_or_path)
            tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
            return tokenizer, model

        # Fallback: treat provided value as a full model path or hub model id.
        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name_or_path)
        return tokenizer, model

    def _build_params(self, article_length: str) -> Dict:
        """Map logical length choices to generation token budgets."""
        if article_length == "short":
            return {"max_new_tokens": 220, "min_new_tokens": 80}
        if article_length == "long":
            return {"max_new_tokens": 420, "min_new_tokens": 160}
        return {"max_new_tokens": 320, "min_new_tokens": 120}

    def _parse_output(self, decoded: str) -> Tuple[str, str]:
        """Parse model output into title and article body robustly."""
        normalized_text = decoded.strip()

        try:
            parsed = json.loads(normalized_text)
            title = str(parsed.get("Title", "Generated Insight Article")).strip()
            article = str(parsed.get("Article", "")).strip()
            return title or "Generated Insight Article", article
        except json.JSONDecodeError:
            pass

        title_match = re.search(
            r'"Title"\s*:\s*"(?P<title>(?:[^"\\]|\\.)*)"',
            normalized_text,
            re.IGNORECASE | re.DOTALL,
        )
        article_match = re.search(
            r'"Article"\s*:\s*(?P<article>.*)$',
            normalized_text,
            re.IGNORECASE | re.DOTALL,
        )

        title_value = "Generated Insight Article"
        if title_match:
            title_value = (
                title_match.group("title").strip() or "Generated Insight Article"
            )

        article_value = normalized_text
        if article_match:
            article_value = article_match.group("article").strip()
            article_value = article_value.lstrip('"').strip()
            article_value = article_value.rstrip('"').strip()
            article_value = article_value.rstrip(",").strip()

        return title_value, article_value

    def _looks_like_placeholder_output(self, text: str) -> bool:
        """Detect known bad template-echo outputs."""
        lowered = text.lower()
        bad_signals = [
            "full article markdown",
            "<title>",
            "<full article",
            "full article - full article",
        ]
        return any(signal in lowered for signal in bad_signals)

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
                temperature=0.8,
                top_p=0.9,
                top_k=50,
                repetition_penalty=1.2,
                no_repeat_ngram_size=4,
                length_penalty=1.0,
                **length_params,
            )

        decoded = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)

        # Retry once with stricter decoding if model echoes template placeholders.
        if self._looks_like_placeholder_output(decoded):
            strict_prompt = (
                prompt
                + "\n\nFinal reminder: output real content only. Never output placeholders or template words."
            )
            strict_inputs = self.tokenizer(
                strict_prompt,
                return_tensors="pt",
                truncation=True,
                max_length=2048,
            ).to(self.device)
            with torch.no_grad():
                output_ids = self.model.generate(
                    **strict_inputs,
                    do_sample=False,
                    num_beams=5,
                    repetition_penalty=1.25,
                    length_penalty=1.1,
                    no_repeat_ngram_size=4,
                    **length_params,
                )
            decoded = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)

        title, article = self._parse_output(decoded)

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
