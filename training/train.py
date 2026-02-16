"""Notebook-aligned trainer script for article generation with FLAN-T5 + LoRA."""

import argparse
import json
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, cast

from datasets import Dataset, load_dataset
from peft import LoraConfig, TaskType, get_peft_model
import torch
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)


@dataclass
class TrainConfig:
    model_name: str
    output_dir: str
    dataset_name: str
    dataset_split: str
    max_records: int
    epochs: float
    batch_size: int
    lr: float
    save_strategy: str
    logging_steps: int
    allow_cpu: bool
    gen_do_sample: bool
    gen_temperature: float
    gen_top_p: float
    gen_top_k: int
    gen_no_repeat_ngram_size: int
    gen_repetition_penalty: float
    gen_length_penalty: float
    gen_min_new_tokens: int
    gen_max_new_tokens: int


def parse_args() -> TrainConfig:
    """Parse and validate CLI arguments for notebook-aligned training."""
    parser = argparse.ArgumentParser(
        description="Fine-tune FLAN-T5 with LoRA for article generation."
    )

    model_group = parser.add_argument_group("Model")
    model_group.add_argument("--model_name", type=str, default="google/flan-t5-base")
    model_group.add_argument(
        "--output_dir", type=str, default="artifacts/article-generator-flan-t5-lora"
    )

    data_group = parser.add_argument_group("Dataset")
    data_group.add_argument(
        "--dataset_name", type=str, default="fabiochiu/medium-articles"
    )
    data_group.add_argument("--dataset_split", type=str, default="train")
    data_group.add_argument("--max_records", type=int, default=10000)

    train_group = parser.add_argument_group("Training")
    train_group.add_argument("--epochs", type=float, default=100)
    train_group.add_argument("--batch_size", type=int, default=2)
    train_group.add_argument("--lr", type=float, default=2e-4)
    train_group.add_argument("--save_strategy", type=str, default="no")
    train_group.add_argument("--logging_steps", type=int, default=5)
    train_group.add_argument(
        "--allow_cpu",
        action="store_true",
        help="Allow CPU fallback if CUDA is unavailable.",
    )

    generation_group = parser.add_argument_group("Generation Defaults")
    generation_group.add_argument(
        "--gen_do_sample",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable sampling at generation time (use --no-gen_do_sample to disable).",
    )
    generation_group.add_argument("--gen_temperature", type=float, default=0.8)
    generation_group.add_argument("--gen_top_p", type=float, default=0.9)
    generation_group.add_argument("--gen_top_k", type=int, default=50)
    generation_group.add_argument("--gen_no_repeat_ngram_size", type=int, default=4)
    generation_group.add_argument("--gen_repetition_penalty", type=float, default=1.2)
    generation_group.add_argument("--gen_length_penalty", type=float, default=1.0)
    generation_group.add_argument("--gen_min_new_tokens", type=int, default=80)
    generation_group.add_argument("--gen_max_new_tokens", type=int, default=220)

    args = parser.parse_args()

    if args.max_records < 1:
        parser.error("--max_records must be >= 1")
    if args.batch_size < 1:
        parser.error("--batch_size must be >= 1")
    if args.epochs <= 0:
        parser.error("--epochs must be > 0")
    if args.lr <= 0:
        parser.error("--lr must be > 0")
    if args.gen_min_new_tokens < 1:
        parser.error("--gen_min_new_tokens must be >= 1")
    if args.gen_max_new_tokens < args.gen_min_new_tokens:
        parser.error("--gen_max_new_tokens must be >= --gen_min_new_tokens")
    if not 0 <= args.gen_top_p <= 1:
        parser.error("--gen_top_p must be in [0, 1]")

    return TrainConfig(**vars(args))


def _normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "")).strip()


def _normalize_tags(raw_tags: Any) -> list[str]:
    if raw_tags is None:
        return []
    if isinstance(raw_tags, list):
        tags = [_normalize_text(str(tag)) for tag in raw_tags]
    else:
        text_tags = _normalize_text(str(raw_tags))
        tags = [
            _normalize_text(part)
            for part in re.split(r"[,|/#]+", text_tags)
            if _normalize_text(part)
        ]

    unique_tags = []
    seen = set()
    for tag in tags:
        lowered = tag.lower()
        if tag and lowered not in seen:
            unique_tags.append(tag)
            seen.add(lowered)
    return unique_tags


def _extract_keywords(text: str, top_k: int = 5) -> list[str]:
    stopwords = {
        "the",
        "and",
        "for",
        "with",
        "that",
        "this",
        "from",
        "are",
        "was",
        "were",
        "has",
        "have",
        "had",
        "will",
        "about",
        "into",
        "your",
        "their",
        "they",
        "them",
    }
    tokens = re.findall(r"[a-zA-Z][a-zA-Z\-]{2,}", text.lower())
    freq = Counter(tok for tok in tokens if tok not in stopwords)
    return [word for word, _ in freq.most_common(top_k)]


def _length_from_text(text: str) -> str:
    word_count = len(text.split())
    if word_count < 700:
        return "short"
    if word_count < 1500:
        return "medium"
    if word_count < 3000:
        return "long"
    return "very long"


def _get_column_names(dataset: Any) -> list[str]:
    column_names = getattr(dataset, "column_names", [])
    if isinstance(column_names, dict):
        return list(next(iter(column_names.values()), []))
    if isinstance(column_names, (list, tuple)):
        return list(column_names)
    return []


def format_medium_record(example: dict[str, Any]) -> dict[str, str]:
    title = _normalize_text(example.get("title", "")) or "Untitled Article"
    text = _normalize_text(example.get("text", ""))
    tags = _normalize_tags(example.get("tags"))
    keywords = _extract_keywords(text)
    target_length = _length_from_text(text)

    topic_category = tags[0] if len(tags) > 0 else "General"
    industry = tags[1] if len(tags) > 1 else topic_category
    tags_line = ", ".join(tags[:8]) if tags else "General"
    keywords_text = ", ".join(keywords) if keywords else "business strategy"

    input_text = (
        f"Topic: {title}; "
        f"Topic Category: {topic_category}; "
        f"Industry: {industry}; "
        f"Target length: {target_length}; "
        f"Tags: {tags_line}; "
        f"Keywords: {keywords_text}"
    )

    return {
        "source": (
            "Instruction: Write an insight-driven business article body based on the given title, category, industry, and tags.\n"
            f"Title: {title}\n"
            f"Input: {input_text}\n"
            "Output (JSON):"
        ),
        "target": json.dumps(
            {"Title": title, "Article": text},
            ensure_ascii=False,
        ),
    }


def _load_training_rows(config: TrainConfig) -> list[dict[str, Any]]:
    dataset_name = config.dataset_name.lstrip("/")
    dataset_stream = load_dataset(
        dataset_name,
        split=config.dataset_split,
        streaming=True,
    )

    iterable_dataset: Iterable[dict[str, Any]]
    if hasattr(dataset_stream, "take"):
        iterable_dataset = cast(Iterable[dict[str, Any]], dataset_stream)
    elif isinstance(dataset_stream, dict) and config.dataset_split in dataset_stream:
        iterable_dataset = cast(
            Iterable[dict[str, Any]], dataset_stream[config.dataset_split]
        )
    else:
        raise ValueError(
            f"Unable to access split '{config.dataset_split}' from dataset '{dataset_name}'."
        )

    effective_records = min(config.max_records, 10000)
    return list(cast(Any, iterable_dataset).take(effective_records))


def _prepare_dataset(rows: list[dict[str, Any]]) -> Dataset:
    train_data = Dataset.from_list(rows)
    required_columns = {"title", "text", "tags"}
    available_columns = set(_get_column_names(train_data))

    if not required_columns.issubset(available_columns):
        raise ValueError(
            "Dataset schema mismatch. Expected columns: ['title', 'text', 'tags']. "
            f"Got: {sorted(available_columns)}"
        )

    raw_column_names = _get_column_names(train_data)
    return cast(
        Dataset,
        train_data.map(format_medium_record, remove_columns=raw_column_names),
    )


def _build_tokenized_dataset(train_data: Dataset, tokenizer: Any) -> Dataset:
    def preprocess(batch: dict[str, list[str]]) -> dict[str, Any]:
        model_inputs = tokenizer(
            batch["source"],
            max_length=768,
            truncation=True,
        )
        labels = tokenizer(
            text_target=batch["target"],
            max_length=768,
            truncation=True,
        )
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    formatted_column_names = _get_column_names(train_data)
    return cast(
        Dataset,
        train_data.map(preprocess, batched=True, remove_columns=formatted_column_names),
    )


def _configure_model_and_tokenizer(config: TrainConfig) -> tuple[Any, Any]:
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(config.model_name)

    lora_config = LoraConfig(
        task_type=TaskType.SEQ_2_SEQ_LM,
        r=16,
        lora_alpha=32,
        lora_dropout=0.1,
        target_modules=["q", "v"],
    )
    model = get_peft_model(model, lora_config)

    generation_config = cast(Any, model).generation_config
    generation_config.do_sample = config.gen_do_sample
    generation_config.temperature = config.gen_temperature
    generation_config.top_p = config.gen_top_p
    generation_config.top_k = config.gen_top_k
    generation_config.no_repeat_ngram_size = config.gen_no_repeat_ngram_size
    generation_config.repetition_penalty = config.gen_repetition_penalty
    generation_config.length_penalty = config.gen_length_penalty
    generation_config.min_new_tokens = config.gen_min_new_tokens
    generation_config.max_new_tokens = config.gen_max_new_tokens

    return model, tokenizer


def main() -> None:
    """Run LoRA fine-tuning and persist adapter/tokenizer artifacts."""
    config = parse_args()
    Path(config.output_dir).mkdir(parents=True, exist_ok=True)

    if not torch.cuda.is_available() and not config.allow_cpu:
        raise RuntimeError(
            "CUDA GPU not available. Use --allow_cpu if you want CPU fallback."
        )

    rows = _load_training_rows(config)
    train_data = _prepare_dataset(rows)

    model, tokenizer = _configure_model_and_tokenizer(config)
    tokenized = _build_tokenized_dataset(train_data, tokenizer)
    collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)

    use_cuda = torch.cuda.is_available()
    use_bf16 = use_cuda and torch.cuda.is_bf16_supported()
    use_fp16 = use_cuda and not use_bf16

    train_args = Seq2SeqTrainingArguments(
        output_dir=config.output_dir,
        learning_rate=config.lr,
        per_device_train_batch_size=config.batch_size,
        num_train_epochs=config.epochs,
        logging_steps=config.logging_steps,
        save_strategy=config.save_strategy,
        no_cuda=not use_cuda,
        fp16=use_fp16,
        bf16=use_bf16,
        dataloader_pin_memory=use_cuda,
        report_to="none",
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=train_args,
        train_dataset=tokenized,
        data_collator=collator,
    )

    trainer.train()
    trainer.save_model(config.output_dir)
    model.save_pretrained(config.output_dir)
    cast(Any, model).generation_config.save_pretrained(config.output_dir)
    tokenizer.save_pretrained(config.output_dir)

    print(f"Training complete. Model saved to {config.output_dir}")


if __name__ == "__main__":
    main()
