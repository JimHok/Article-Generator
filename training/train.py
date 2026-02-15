"""
Fine-tuning script for Jenosize article generation prototype.

Default model: google/flan-t5-base
Method: PEFT LoRA for efficient adaptation

Default dataset: ag_news (Hugging Face, publicly available)
"""

import argparse
import re
from collections import Counter
from pathlib import Path

from datasets import load_dataset
from peft import LoraConfig, TaskType, get_peft_model
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)


def parse_args():
    """Parse CLI arguments for fine-tuning configuration."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="google/flan-t5-base")
    parser.add_argument("--dataset_name", type=str, default="ag_news")
    parser.add_argument("--dataset_split", type=str, default="train")
    parser.add_argument("--max_records", type=int, default=5000)
    parser.add_argument("--output_dir", type=str, default="artifacts/jenosize-flan-t5-lora")
    parser.add_argument("--epochs", type=float, default=3)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=2e-4)
    return parser.parse_args()


def _normalize_text(text: str) -> str:
    """Normalize whitespace for cleaner prompts and labels."""
    return re.sub(r"\s+", " ", (text or "")).strip()


def _extract_keywords(text: str, top_k: int = 5):
    """Extract simple frequency-based keywords from source text."""
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


def format_ag_news_record(example):
    """Transform AG News row into instruction-tuning source/target fields.

    AG News labels:
    - 0: World
    - 1: Sports
    - 2: Business
    - 3: Sci/Tech
    """
    label = int(example["label"])
    text = _normalize_text(example["text"])
    title = _normalize_text(text.split(" - ")[0]) if " - " in text else _normalize_text(text[:120])
    keywords = _extract_keywords(text)

    if label == 2:
        topic_category = "Business Trends"
        industry = "Cross-Industry"
        audience = "CMO, Strategy Lead, Growth Team"
    else:
        topic_category = "Technology Trends"
        industry = "Technology"
        audience = "Product, Innovation, and Marketing Leaders"

    input_text = (
        f"Topic: {title}; "
        f"Topic Category: {topic_category}; "
        f"Industry: {industry}; "
        f"Audience: {audience}; "
        f"Keywords: {', '.join(keywords) if keywords else 'business strategy'}"
    )
    output_text = (
        f"Title: {title}\n"
        "Article:\n"
        "## Why this trend matters now\n"
        f"{text}\n\n"
        "## Strategic implications\n"
        "- Reassess changing customer and market signals.\n"
        "- Align teams around clear business outcomes and KPIs.\n"
        "- Run fast experiments and scale validated ideas.\n\n"
        "## Action framework\n"
        "1. Diagnose the shift and define target audience impact.\n"
        "2. Prioritize initiatives by value, effort, and urgency.\n"
        "3. Execute, measure, and iterate in short cycles.\n"
    )
    return {
        "source": (
            "Instruction: Write an insight-driven business article with this style: start with a bold misconception or tension, define the concept briefly, present numbered misconceptions or insights with short practical examples, include a simple decision matrix/framework, keep the tone analytical but easy to read, and close with actionable recommendations plus a concise partnership-oriented call to action.\n"
            f"Input: {input_text}\n"
            "Output:"
        ),
        "target": output_text,
    }


def main():
    """Run LoRA fine-tuning and persist adapter/tokenizer artifacts."""
    args = parse_args()
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    # Use only public Hugging Face dataset and transform rows.
    dataset = load_dataset(args.dataset_name, split=args.dataset_split)
    # Keep rows most relevant to this use case (Business and Sci/Tech).
    dataset = dataset.filter(lambda row: int(row["label"]) in {2, 3})
    if args.max_records > 0:
        dataset = dataset.select(range(min(args.max_records, len(dataset))))
    dataset = dataset.map(format_ag_news_record, remove_columns=dataset.column_names)

    # Initialize base model components from Hugging Face.
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model_name)

    # Apply PEFT LoRA for parameter-efficient adaptation.
    lora_config = LoraConfig(
        task_type=TaskType.SEQ_2_SEQ_LM,
        r=16,
        lora_alpha=32,
        lora_dropout=0.1,
        target_modules=["q", "v"],
    )
    model = get_peft_model(model, lora_config)

    def preprocess(batch):
        """Tokenize source/target text into model input tensors."""
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

    # Remove raw text columns after tokenization to reduce memory usage.
    tokenized = dataset.map(preprocess, batched=True, remove_columns=dataset.column_names)
    collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)

    train_args = Seq2SeqTrainingArguments(
        output_dir=args.output_dir,
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        logging_steps=5,
        save_strategy="epoch",
        fp16=False,
        report_to="none",
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=train_args,
        train_dataset=tokenized,
        data_collator=collator,
        tokenizer=tokenizer,
    )

    # Persist adapter + tokenizer so API can load fine-tuned artifacts.
    trainer.train()
    trainer.model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    print(f"Training complete. Model saved to {args.output_dir}")


if __name__ == "__main__":
    main()
