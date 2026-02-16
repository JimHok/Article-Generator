---
base_model: google/flan-t5-base
library_name: peft
tags:
- base_model:adapter:google/flan-t5-base
- lora
- transformers
---
# Article Generator LoRA Adapter (FLAN-T5 Base)

This repository contains a PEFT LoRA adapter fine-tuned for article-style text generation using `google/flan-t5-base`.

## Model Details

### Model Description

This model is an instruction-tuned adapter intended to improve article generation quality for structured business/insight content. It is designed to run by loading LoRA weights on top of the FLAN-T5 base model.

- **Developed by:** Project team (local prototype)
- **Funded by:** Self-funded / internal prototype work
- **Shared by:** Not publicly published by default
- **Model type:** PEFT LoRA adapter for Seq2Seq generation
- **Language(s):** English
- **License:** Follows base-model and dataset license terms; verify before redistribution
- **Finetuned from model:** `google/flan-t5-base`

### Model Sources

- **Base model:** https://huggingface.co/google/flan-t5-base
- **Training dataset:** https://huggingface.co/datasets/fabiochiu/medium-articles

## Uses

### Direct Use

- Generate article drafts from structured prompts.
- Produce long-form text with controllable topic/category/industry/tone cues.

### Downstream Use

- Backend generation service (e.g., FastAPI endpoint).
- Internal content ideation assistant for business/marketing teams.

### Out-of-Scope Use

- High-stakes factual domains without human verification.
- Legal, medical, financial advice generation.
- Identity-sensitive, harmful, or deceptive content workflows.

## Bias, Risks, and Limitations

- May hallucinate facts or fabricate specifics.
- Reflects biases present in base model + fine-tuning data.
- Quality varies by prompt clarity and topic distribution.
- Not a replacement for editorial review.

### Recommendations

- Always perform human review before publishing.
- Add retrieval/grounding if factual precision is required.
- Use moderation and policy filters for production deployments.

## How to Get Started with the Model

```python
from peft import PeftConfig, PeftModel
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

adapter_path = "model/article-generator-flan-t5-lora"
peft_config = PeftConfig.from_pretrained(adapter_path)

base = AutoModelForSeq2SeqLM.from_pretrained(peft_config.base_model_name_or_path)
model = PeftModel.from_pretrained(base, adapter_path)
tokenizer = AutoTokenizer.from_pretrained(adapter_path)

prompt = "Write an insight-driven article on retail digital transformation."
inputs = tokenizer(prompt, return_tensors="pt")
out = model.generate(**inputs, max_new_tokens=512)
print(tokenizer.decode(out[0], skip_special_tokens=True))
```

## Training Details

### Training Data

- **Dataset:** `fabiochiu/medium-articles`
- **Columns used:** `title`, `text`, `tags`
- **Row cap:** first 10000 rows via streaming + `take(10000)`

### Training Procedure

#### Preprocessing

- Normalize text whitespace.
- Normalize and deduplicate tags.
- Build instruction-style `source` prompt from title/category/industry/tags/keywords.
- Use article `text` as target output.

#### Training Hyperparameters

Below are the tuning parameters used in `training/train.py` and why they were selected.

**Core fine-tuning settings**

- **Base model:** `google/flan-t5-base`Reason: strong instruction-following quality with manageable resource usage.
- **Adapter type:** LoRA (`peft`)Reason: parameter-efficient fine-tuning to reduce VRAM and training time while preserving base model knowledge.
- **Task type:** `SEQ_2_SEQ_LM`Reason: FLAN-T5 is an encoder-decoder architecture and needs seq2seq adaptation.
- **LoRA rank (`r`) = 16**Reason: balanced adapter capacity; larger values increase capacity but cost more memory/compute.
- **LoRA alpha = 32**Reason: provides moderate adapter scaling (`alpha/r = 2`) to improve learning signal without aggressive updates.
- **LoRA dropout = 0.1**Reason: regularization to reduce overfitting on a relatively small capped dataset.
- **Target modules = [`q`, `v`]**
  Reason: adapting attention query/value projections is a common efficient strategy with good quality-cost tradeoff.

**Training loop settings**

- **Dataset name:** `fabiochiu/medium-articles`Reason: article-style domain data aligned with long-form generation goals.
- **Dataset split:** `train`Reason: prototype setup uses the main train split for adaptation.
- **Max records (default):** `10000` with streaming and `take(10000)`Reason: controls data volume and runtime while avoiding full dataset download.
- **Epochs (default):** `100`Reason: allows deeper adaptation; should be tuned down/up based on validation quality and overfitting signs.
- **Batch size:** `2`Reason: fits common single-GPU memory limits for FLAN-T5 + LoRA.
- **Learning rate:** `2e-4`Reason: common LoRA starting point; fast enough for adapter learning without severe instability.
- **Save strategy:** `no` (default)Reason: faster prototype runs and lower disk use; can switch to `epoch` for checkpointing.
- **Logging steps:** `5`Reason: frequent progress visibility during short/medium runs.
- **Precision:** auto-selected (`bf16` if supported, else `fp16` on CUDA)Reason: speed and memory efficiency on GPU while maintaining numeric stability.
- **CPU fallback:** disabled by default (`--allow_cpu` to enable)
  Reason: this training flow is intended for GPU-first execution.

**Tokenization settings**

- **Source max length:** `768`
- **Target max length:** `768`
  Reason: keeps memory bounded while preserving enough context for article-like samples.

**Generation defaults saved into adapter config**

- **do_sample:** `True`
- **temperature:** `0.8`
- **top_p:** `0.9`
- **top_k:** `50`
- **no_repeat_ngram_size:** `4`
- **repetition_penalty:** `1.2`
- **length_penalty:** `1.0`
- **min_new_tokens:** `80`
- **max_new_tokens:** `220`

Reason: these defaults favor coherent but non-repetitive long-form output, with moderate creativity and bounded response length.

## Evaluation

### Testing Data, Factors & Metrics

- **Testing data:** Not separately curated in this prototype.
- **Factors:** Prompt diversity by topic/category/industry.
- **Metrics:** Manual quality checks (structure, relevance, coherence) in current setup.

### Results

No formal benchmark scores are reported in this prototype card.

#### Summary

Adapter is suitable for draft-generation workflows and should be paired with editorial review.

## Technical Specifications

### Model Architecture and Objective

- Encoder-decoder transformer (`FLAN-T5`) with LoRA adapters.
- Objective: improve article-generation behavior for structured prompts.

### Compute Infrastructure

#### Hardware

- CUDA-capable NVIDIA GPU (project validated on RTX 40-series class GPU).

#### Software

- Python 3.13
- PyTorch (CUDA build)
- Transformers
- PEFT
- Datasets

## Citation

If you use this adapter, cite the base model and dataset:

- FLAN-T5: https://huggingface.co/google/flan-t5-base
- Medium Articles dataset: https://huggingface.co/datasets/fabiochiu/medium-articles

## Model Card Authors

Project maintainers.

## Model Card Contact

Open an issue or contact the project maintainer through the repository communication channel.

### Framework versions

- PEFT 0.17.0
