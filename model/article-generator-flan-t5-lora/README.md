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

- **Base model:** `google/flan-t5-base`
- **Adapter type:** LoRA
- **Task type:** `SEQ_2_SEQ_LM`
- **LoRA rank (`r`):** 16
- **LoRA alpha:** 32
- **LoRA dropout:** 0.1
- **Target modules:** `q`, `v`
- **Batch size:** 2
- **Learning rate:** 2e-4
- **Epochs:** user-configurable, capped at 1000
- **Precision:** bf16 if supported, else fp16 on CUDA

## Evaluation

### Testing Data, Factors & Metrics

- **Testing data:** Not separately curated in this prototype.
- **Factors:** Prompt diversity by topic/category/industry.
- **Metrics:** Manual quality checks (structure, relevance, coherence) in current setup.

### Results

No formal benchmark scores are reported in this prototype card.

#### Summary

Adapter is suitable for draft-generation workflows and should be paired with editorial review.

## Environmental Impact

Carbon emissions can be estimated with the [ML CO2 calculator](https://mlco2.github.io/impact#compute).

- **Hardware Type:** Local GPU workstation
- **Hours used:** Not systematically logged
- **Cloud Provider:** N/A (local)
- **Compute Region:** Local machine
- **Carbon Emitted:** Not measured

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
