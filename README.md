# Jenosize Content Generation Prototype

A simple end-to-end prototype for generating insight articles about business trends and future ideas.

This project covers:

1. **Model Selection & Fine-Tuning** (Hugging Face + LoRA)
2. **Data Engineering Pipeline** (parameter cleaning + source enrichment)
3. **Model Deployment** (FastAPI endpoint)
4. **Documentation** (setup, training, run, and test)

---

## 1) Model Selection & Fine-Tuning

### Chosen model

- **Base model**: `google/flan-t5-base`
- **Why this model**:
  - Strong instruction-following behavior
  - Good quality-to-resource ratio for prototype work
  - Compatible with efficient fine-tuning via PEFT/LoRA

### Fine-tuning strategy

- **Method**: LoRA (`peft`) on top of FLAN-T5
- **Primary dataset source**: Hugging Face public dataset `ag_news`
- **Training script**: [training/train.py](training/train.py)

### Public dataset selection

- **Dataset**: `ag_news` (Hugging Face Datasets)
- **Selection logic for this use case**:
  - Use label `2` (**Business**) and label `3` (**Sci/Tech**)
  - Transform rows into instruction-tuning format for strategic article generation

### Dataset guidance

For better quality, expand the dataset with:

- Business trend articles
- Marketing strategy and campaign concept examples
- Industry-specific writing samples aligned with Jenosize tone

Keep all data compliant with licensing and internal usage permissions.

---

## 2) Data Engineering Pipeline

Pipeline implementation: [app/pipeline.py](app/pipeline.py)

### What it does

- Cleans and normalizes text parameters
- Cleans and deduplicates SEO keywords
- Optionally extracts source context from `source_url`
- Accepts `source_text` for document-based context
- Produces a standardized object for prompt construction

Prompt construction: [app/prompting.py](app/prompting.py)

### Supported topic variety

The payload structure supports diverse business domains:

- Technology trends
- Digital transformation
- Marketing and consumer behavior
- Industry-specific strategic content

---

## 3) API Deployment (FastAPI)

API app: [app/main.py](app/main.py)

### Endpoints

- `GET /health`: model health + model path
- `POST /generate`: generate an article from structured input parameters

Schema definitions: [app/schemas.py](app/schemas.py)
Generation engine: [app/generator.py](app/generator.py)

---

## 4) Setup & Run

### A. Install dependencies

```bash
pip install -r requirements.txt
```

### B. Fine-tune model (using Hugging Face public dataset)

```bash
python training/train.py \
  --model_name google/flan-t5-base \
  --dataset_name ag_news \
  --dataset_split train \
  --max_records 5000 \
  --output_dir artifacts/jenosize-flan-t5-lora \
  --epochs 3 \
  --batch_size 2 \
  --lr 2e-4
```

### C. Run API

```bash
set MODEL_PATH=artifacts/jenosize-flan-t5-lora
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

### D. Test API

```bash
curl -X POST "http://127.0.0.1:8000/generate" \
  -H "Content-Type: application/json" \
  -d @sample_request.json
```

---

## Example Input

See [sample_request.json](sample_request.json).

---

## Notes on Style Alignment

The prompt template is tuned to reflect a Jenosize-like style:

- Insight-first narrative
- Structured sections (problem, misconceptions/insights, framework, actions)
- Practical and strategic recommendations
- Natural SEO keyword integration

Reference article used for style calibration:

- https://www.jenosize.com/en/ideas/understand-people-and-consumer/customer-journey-misconceptions

---

## Suggested Next Improvements

- Expand dataset (100-1,000+ high-quality examples)
- Add evaluation metrics (keyword coverage, structure, readability)
- Add RAG with trusted business sources
- Add Thai language support and bilingual generation
- Containerize deployment (Docker) for production use
