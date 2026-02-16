# Article Generator Prototype

End-to-end prototype for business insight article generation using FLAN-T5 + LoRA, a FastAPI service, and Docker deployment.

## What’s in this project

1. **Model fine-tuning** with PEFT LoRA on top of `google/flan-t5-base`
2. **Data pipeline** for topic/industry/keywords/context preprocessing
3. **API deployment** with FastAPI (`/health`, `/generate`)
4. **Container deployment** for CPU and GPU with Docker

---

## Model & Data

- **Base model:** `google/flan-t5-base`
- **Fine-tuning method:** LoRA (`peft`)
- **Dataset:** `fabiochiu/medium-articles`
- **Columns used:** `title`, `text`, `tags`
- **Download strategy:** streaming + `take()`
- **Row cap:** hard-capped at **10000 rows**

Training script: [training/train.py](training/train.py)

---

## API

API app: [app/main.py](app/main.py)

Endpoints:

- `GET /health`
- `POST /generate`

Request schema: [app/schemas.py](app/schemas.py)
Prompt builder: [app/prompting.py](app/prompting.py)
Generator: [app/generator.py](app/generator.py)

Default model path (from env):

- `MODEL_PATH=model/article-generator-flan-t5-lora`

---

## Local setup (uv)

Install/sync dependencies:

```bash
uv sync
```

Run training:

```bash
uv run python training/train.py
```

Optional example with explicit args:

```bash
uv run python training/train.py \
  --model_name google/flan-t5-base \
  --dataset_name fabiochiu/medium-articles \
  --dataset_split train \
  --max_records 10000 \
  --output_dir model/article-generator-flan-t5-lora \
  --epochs 100 \
  --batch_size 2 \
  --lr 2e-4
```

Run API:

```bash
uv run uvicorn app.main:app --host 0.0.0.0 --port 8000
```

Test health:

```bash
curl http://127.0.0.1:8000/health
```

Test generation:

```bash
curl -X POST "http://127.0.0.1:8000/generate" \
  -H "Content-Type: application/json" \
  -d '{
    "topic_category": "Retail Intelligence",
    "industry": "Retail",
    "seo_keywords": ["customer journey", "personalization", "conversion"],
    "article_length": "medium",
    "tone": "insightful, strategic, clear"
  }'
```

Note: [sample_request.json](sample_request.json) currently contains a list of example payloads.

---

## Docker (uv + pyproject)

Both Dockerfiles use `uv` with `pyproject.toml` and `uv.lock`.

### CPU image

Build:

```bash
docker build -f Dockerfile -t article-generator:latest .
```

Run:

```bash
docker run --rm -p 8000:8000 --name article-generator-api article-generator:latest
```

### GPU image

Requirements:

- NVIDIA GPU driver installed
- NVIDIA Container Toolkit installed

Build:

```bash
docker build -f Dockerfile.gpu -t article-generator:gpu .
```

Run:

```bash
docker run --rm --gpus all -p 8000:8000 --name article-generator-api-gpu article-generator:gpu
```

### One-command deploy script

Script: [deploy_docker.sh](deploy_docker.sh)

```bash
./deploy_docker.sh gpu
```

```bash
./deploy_docker.sh cpu
```

Custom port:

```bash
PORT=8080 ./deploy_docker.sh gpu
```

---

## Notes

- Training is GPU-first (`--allow_cpu` for fallback).
- Generation output is parsed from JSON-like model output (`Title` and `Article`).
- Cached model loading is used in API for better performance.
