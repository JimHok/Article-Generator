# Short Project Report: Article Generation Prototype

## 1) Approach

### Project objective

The goal was to build a practical prototype that generates business insight articles from structured inputs (topic category, industry, keywords, and optional source context), then expose that capability through an API and deployment-ready setup.

### End-to-end architecture

The implementation follows a clear pipeline:

1. **Input processing** API requests are normalized and cleaned before generation. Keyword cleanup, optional source-text enrichment, and length/tone handling are done in the preprocessing layer.
2. **Prompt construction** A controlled instruction prompt is built from normalized fields. The prompt is designed to push structured, relevant, and readable article outputs while avoiding template artifacts.
3. **Model generation** The generator loads either:

   - a base Hugging Face model, or
   - a local LoRA adapter output fine-tuned for this project.
4. **Output parsing** Generated text is parsed into API response fields (`title`, `article`) with cleanup/guard logic.
5. **Serving layer** A FastAPI app exposes `/health` and `/generate`, with typed request/response schemas.
6. **Deployment layer**
   The project supports local run, Docker CPU, Docker GPU, and scripted Docker deployment.

### Model and fine-tuning strategy

The base model is **FLAN-T5 (google/flan-t5-base)**, chosen because it performs well for instruction-following and is efficient enough for prototype iteration.

Fine-tuning uses **PEFT LoRA**, which was selected for:

- lower VRAM requirements,
- faster iteration cycles,
- reduced risk of overfitting compared with full-model tuning in a small-scale prototype.

Key training design choices include:

- Dataset source: `fabiochiu/medium-articles`
- Required columns: `title`, `text`, `tags`
- Streaming data load with capped take to control compute/runtime
- Instruction-style source-target transformation
- GPU-first training with mixed precision (`bf16` when available, else `fp16`)

### Data engineering choices

Raw dataset records are transformed into a stable training structure:

- `title` is retained as the article title signal,
- `text` is used as target article body,
- `tags` are normalized and reused as topical context (industry/category cues).

Additional derived features include:

- keyword extraction from article text,
- normalized text/tags,
- length label inferred from text length.

These transformations were implemented to make training prompts consistent and predictable, which improves generation reliability.

### API and operational delivery

The API was made production-friendly for prototype usage:

- strict request/response schemas,
- model-path based loading (base model vs adapter folder),
- health endpoint for quick diagnostics,
- Dockerized runtime for reproducibility,
- GPU-ready Docker option for accelerated inference.

The deployment scripts and Dockerfiles were aligned to a **uv-based** dependency workflow using `pyproject.toml` and `uv.lock`.

---

## 2) Challenges Faced

### Challenge 1: Data schema mismatch across datasets

Early training assumptions reflected a different dataset schema (`ag_news`) while the final dataset (`medium-articles`) has a different structure (`title`, `text`, `tags`). This caused mapping/removal errors during preprocessing.

**Resolution:**

- Reworked formatting logic specifically for `medium-articles`.
- Added strict schema checks before mapping.
- Updated transformation flow to ensure fields are always available.

### Challenge 2: Output quality issues (template echo / malformed responses)

At times, generated output echoed template placeholders instead of producing clean article text. This is a common issue when instruction prompts are overly rigid or parsing assumes ideal output.

**Resolution:**

- Generalized prompt constraints to reduce overfitting to template wording.
- Strengthened output parsing and normalization.
- Added structure-aware handling to reduce malformed API responses.

### Challenge 3: API contract evolution

Input keys changed over time (e.g., audience removal), requiring synchronized updates across schema, preprocessing, sample payloads, training prompts, and docs.

**Resolution:**

- performed coordinated updates to avoid drift,
- validated against static errors after each change,
- refreshed README/model docs to reflect actual behavior.

---

## 3) Potential Improvements

### A. Improve evaluation rigor

Current evaluation is mostly manual quality checking. A stronger next step is adding automatic evaluation metrics such as:

- structure compliance,
- keyword coverage,
- factual consistency (where applicable),
- readability and coherence scoring.

A small benchmark suite with fixed prompts and expected quality thresholds would improve regression tracking.

### B. Add retrieval grounding for factual reliability

The model can still hallucinate details. Integrating retrieval (RAG) from trusted business sources would improve factual accuracy and citation quality for trend-oriented content.

### C. Better prompt-control and output contracts

Move to stricter structured generation (JSON schema validation + retry policy + fallback templates) to guarantee robust API outputs under noisy model behavior.

### D. Expand and curate dataset quality

While the current dataset is workable for prototype speed, performance would benefit from curated domain-specific examples:

- business trend analysis,
- strategy frameworks,
- campaign concept writing,
- controlled style examples.

A balanced, quality-reviewed dataset typically improves consistency more than hyperparameter tuning alone.

### E. Production hardening

For deployment maturity:

- add authentication/rate limiting,
- add request/response observability (tracing + metrics),
- expose model/version metadata in health diagnostics,
- add CI checks for schema + prompt + generation regression,
- add fallback model routing when GPU is unavailable.

### F. Cost and latency optimization

For real usage:

- cache repeated prompt segments,
- tune generation token budgets by requested length,
- batch requests where possible,
- evaluate smaller distilled models for lower-latency routes.

---

## Conclusion

The prototype successfully delivers an end-to-end article-generation workflow: training, preprocessing, inference API, and containerized deployment (CPU/GPU). The implementation emphasized practical engineering trade-offs—fast iteration with LoRA, GPU-first operation, and operational reproducibility with uv + Docker. The main remaining gap is robust automated evaluation and stronger factual grounding. Addressing those areas would move the project from prototype to a reliable production-grade content platform.
