# Fabrix Adapter Layer

This service acts as an intermediate adapter layer that converts between the OpenAI-compatible contracts used by OpsGuardian and the customer/Fabrix upstream services. It relays the three model APIs OpsGuardian uses: chat completions, embeddings, and reranking.

## Features
- OpenAI-compatible endpoints:
   - `POST /v1/chat/completions` (non-streaming is the customer validation scope; streaming is not validated in this cycle)
  - `POST /v1/embeddings`
  - `POST /score` (reranking, `text_1` / `text_2` format)
  - `GET /health`
- Chat requests forward `temperature`, `top_p`, `max_tokens`, `tools`, `tool_choice`, `response_format`, and `stop`; responses preserve `tool_calls` and `reasoning_content`.
- Embedding accepts both OpenAI-style `input` and OpsG splitter-style `sentences`.
- Reranking keeps the OpsG-facing `/score` contract and can translate to a customer `/v1/rerank` upstream response with sorted `results[].relevance_score`.
- Uses environment variables for configuration (see `.env.example`)
- `/health` reports liveness plus redacted chat/embedding/rerank configuration status. It does not actively call upstream services.

> **⚠️ Warning**: This program is intended for testing only. It has not been reviewed for customer security policy requirements and must **not** be used in production environments.

## Setup
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Copy `.env.example` to `.env` and fill in your values:
   - `FABRIX_CHAT_BASE_URL`
   - `FABRIX_API_KEY`
   - `FABRIX_CLIENT_KEY`
   - `FABRIX_MODEL` (chat model id)
   - `EMBEDDING_URL` (full upstream endpoint URL, for example `http://host/v1/embeddings`)
   - `EMBEDDING_MODEL` (default `bge-m3`)
   - `RERANK_URL` (full upstream endpoint URL, for example `http://host/v1/rerank`)
   - `RERANK_MODEL` (customer model name, for example `bge-reranker`)
   - `RERANK_UPSTREAM_STYLE` (`customer_v1_rerank` or `legacy_score`)
   - Optional auth/TLS: `EMBEDDING_API_KEY`, `RERANK_API_KEY`, `EMBEDDING_CERT`, `EMBEDDING_KEY`, `RERANK_CERT`, `RERANK_KEY`, `EMBEDDING_VERIFY_TLS`, `RERANK_VERIFY_TLS`
3. Run the adapter:
   ```bash
   python fabrix_adapter.py
   ```
   or with Docker:
   ```bash
   docker build -t fabrix-adapter .
   docker run -p 8000:8000 --env-file .env fabrix-adapter
   ```

## Endpoints
- `POST /v1/chat/completions` – OpenAI Chat Completions API for non-streaming OpsG calls.
- `POST /v1/embeddings` – OpenAI Embeddings API. `input` accepts a string or an array of strings. `sentences` accepts an array of strings for the OpsG semantic splitter path. Returns `data[].embedding` aligned to input order.
- `POST /score` – OpsG-facing reranking. Body `{"model", "text_1", "text_2": [...]}`; returns `data[].score` aligned to `text_2` order. The adapter does not expose inbound `/v1/rerank`; `/v1/rerank` is only an upstream customer endpoint selected by `RERANK_URL` + `RERANK_UPSTREAM_STYLE=customer_v1_rerank`.
- `GET /health` – liveness and configuration status.

## Customer Contract Test
Run the standalone customer-site contract test with:
```bash
python test_adapter_contracts.py --group all
```
Useful environment variables:
```bash
export ADAPTER_BASE_URL="http://127.0.0.1:8000"
export OPENAI_API_KEY="example1111"
export MODEL_NAME="openai.gpt-oss-120b-1:0"
export EXPECTED_EMBEDDING_DIM=1024
```
Groups can be rerun independently with `--group health`, `--group chat`, `--group embedding`, or `--group rerank`. The script writes `adapter_contract_report.md` before exiting. Any transport or semantic validation failure returns a non-zero exit code.

## Testing
Run unit tests with:
```bash
python -m pytest tests/test_adapter.py
```