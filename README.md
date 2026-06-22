# Fabrix Adapter Layer

This service acts as an intermediate adapter layer that converts between the standard OpenAI API format and the client's Fabrix LLM request style. It relays the three model APIs the OpsGuardian backend uses: chat completions, embeddings, and reranking.

## Features
- OpenAI-compatible endpoints:
  - `POST /v1/chat/completions` (non-streaming **and** `stream=True` Server-Sent Events)
  - `POST /v1/embeddings`
  - `POST /score` (reranking, `text_1` / `text_2` format)
  - `GET /health`
- Chat requests forward `temperature`, `top_p`, `max_tokens`, `tools`, `tool_choice`, `response_format`, and `stop`; responses preserve `tool_calls` and `reasoning_content`.
- Uses environment variables for configuration (see `.env.example`)
- Requires a configured Fabrix backend; returns a clear 503 error if not available.

> **⚠️ Warning**: This program is intended for testing only. It has not been reviewed for customer security policy requirements and must **not** be used in production environments.

## Setup
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Copy `.env.example` to `.env` and fill in your values:
   - `FABRIX_BASE_URL`
   - `FABRIX_API_KEY`
   - `FABRIX_CLIENT_KEY`
   - `FABRIX_MODEL` (chat model id)
   - `FABRIX_EMBEDDING_MODEL` (default `bge-m3`)
   - `FABRIX_RERANK_MODEL` (default `bge-reranker-v2-m3`)
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
- `POST /v1/chat/completions` – OpenAI Chat Completions API. Set `"stream": true` to receive an SSE stream of `chat.completion.chunk` objects terminated by `data: [DONE]`.
- `POST /v1/embeddings` – OpenAI Embeddings API. `input` accepts a string or an array of strings; returns `data[].embedding`.
- `POST /score` – reranking. Body `{"model", "text_1", "text_2": [...]}`; returns `data[].score` aligned to `text_2` order.
- `GET /health` – liveness and configuration status.

## Testing
Run unit tests with:
```bash
pytest
```