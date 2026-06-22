import json
import os
import random
import time
from typing import Any, Dict, List, Optional, Union

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from dotenv import load_dotenv

# Import LangChain message types (always available if langchain-core is installed)
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

load_dotenv()  # take environment variables from .env

app = FastAPI(title="Fabrix Adapter Layer")

# Configuration
FABRIX_BASE_URL = os.getenv("FABRIX_BASE_URL", "")
FABRIX_API_KEY = os.getenv("FABRIX_API_KEY", "")
FABRIX_CLIENT_KEY = os.getenv("FABRIX_CLIENT_KEY", "")
FABRIX_MODEL = os.getenv("FABRIX_MODEL", "")
# Embedding / reranking model ids served by Fabrix (defaults match the OpsG backend).
FABRIX_EMBEDDING_MODEL = os.getenv("FABRIX_EMBEDDING_MODEL", "bge-m3")
FABRIX_RERANK_MODEL = os.getenv("FABRIX_RERANK_MODEL", "bge-reranker-v2-m3")

# Try to import LangChain Fabrix integration
try:
    from langchain_fabrix import ChatFabrix
    HAS_FABRIX = True
except Exception:
    HAS_FABRIX = False
    ChatFabrix = None  # type: ignore

# Optional LangChain Fabrix embeddings integration. Falls back to HTTP when absent.
try:
    from langchain_fabrix import FabrixEmbeddings
    HAS_FABRIX_EMBEDDINGS = True
except Exception:
    HAS_FABRIX_EMBEDDINGS = False
    FabrixEmbeddings = None  # type: ignore

# Optional LangChain Fabrix reranker integration. Falls back to HTTP when absent.
try:
    from langchain_fabrix import FabrixReranker
    HAS_FABRIX_RERANKER = True
except Exception:
    HAS_FABRIX_RERANKER = False
    FabrixReranker = None  # type: ignore


# ---------------------------------------------------------------------------
# OpenAI-compatible request / response models
# ---------------------------------------------------------------------------
class ChatMessage(BaseModel):
    role: str
    content: Optional[str] = None
    # Pass-through fields used by tool-calling / reasoning models.
    tool_calls: Optional[List[Dict[str, Any]]] = None
    reasoning_content: Optional[str] = None


class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[ChatMessage]
    temperature: Optional[float] = 0.7
    top_p: Optional[float] = None
    max_tokens: Optional[int] = None
    stream: Optional[bool] = False
    # Pass-through OpenAI fields used by the OpsG backend.
    tools: Optional[List[Dict[str, Any]]] = None
    tool_choice: Optional[Union[str, Dict[str, Any]]] = None
    response_format: Optional[Dict[str, Any]] = None
    stop: Optional[Union[str, List[str]]] = None


class ChatCompletionResponseChoice(BaseModel):
    index: int
    message: ChatMessage
    finish_reason: str = "stop"


class ChatCompletionResponse(BaseModel):
    id: str = Field(default_factory=lambda: f"chatcmpl-{random.randint(100000, 999999)}")
    object: str = "chat.completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: List[ChatCompletionResponseChoice]
    usage: dict = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}


class EmbeddingRequest(BaseModel):
    model: Optional[str] = None
    input: Union[str, List[str]]


class EmbeddingData(BaseModel):
    index: int
    object: str = "embedding"
    embedding: List[float]


class EmbeddingResponse(BaseModel):
    id: str = Field(default_factory=lambda: f"embd-{random.randint(100000, 999999)}")
    object: str = "list"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    data: List[EmbeddingData]
    usage: dict = {"prompt_tokens": 0, "total_tokens": 0, "completion_tokens": 0}


class RerankRequest(BaseModel):
    model: Optional[str] = None
    text_1: str
    text_2: List[str]


class RerankData(BaseModel):
    index: int
    object: str = "score"
    score: float


class RerankResponse(BaseModel):
    id: str = Field(default_factory=lambda: f"score-{random.randint(100000, 999999)}")
    object: str = "list"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    data: List[RerankData]
    usage: dict = {"prompt_tokens": 0, "total_tokens": 0, "completion_tokens": 0}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _convert_to_langchain_message(msg: ChatMessage):
    role = msg.role.lower()
    content = msg.content or ""
    if role == "system":
        return SystemMessage(content=content)
    elif role == "assistant":
        return AIMessage(content=content)
    else:  # default to user
        return HumanMessage(content=content)


def _chat_model_kwargs(request: ChatCompletionRequest) -> Dict[str, Any]:
    """Collect optional OpenAI parameters to forward to the Fabrix chat model."""
    model_kwargs: Dict[str, Any] = {}
    if request.top_p is not None:
        model_kwargs["top_p"] = request.top_p
    if request.tools is not None:
        model_kwargs["tools"] = request.tools
    if request.tool_choice is not None:
        model_kwargs["tool_choice"] = request.tool_choice
    if request.response_format is not None:
        model_kwargs["response_format"] = request.response_format
    if request.stop is not None:
        model_kwargs["stop"] = request.stop
    return model_kwargs


def _build_chat_fabrix(request: ChatCompletionRequest, streaming: bool):
    return ChatFabrix(
        model_id=FABRIX_MODEL,
        openapi_token=FABRIX_API_KEY,
        fabrix_client=FABRIX_CLIENT_KEY,
        fabrix_base_url=FABRIX_BASE_URL,
        temperature=request.temperature if request.temperature is not None else 0.7,
        max_tokens=request.max_tokens,
        streaming=streaming,
        model_kwargs=_chat_model_kwargs(request),
    )


def _require_fabrix_configured():
    if not (HAS_FABRIX and FABRIX_BASE_URL and FABRIX_API_KEY and FABRIX_CLIENT_KEY and FABRIX_MODEL):
        raise HTTPException(
            status_code=503,
            detail="Fabrix backend not available or not configured. Please set FABRIX_BASE_URL, FABRIX_API_KEY, FABRIX_CLIENT_KEY, FABRIX_MODEL and ensure langchain-fabrix is installed.",
        )


def _require_fabrix_endpoint_configured():
    """Embedding / reranking only need connection config, not the chat model id."""
    if not (FABRIX_BASE_URL and FABRIX_API_KEY and FABRIX_CLIENT_KEY):
        raise HTTPException(
            status_code=503,
            detail="Fabrix backend not configured. Please set FABRIX_BASE_URL, FABRIX_API_KEY, FABRIX_CLIENT_KEY.",
        )


def _extract_reasoning(result) -> Optional[str]:
    meta = getattr(result, "additional_kwargs", None)
    reasoning = meta.get("reasoning_content") if isinstance(meta, dict) else None
    if reasoning is None:
        response_meta = getattr(result, "response_metadata", None)
        if isinstance(response_meta, dict):
            reasoning = response_meta.get("reasoning_content")
    return reasoning if isinstance(reasoning, str) else None


def _extract_tool_calls(result) -> Optional[List[Dict[str, Any]]]:
    meta = getattr(result, "additional_kwargs", None)
    tool_calls = meta.get("tool_calls") if isinstance(meta, dict) else None
    return tool_calls if isinstance(tool_calls, list) and tool_calls else None


# ---------------------------------------------------------------------------
# Chat completions
# ---------------------------------------------------------------------------
def _stream_chat_completions(request: ChatCompletionRequest, lc_messages):
    completion_id = f"chatcmpl-{random.randint(100000, 999999)}"
    created = int(time.time())

    def event_stream():
        try:
            chat = _build_chat_fabrix(request, streaming=True)
            # First chunk advertises the assistant role.
            first = {
                "id": completion_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": request.model,
                "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}],
            }
            yield f"data: {json.dumps(first)}\n\n"

            for chunk in chat.stream(lc_messages):
                content = getattr(chunk, "content", None)
                if not content:
                    continue
                payload = {
                    "id": completion_id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": request.model,
                    "choices": [{"index": 0, "delta": {"content": content}, "finish_reason": None}],
                }
                yield f"data: {json.dumps(payload)}\n\n"

            final = {
                "id": completion_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": request.model,
                "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
            }
            yield f"data: {json.dumps(final)}\n\n"
            yield "data: [DONE]\n\n"
        except Exception as e:  # surface Fabrix errors inside the stream
            err = {"error": {"message": f"Failed to communicate with Fabrix service: {str(e)}", "type": "server_error"}}
            yield f"data: {json.dumps(err)}\n\n"
            yield "data: [DONE]\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    _require_fabrix_configured()

    # Convert messages to LangChain format
    lc_messages = [_convert_to_langchain_message(m) for m in request.messages]

    if request.stream:
        return _stream_chat_completions(request, lc_messages)

    try:
        chat = _build_chat_fabrix(request, streaming=False)
        # LangChain chat models have .invoke method returning AIMessage
        result = chat.invoke(lc_messages)
        content = result.content if hasattr(result, "content") else str(result)
        reasoning = _extract_reasoning(result)
        tool_calls = _extract_tool_calls(result)
        # Estimate token usage (simple approximation)
        prompt_tokens = sum(len((m.content or "").split()) for m in request.messages)
        completion_tokens = len((content or "").split())
        total_tokens = prompt_tokens + completion_tokens
        response = ChatCompletionResponse(
            model=request.model,
            choices=[ChatCompletionResponseChoice(
                index=0,
                message=ChatMessage(
                    role="assistant",
                    content=content,
                    tool_calls=tool_calls,
                    reasoning_content=reasoning,
                ),
                finish_reason="tool_calls" if tool_calls else "stop",
            )],
            usage={"prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens, "total_tokens": total_tokens},
        )
        return response
    except Exception as e:
        # Any error communicating with Fabrix results in 503
        raise HTTPException(
            status_code=503,
            detail=f"Failed to communicate with Fabrix service: {str(e)}",
        )


# ---------------------------------------------------------------------------
# Embeddings
# ---------------------------------------------------------------------------
def _fabrix_embed(texts: List[str], model: str) -> List[List[float]]:
    """Return one embedding vector per input text via Fabrix."""
    if HAS_FABRIX_EMBEDDINGS:
        embedder = FabrixEmbeddings(
            model_id=model,
            openapi_token=FABRIX_API_KEY,
            fabrix_client=FABRIX_CLIENT_KEY,
            fabrix_base_url=FABRIX_BASE_URL,
        )
        return embedder.embed_documents(texts)

    # HTTP fallback: OpenAI-compatible /v1/embeddings on the Fabrix gateway.
    import httpx

    url = f"{FABRIX_BASE_URL.rstrip('/')}/v1/embeddings"
    headers = {
        "Authorization": f"Bearer {FABRIX_API_KEY}",
        "Content-Type": "application/json",
    }
    if FABRIX_CLIENT_KEY:
        headers["fabrix-client"] = FABRIX_CLIENT_KEY
    with httpx.Client(timeout=60.0) as http:
        resp = http.post(url, headers=headers, json={"model": model, "input": texts})
        resp.raise_for_status()
        data = resp.json()
    return [item["embedding"] for item in data["data"]]


@app.post("/v1/embeddings")
async def embeddings(request: EmbeddingRequest):
    _require_fabrix_endpoint_configured()

    model = request.model or FABRIX_EMBEDDING_MODEL
    texts = [request.input] if isinstance(request.input, str) else list(request.input)

    try:
        vectors = _fabrix_embed(texts, model)
        data = [EmbeddingData(index=i, embedding=vec) for i, vec in enumerate(vectors)]
        prompt_tokens = sum(len(t.split()) for t in texts)
        return EmbeddingResponse(
            model=model,
            data=data,
            usage={"prompt_tokens": prompt_tokens, "total_tokens": prompt_tokens, "completion_tokens": 0},
        )
    except Exception as e:
        raise HTTPException(
            status_code=503,
            detail=f"Failed to communicate with Fabrix service: {str(e)}",
        )


# ---------------------------------------------------------------------------
# Reranking (kept at /score with text_1/text_2 to match the OpsG backend)
# ---------------------------------------------------------------------------
def _fabrix_rerank(query: str, documents: List[str], model: str) -> List[float]:
    """Return one relevance score per document via Fabrix."""
    if HAS_FABRIX_RERANKER:
        reranker = FabrixReranker(
            model_id=model,
            openapi_token=FABRIX_API_KEY,
            fabrix_client=FABRIX_CLIENT_KEY,
            fabrix_base_url=FABRIX_BASE_URL,
        )
        return reranker.score(query, documents)

    # HTTP fallback: /score endpoint on the Fabrix gateway.
    import httpx

    url = f"{FABRIX_BASE_URL.rstrip('/')}/score"
    headers = {
        "Authorization": f"Bearer {FABRIX_API_KEY}",
        "Content-Type": "application/json",
    }
    if FABRIX_CLIENT_KEY:
        headers["fabrix-client"] = FABRIX_CLIENT_KEY
    with httpx.Client(timeout=60.0) as http:
        resp = http.post(url, headers=headers, json={"model": model, "text_1": query, "text_2": documents})
        resp.raise_for_status()
        data = resp.json()
    return [item["score"] for item in data["data"]]


@app.post("/score")
async def score(request: RerankRequest):
    _require_fabrix_endpoint_configured()

    model = request.model or FABRIX_RERANK_MODEL
    try:
        scores = _fabrix_rerank(request.text_1, request.text_2, model)
        data = [RerankData(index=i, score=s) for i, s in enumerate(scores)]
        prompt_tokens = len(request.text_1.split()) + sum(len(t.split()) for t in request.text_2)
        return RerankResponse(
            model=model,
            data=data,
            usage={"prompt_tokens": prompt_tokens, "total_tokens": prompt_tokens, "completion_tokens": 0},
        )
    except Exception as e:
        raise HTTPException(
            status_code=503,
            detail=f"Failed to communicate with Fabrix service: {str(e)}",
        )


# Health check
@app.get("/health")
async def health():
    # Also check configuration status
    configured = bool(HAS_FABRIX and FABRIX_BASE_URL and FABRIX_API_KEY and FABRIX_CLIENT_KEY and FABRIX_MODEL)
    return {"status": "ok", "fabrix_configured": configured}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
