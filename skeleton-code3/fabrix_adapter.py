import json
import logging
import os
import random
import time
import uuid
from typing import Any, Dict, List, Optional, Union

from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from dotenv import load_dotenv

# Import LangChain message types (always available if langchain-core is installed)
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
try:
    from langchain_core.messages import ToolMessage
except Exception:  # pragma: no cover - depends on langchain-core version
    ToolMessage = None  # type: ignore

load_dotenv()  # take environment variables from .env

app = FastAPI(title="Fabrix Adapter Layer")
logger = logging.getLogger(__name__)

# Configuration
# Legacy chat/Fabrix settings retained for compatibility.
FABRIX_BASE_URL = os.getenv("FABRIX_BASE_URL", "")
FABRIX_CHAT_BASE_URL = os.getenv("FABRIX_CHAT_BASE_URL", "")
FABRIX_API_KEY = os.getenv("FABRIX_API_KEY", "")
FABRIX_CLIENT_KEY = os.getenv("FABRIX_CLIENT_KEY", "")
FABRIX_MODEL = os.getenv("FABRIX_MODEL", "")

# Embedding / reranking upstream settings. OpsG-facing contracts stay unchanged;
# these values describe the customer-side servers the adapter calls.
EMBEDDING_URL = os.getenv("EMBEDDING_URL", "")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", os.getenv("FABRIX_EMBEDDING_MODEL", "bge-m3"))
EMBEDDING_API_KEY = os.getenv("EMBEDDING_API_KEY", "")
EMBEDDING_CERT = os.getenv("EMBEDDING_CERT", "")
EMBEDDING_KEY = os.getenv("EMBEDDING_KEY", "")
EMBEDDING_VERIFY_TLS = os.getenv("EMBEDDING_VERIFY_TLS", "true")
EMBEDDING_TIMEOUT = float(os.getenv("EMBEDDING_TIMEOUT", "60"))

RERANK_URL = os.getenv("RERANK_URL", "")
RERANK_MODEL = os.getenv("RERANK_MODEL", os.getenv("FABRIX_RERANK_MODEL", "bge-reranker-v2-m3"))
RERANK_API_KEY = os.getenv("RERANK_API_KEY", "")
RERANK_CERT = os.getenv("RERANK_CERT", "")
RERANK_KEY = os.getenv("RERANK_KEY", "")
RERANK_VERIFY_TLS = os.getenv("RERANK_VERIFY_TLS", "true")
RERANK_TIMEOUT = float(os.getenv("RERANK_TIMEOUT", "60"))
RERANK_UPSTREAM_STYLE = os.getenv("RERANK_UPSTREAM_STYLE", "legacy_score")

# Backward-compatible names used by older tests/docs.
FABRIX_EMBEDDING_MODEL = EMBEDDING_MODEL
FABRIX_RERANK_MODEL = RERANK_MODEL

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
    tool_call_id: Optional[str] = None
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
    input: Optional[Union[str, List[str]]] = None
    sentences: Optional[List[str]] = None


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
def _env_bool(value: str, default: bool = True) -> bool:
    if value is None:
        return default
    return str(value).strip().lower() not in {"0", "false", "no", "off"}


def _chat_base_url() -> str:
    return FABRIX_CHAT_BASE_URL or FABRIX_BASE_URL


def _embedding_url() -> str:
    if EMBEDDING_URL:
        return EMBEDDING_URL
    if FABRIX_BASE_URL:
        return f"{FABRIX_BASE_URL.rstrip('/')}/v1/embeddings"
    return ""


def _rerank_url() -> str:
    if RERANK_URL:
        return RERANK_URL
    if FABRIX_BASE_URL:
        return f"{FABRIX_BASE_URL.rstrip('/')}/score"
    return ""


def _request_id(request: Request) -> str:
    return request.headers.get("X-Request-ID") or f"req-{uuid.uuid4()}"


def _client_cert(cert_path: str, key_path: str):
    return (cert_path, key_path) if cert_path and key_path else None


def _auth_headers(api_key: str) -> Dict[str, str]:
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    return headers


def _sanitize_detail(text: str, max_len: int = 1000) -> str:
    return text if len(text) <= max_len else f"{text[:max_len]}..."


def _post_json(
    *,
    api_name: str,
    url: str,
    headers: Dict[str, str],
    json_body: Dict[str, Any],
    timeout: float,
    verify_tls: bool,
    cert,
) -> Dict[str, Any]:
    import httpx

    try:
        with httpx.Client(timeout=timeout, verify=verify_tls, cert=cert) as http:
            resp = http.post(url, headers=headers, json=json_body)
            resp.raise_for_status()
            return resp.json()
    except httpx.TimeoutException as exc:
        raise HTTPException(status_code=504, detail=f"{api_name} upstream timed out: {exc}") from exc
    except httpx.HTTPStatusError as exc:
        body = _sanitize_detail(exc.response.text or "")
        raise HTTPException(
            status_code=502,
            detail=f"{api_name} upstream returned HTTP {exc.response.status_code}: {body}",
        ) from exc
    except httpx.RequestError as exc:
        raise HTTPException(status_code=503, detail=f"{api_name} upstream unavailable: {exc}") from exc
    except ValueError as exc:
        raise HTTPException(status_code=502, detail=f"{api_name} upstream returned non-JSON response") from exc


def _convert_to_langchain_message(msg: ChatMessage):
    role = msg.role.lower()
    content = msg.content or ""
    if role == "system":
        return SystemMessage(content=content)
    if role == "user":
        return HumanMessage(content=content)
    if role == "assistant":
        additional_kwargs: Dict[str, Any] = {}
        if msg.tool_calls:
            additional_kwargs["tool_calls"] = msg.tool_calls
        if msg.reasoning_content:
            additional_kwargs["reasoning_content"] = msg.reasoning_content
        if additional_kwargs:
            return AIMessage(content=content, additional_kwargs=additional_kwargs)
        return AIMessage(content=content)
    if role == "tool":
        if ToolMessage is None:
            raise ValueError("role='tool' is not supported by the installed langchain-core version")
        if not msg.tool_call_id:
            raise ValueError("role='tool' messages require tool_call_id")
        return ToolMessage(content=content, tool_call_id=msg.tool_call_id)
    raise ValueError(f"Unsupported chat message role: {msg.role}")


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
        fabrix_base_url=_chat_base_url(),
        temperature=request.temperature if request.temperature is not None else 0.7,
        max_tokens=request.max_tokens,
        streaming=streaming,
        model_kwargs=_chat_model_kwargs(request),
    )


def _require_fabrix_configured():
    if not (HAS_FABRIX and _chat_base_url() and FABRIX_API_KEY and FABRIX_CLIENT_KEY and FABRIX_MODEL):
        raise HTTPException(
            status_code=503,
            detail="Fabrix chat backend not available or not configured. Please set FABRIX_CHAT_BASE_URL or FABRIX_BASE_URL, FABRIX_API_KEY, FABRIX_CLIENT_KEY, FABRIX_MODEL and ensure langchain-fabrix is installed.",
        )


def _require_embedding_configured():
    if not _embedding_url():
        raise HTTPException(
            status_code=503,
            detail="Embedding upstream not configured. Please set EMBEDDING_URL.",
        )


def _require_rerank_configured():
    if not _rerank_url() and not HAS_FABRIX_RERANKER:
        raise HTTPException(
            status_code=503,
            detail="Rerank upstream not configured. Please set RERANK_URL.",
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
    candidates = getattr(result, "tool_calls", None)
    if isinstance(candidates, list) and candidates:
        return [_normalize_tool_call(call, idx) for idx, call in enumerate(candidates)]

    meta = getattr(result, "additional_kwargs", None)
    tool_calls = meta.get("tool_calls") if isinstance(meta, dict) else None
    if isinstance(tool_calls, list) and tool_calls:
        return [_normalize_tool_call(call, idx) for idx, call in enumerate(tool_calls)]

    response_meta = getattr(result, "response_metadata", None)
    tool_calls = response_meta.get("tool_calls") if isinstance(response_meta, dict) else None
    if isinstance(tool_calls, list) and tool_calls:
        return [_normalize_tool_call(call, idx) for idx, call in enumerate(tool_calls)]
    return None


def _normalize_tool_call(call: Any, index: int) -> Dict[str, Any]:
    if not isinstance(call, dict):
        call = {
            "id": getattr(call, "id", None),
            "name": getattr(call, "name", None),
            "args": getattr(call, "args", None),
            "function": getattr(call, "function", None),
            "type": getattr(call, "type", None),
        }

    call_id = call.get("id") or f"call_{index}"
    if isinstance(call.get("function"), dict):
        function = call["function"]
        name = function.get("name") or call.get("name") or "unknown_tool"
        arguments = function.get("arguments", "{}")
        if not isinstance(arguments, str):
            arguments = json.dumps(arguments, ensure_ascii=False)
        return {"id": call_id, "type": "function", "function": {"name": name, "arguments": arguments}}

    function_obj = call.get("function")
    if function_obj is not None:
        name = getattr(function_obj, "name", None) or call.get("name") or "unknown_tool"
        arguments = getattr(function_obj, "arguments", "{}")
    else:
        name = call.get("name") or "unknown_tool"
        arguments = call.get("args", call.get("arguments", "{}"))

    if not isinstance(arguments, str):
        arguments = json.dumps(arguments if arguments is not None else {}, ensure_ascii=False)
    return {"id": call_id, "type": "function", "function": {"name": name, "arguments": arguments}}


def _content_text(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    return json.dumps(content, ensure_ascii=False)


def _embedding_texts(request: EmbeddingRequest) -> List[str]:
    if request.input is not None and request.sentences is not None:
        raise HTTPException(status_code=400, detail="Use either 'input' or 'sentences', not both.")
    if request.sentences is not None:
        return list(request.sentences)
    if request.input is None:
        raise HTTPException(status_code=400, detail="Embedding request requires 'input' or 'sentences'.")
    return [request.input] if isinstance(request.input, str) else list(request.input)


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
async def chat_completions(request: ChatCompletionRequest, raw_request: Request, response: Response):
    request_id = _request_id(raw_request)
    response.headers["X-Request-ID"] = request_id
    _require_fabrix_configured()

    try:
        # Convert messages to LangChain format
        lc_messages = [_convert_to_langchain_message(m) for m in request.messages]
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    if request.stream:
        stream_response = _stream_chat_completions(request, lc_messages)
        stream_response.headers["X-Request-ID"] = request_id
        return stream_response

    try:
        chat = _build_chat_fabrix(request, streaming=False)
        # LangChain chat models have .invoke method returning AIMessage
        result = chat.invoke(lc_messages)
        content = _content_text(result.content if hasattr(result, "content") else str(result))
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
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=502,
            detail=f"Failed to communicate with Fabrix service: {str(e)}",
        ) from e


# ---------------------------------------------------------------------------
# Embeddings
# ---------------------------------------------------------------------------
def _fabrix_embed(texts: List[str], model: str) -> List[List[float]]:
    """Return one embedding vector per input text via Fabrix."""
    if HAS_FABRIX_EMBEDDINGS:
        embedder = FabrixEmbeddings(
            model_id=model,
            openapi_token=EMBEDDING_API_KEY,
            fabrix_client=FABRIX_CLIENT_KEY,
            fabrix_base_url=_embedding_url(),
        )
        return embedder.embed_documents(texts)

    data = _post_json(
        api_name="embedding",
        url=_embedding_url(),
        headers=_auth_headers(EMBEDDING_API_KEY),
        json_body={"model": model, "input": texts},
        timeout=EMBEDDING_TIMEOUT,
        verify_tls=_env_bool(EMBEDDING_VERIFY_TLS, True),
        cert=_client_cert(EMBEDDING_CERT, EMBEDDING_KEY),
    )
    return [item["embedding"] for item in data["data"]]


@app.post("/v1/embeddings")
async def embeddings(request: EmbeddingRequest, raw_request: Request, response: Response):
    response.headers["X-Request-ID"] = _request_id(raw_request)
    _require_embedding_configured()

    upstream_model = EMBEDDING_MODEL
    response_model = request.model or EMBEDDING_MODEL
    texts = _embedding_texts(request)

    try:
        vectors = _fabrix_embed(texts, upstream_model)
        data = [EmbeddingData(index=i, embedding=vec) for i, vec in enumerate(vectors)]
        prompt_tokens = sum(len(t.split()) for t in texts)
        return EmbeddingResponse(
            model=response_model,
            data=data,
            usage={"prompt_tokens": prompt_tokens, "total_tokens": prompt_tokens, "completion_tokens": 0},
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=502,
            detail=f"Failed to communicate with Fabrix service: {str(e)}",
        ) from e


# ---------------------------------------------------------------------------
# Reranking (kept at /score with text_1/text_2 to match the OpsG backend)
# ---------------------------------------------------------------------------
def _fabrix_rerank(query: str, documents: List[str], model: str) -> List[float]:
    """Return one relevance score per document via Fabrix."""
    if RERANK_UPSTREAM_STYLE == "legacy_score" and HAS_FABRIX_RERANKER:
        reranker = FabrixReranker(
            model_id=model,
            openapi_token=RERANK_API_KEY,
            fabrix_client=FABRIX_CLIENT_KEY,
            fabrix_base_url=_rerank_url(),
        )
        return reranker.score(query, documents)

    if RERANK_UPSTREAM_STYLE == "customer_v1_rerank":
        data = _post_json(
            api_name="rerank",
            url=_rerank_url(),
            headers=_auth_headers(RERANK_API_KEY),
            json_body={"model": model, "query": query, "documents": documents},
            timeout=RERANK_TIMEOUT,
            verify_tls=_env_bool(RERANK_VERIFY_TLS, True),
            cert=_client_cert(RERANK_CERT, RERANK_KEY),
        )
        return _scores_from_customer_rerank(data, len(documents))

    if RERANK_UPSTREAM_STYLE != "legacy_score":
        raise HTTPException(status_code=400, detail=f"Unsupported RERANK_UPSTREAM_STYLE: {RERANK_UPSTREAM_STYLE}")

    data = _post_json(
        api_name="rerank",
        url=_rerank_url(),
        headers=_auth_headers(RERANK_API_KEY),
        json_body={"model": model, "text_1": query, "text_2": documents},
        timeout=RERANK_TIMEOUT,
        verify_tls=_env_bool(RERANK_VERIFY_TLS, True),
        cert=_client_cert(RERANK_CERT, RERANK_KEY),
    )
    return [item["score"] for item in data["data"]]


def _scores_from_customer_rerank(data: Dict[str, Any], expected_count: int) -> List[float]:
    results = data.get("results")
    if not isinstance(results, list):
        raise HTTPException(status_code=502, detail="rerank upstream response missing results[]")

    scores: List[Optional[float]] = [None] * expected_count
    for item in results:
        if not isinstance(item, dict) or "index" not in item or "relevance_score" not in item:
            raise HTTPException(status_code=502, detail="rerank upstream result missing index or relevance_score")
        index = item["index"]
        if not isinstance(index, int) or index < 0 or index >= expected_count:
            raise HTTPException(status_code=502, detail=f"rerank upstream result index out of range: {index}")
        if scores[index] is not None:
            raise HTTPException(status_code=502, detail=f"rerank upstream returned duplicate index: {index}")
        scores[index] = float(item["relevance_score"])

    missing = [idx for idx, score in enumerate(scores) if score is None]
    if missing:
        raise HTTPException(status_code=502, detail=f"rerank upstream missing scores for indexes: {missing}")
    return [float(score) for score in scores]


@app.post("/score")
async def score(request: RerankRequest, raw_request: Request, response: Response):
    response.headers["X-Request-ID"] = _request_id(raw_request)
    _require_rerank_configured()

    upstream_model = RERANK_MODEL
    response_model = request.model or RERANK_MODEL
    try:
        scores = _fabrix_rerank(request.text_1, request.text_2, upstream_model)
        data = [RerankData(index=i, score=s) for i, s in enumerate(scores)]
        prompt_tokens = len(request.text_1.split()) + sum(len(t.split()) for t in request.text_2)
        return RerankResponse(
            model=response_model,
            data=data,
            usage={"prompt_tokens": prompt_tokens, "total_tokens": prompt_tokens, "completion_tokens": 0},
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=502,
            detail=f"Failed to communicate with Fabrix service: {str(e)}",
        ) from e


# Health check
@app.get("/health")
async def health():
    chat_missing = []
    if not HAS_FABRIX:
        chat_missing.append("langchain-fabrix")
    if not _chat_base_url():
        chat_missing.append("FABRIX_CHAT_BASE_URL or FABRIX_BASE_URL")
    for name, value in {
        "FABRIX_API_KEY": FABRIX_API_KEY,
        "FABRIX_CLIENT_KEY": FABRIX_CLIENT_KEY,
        "FABRIX_MODEL": FABRIX_MODEL,
    }.items():
        if not value:
            chat_missing.append(name)

    embedding_missing = [] if _embedding_url() else ["EMBEDDING_URL"]
    rerank_missing = [] if (_rerank_url() or HAS_FABRIX_RERANKER) else ["RERANK_URL"]

    return {
        "status": "ok",
        "fabrix_configured": not chat_missing,
        "components": {
            "chat": {"configured": not chat_missing, "missing": chat_missing},
            "embedding": {"configured": not embedding_missing, "missing": embedding_missing},
            "rerank": {"configured": not rerank_missing, "missing": rerank_missing},
        },
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
