import json
import os
import unittest
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


BASE_URL = os.getenv("FABRIX_ADAPTER_BASE_URL", "http://127.0.0.1:8000")


def _healthcheck() -> None:
    try:
        request = Request(f"{BASE_URL}/health", method="GET")
        with urlopen(request, timeout=3.0) as response:
            status_code = response.status
            body = response.read().decode("utf-8")
    except (HTTPError, URLError) as exc:
        raise unittest.SkipTest(f"Fabrix adapter is not reachable at {BASE_URL}: {exc}") from exc

    if status_code != 200:
        raise unittest.SkipTest(
            f"Fabrix adapter at {BASE_URL} is not ready for integration tests: "
            f"status={status_code}, body={body}"
        )


def _post_json(path: str, payload: dict, timeout: float = 60.0) -> tuple[int, dict, str]:
    data = json.dumps(payload).encode("utf-8")
    request = Request(
        f"{BASE_URL}{path}",
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urlopen(request, timeout=timeout) as response:
        body = response.read().decode("utf-8")
        return response.status, json.loads(body), body


def test_chat_completions_against_running_gateway():
    _healthcheck()

    payload = {
        "model": os.getenv("FABRIX_MODEL", "test"),
        "messages": [{"role": "user", "content": "Hello"}],
        "temperature": 0.0,
        "max_tokens": 32,
    }

    status_code, data, body = _post_json("/v1/chat/completions", payload)

    assert status_code == 200, body
    assert data["object"] == "chat.completion"
    assert data["model"] == payload["model"]
    assert len(data["choices"]) == 1
    assert data["choices"][0]["message"]["role"] == "assistant"
    assert "content" in data["choices"][0]["message"]


def test_embeddings_against_running_gateway():
    _healthcheck()

    status_code, data, body = _post_json("/v1/embeddings", {"input": ["first", "second"]})

    assert status_code == 200, body
    assert data["object"] == "list"
    assert len(data["data"]) == 2
    assert data["data"][0]["object"] == "embedding"
    assert isinstance(data["data"][0]["embedding"], list)


def test_rerank_against_running_gateway():
    _healthcheck()

    status_code, data, body = _post_json(
        "/score",
        {
            "text_1": "the query",
            "text_2": ["doc a", "doc b"],
        },
    )

    assert status_code == 200, body
    assert data["object"] == "list"
    assert len(data["data"]) == 2
    assert data["data"][0]["object"] == "score"
    assert isinstance(data["data"][0]["score"], float)