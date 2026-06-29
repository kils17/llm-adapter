import os
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient

# Set environment variables for Fabrix configuration before importing the app
os.environ["FABRIX_BASE_URL"] = "http://fake.example.com"
os.environ["FABRIX_API_KEY"] = "fake-api-key"
os.environ["FABRIX_CLIENT_KEY"] = "fake-client-key"
os.environ["FABRIX_MODEL"] = "fake-model"

# Now import the app after env vars are set
from fabrix_adapter import app

client = TestClient(app)

def test_chat_completions(monkeypatch):
    # Ensure HAS_FABRIX is True and ChatFabrix is available
    with patch('fabrix_adapter.HAS_FABRIX', True), \
         patch('fabrix_adapter.ChatFabrix') as MockChatFabrix:
        mock_chat_instance = MagicMock()
        mock_result = MagicMock()
        mock_result.content = "Mocked response from Fabrix"
        mock_chat_instance.invoke.return_value = mock_result
        MockChatFabrix.return_value = mock_chat_instance

        response = client.post("/v1/chat/completions", json={
            "model": "test",
            "messages": [{"role": "user", "content": "Hello"}],
            "temperature": 0.7,
            "max_tokens": 100
        })
    assert response.status_code == 200
    data = response.json()
    assert data["object"] == "chat.completion"
    assert len(data["choices"]) == 1
    assert data["choices"][0]["message"]["role"] == "assistant"
    assert data["choices"][0]["message"]["content"] == "Mocked response from Fabrix"
    MockChatFabrix.assert_called_once()
    mock_chat_instance.invoke.assert_called_once()
    args, _ = mock_chat_instance.invoke.call_args
    from langchain_core.messages import HumanMessage
    assert len(args[0]) == 1
    assert isinstance(args[0][0], HumanMessage)
    assert args[0][0].content == "Hello"

def test_chat_completions_with_system(monkeypatch):
    with patch('fabrix_adapter.HAS_FABRIX', True), \
         patch('fabrix_adapter.ChatFabrix') as MockChatFabrix:
        mock_chat_instance = MagicMock()
        mock_result = MagicMock()
        mock_result.content = "Mocked response with system"
        mock_chat_instance.invoke.return_value = mock_result
        MockChatFabrix.return_value = mock_chat_instance

        response = client.post("/v1/chat/completions", json={
            "model": "test",
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Hello"}
            ],
            "temperature": 0.0,
            "max_tokens": 50
        })
    assert response.status_code == 200
    data = response.json()
    assert data["object"] == "chat.completion"
    assert len(data["choices"]) == 1
    assert data["choices"][0]["message"]["role"] == "assistant"
    assert data["choices"][0]["message"]["content"] == "Mocked response with system"
    MockChatFabrix.assert_called_once()
    mock_chat_instance.invoke.assert_called_once()
    args, _ = mock_chat_instance.invoke.call_args
    from langchain_core.messages import SystemMessage, HumanMessage
    assert len(args[0]) == 2
    assert isinstance(args[0][0], SystemMessage)
    assert args[0][0].content == "You are a helpful assistant."
    assert isinstance(args[0][1], HumanMessage)
    assert args[0][1].content == "Hello"


def test_chat_completions_extracts_tool_calls_from_result_attribute(monkeypatch):
    with patch('fabrix_adapter.HAS_FABRIX', True), \
         patch('fabrix_adapter.ChatFabrix') as MockChatFabrix:
        mock_chat_instance = MagicMock()
        mock_result = MagicMock()
        mock_result.content = ""
        mock_result.tool_calls = [{"id": "call_1", "name": "demo_tool", "args": {"value": 7}}]
        mock_chat_instance.invoke.return_value = mock_result
        MockChatFabrix.return_value = mock_chat_instance

        response = client.post("/v1/chat/completions", json={
            "model": "test",
            "messages": [{"role": "user", "content": "call a tool"}],
            "tools": [{
                "type": "function",
                "function": {
                    "name": "demo_tool",
                    "description": "demo",
                    "parameters": {"type": "object", "properties": {}, "additionalProperties": False},
                    "strict": True,
                },
            }],
            "tool_choice": "auto",
        })
    assert response.status_code == 200
    data = response.json()
    assert data["choices"][0]["finish_reason"] == "tool_calls"
    assert data["choices"][0]["message"]["tool_calls"] == [{
        "id": "call_1",
        "type": "function",
        "function": {"name": "demo_tool", "arguments": '{"value": 7}'},
    }]


def test_chat_completions_converts_assistant_tool_history(monkeypatch):
    with patch('fabrix_adapter.HAS_FABRIX', True), \
         patch('fabrix_adapter.ChatFabrix') as MockChatFabrix:
        mock_chat_instance = MagicMock()
        mock_result = MagicMock()
        mock_result.content = "Tool result summarized"
        mock_result.tool_calls = []
        mock_chat_instance.invoke.return_value = mock_result
        MockChatFabrix.return_value = mock_chat_instance

        response = client.post("/v1/chat/completions", json={
            "model": "test",
            "messages": [
                {"role": "user", "content": "List devices"},
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [{
                        "id": "call_abc",
                        "type": "function",
                        "function": {"name": "topo_get_group_members", "arguments": '{"groups":["X"]}'},
                    }],
                },
                {"role": "tool", "tool_call_id": "call_abc", "content": '["R1", "R2"]'},
                {"role": "user", "content": "Summarize"},
            ],
        })
    assert response.status_code == 200
    args, _ = mock_chat_instance.invoke.call_args
    from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
    assert isinstance(args[0][0], HumanMessage)
    assert isinstance(args[0][1], AIMessage)
    assert args[0][1].additional_kwargs["tool_calls"][0]["id"] == "call_abc"
    assert isinstance(args[0][2], ToolMessage)
    assert args[0][2].tool_call_id == "call_abc"

def test_chat_completions_streaming(monkeypatch):
    from langchain_core.messages import AIMessageChunk

    with patch('fabrix_adapter.HAS_FABRIX', True), \
         patch('fabrix_adapter.ChatFabrix') as MockChatFabrix:
        mock_chat_instance = MagicMock()
        mock_chat_instance.stream.return_value = iter([
            AIMessageChunk(content="Hello"),
            AIMessageChunk(content=" world"),
        ])
        MockChatFabrix.return_value = mock_chat_instance

        response = client.post("/v1/chat/completions", json={
            "model": "test",
            "messages": [{"role": "user", "content": "Hi"}],
            "stream": True,
        })
        assert response.status_code == 200
        body = response.text

    assert "chat.completion.chunk" in body
    assert "Hello" in body
    assert "world" in body
    assert "data: [DONE]" in body
    mock_chat_instance.stream.assert_called_once()


def test_embeddings(monkeypatch):
    with patch('fabrix_adapter.HAS_FABRIX_EMBEDDINGS', True), \
         patch('fabrix_adapter.FabrixEmbeddings') as MockEmbeddings:
        mock_embedder = MagicMock()
        mock_embedder.embed_documents.return_value = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
        MockEmbeddings.return_value = mock_embedder

        response = client.post("/v1/embeddings", json={
            "model": "bge-m3",
            "input": ["first", "second"],
        })
    assert response.status_code == 200
    data = response.json()
    assert data["object"] == "list"
    assert data["model"] == "bge-m3"
    assert len(data["data"]) == 2
    assert data["data"][0]["object"] == "embedding"
    assert data["data"][0]["embedding"] == [0.1, 0.2, 0.3]
    assert data["data"][1]["index"] == 1


def test_embeddings_single_string(monkeypatch):
    with patch('fabrix_adapter.HAS_FABRIX_EMBEDDINGS', True), \
         patch('fabrix_adapter.FabrixEmbeddings') as MockEmbeddings:
        mock_embedder = MagicMock()
        mock_embedder.embed_documents.return_value = [[0.9, 0.8]]
        MockEmbeddings.return_value = mock_embedder

        response = client.post("/v1/embeddings", json={"input": "only one"})
    assert response.status_code == 200
    data = response.json()
    # Falls back to FABRIX_EMBEDDING_MODEL default when model is omitted.
    assert data["model"] == "bge-m3"
    assert len(data["data"]) == 1
    args, _ = mock_embedder.embed_documents.call_args
    assert args[0] == ["only one"]


def test_embeddings_sentences_format(monkeypatch):
    with patch('fabrix_adapter.HAS_FABRIX_EMBEDDINGS', True), \
         patch('fabrix_adapter.FabrixEmbeddings') as MockEmbeddings:
        mock_embedder = MagicMock()
        mock_embedder.embed_documents.return_value = [[0.1], [0.2]]
        MockEmbeddings.return_value = mock_embedder

        response = client.post("/v1/embeddings", json={"sentences": ["first", "second"]})
    assert response.status_code == 200
    data = response.json()
    assert len(data["data"]) == 2
    args, _ = mock_embedder.embed_documents.call_args
    assert args[0] == ["first", "second"]


def test_score_rerank(monkeypatch):
    with patch('fabrix_adapter.HAS_FABRIX_RERANKER', True), \
         patch('fabrix_adapter.FabrixReranker') as MockReranker:
        mock_reranker = MagicMock()
        mock_reranker.score.return_value = [0.78, 0.0002, 0.13]
        MockReranker.return_value = mock_reranker

        response = client.post("/score", json={
            "model": "bge-reranker-v2-m3",
            "text_1": "the query",
            "text_2": ["doc a", "doc b", "doc c"],
        })
    assert response.status_code == 200
    data = response.json()
    assert data["object"] == "list"
    assert data["model"] == "bge-reranker-v2-m3"
    assert len(data["data"]) == 3
    assert data["data"][0]["object"] == "score"
    assert data["data"][0]["score"] == 0.78
    assert data["data"][2]["index"] == 2
    args, _ = mock_reranker.score.call_args
    assert args[0] == "the query"
    assert args[1] == ["doc a", "doc b", "doc c"]


def test_score_customer_v1_rerank_restores_original_order(monkeypatch):
    upstream_response = {
        "results": [
            {"index": 1, "document": {"text": "Mouse"}, "relevance_score": 0.9},
            {"index": 2, "document": {"text": "Cheese"}, "relevance_score": 0.2},
            {"index": 0, "document": {"text": "Cat"}, "relevance_score": 0.1},
        ]
    }
    with patch('fabrix_adapter.RERANK_UPSTREAM_STYLE', 'customer_v1_rerank'), \
         patch('fabrix_adapter.RERANK_URL', 'http://rerank.example/v1/rerank'), \
         patch('fabrix_adapter.RERANK_MODEL', 'bge-reranker'), \
         patch('fabrix_adapter._post_json', return_value=upstream_response) as mock_post:
        response = client.post("/score", json={
            "model": "bge-reranker-v2-m3",
            "text_1": "what is stronger",
            "text_2": ["Cat", "Mouse", "Cheese"],
        })
    assert response.status_code == 200
    data = response.json()
    assert data["model"] == "bge-reranker-v2-m3"
    assert [item["score"] for item in data["data"]] == [0.1, 0.9, 0.2]
    _, kwargs = mock_post.call_args
    assert kwargs["url"] == "http://rerank.example/v1/rerank"
    assert kwargs["json_body"] == {
        "model": "bge-reranker",
        "query": "what is stronger",
        "documents": ["Cat", "Mouse", "Cheese"],
    }


def test_score_customer_v1_rerank_rejects_missing_index(monkeypatch):
    upstream_response = {"results": [{"index": 0, "relevance_score": 0.1}]}
    with patch('fabrix_adapter.RERANK_UPSTREAM_STYLE', 'customer_v1_rerank'), \
         patch('fabrix_adapter.RERANK_URL', 'http://rerank.example/v1/rerank'), \
         patch('fabrix_adapter._post_json', return_value=upstream_response):
        response = client.post("/score", json={
            "model": "bge-reranker-v2-m3",
            "text_1": "query",
            "text_2": ["a", "b"],
        })
    assert response.status_code == 502
    assert "missing scores" in response.json()["detail"]


def test_missing_config(monkeypatch):
    # Patch config values to empty strings to simulate missing config
    with patch('fabrix_adapter.FABRIX_BASE_URL', ''), \
         patch('fabrix_adapter.FABRIX_API_KEY', ''), \
         patch('fabrix_adapter.FABRIX_CLIENT_KEY', ''), \
         patch('fabrix_adapter.FABRIX_MODEL', ''), \
         patch('fabrix_adapter.HAS_FABRIX', True):  # still have library but missing config
        response = client.post("/v1/chat/completions", json={
            "model": "test",
            "messages": [{"role": "user", "content": "Hello"}],
        })
    assert response.status_code == 503
    assert "Fabrix chat backend not available or not configured" in response.json()["detail"]
