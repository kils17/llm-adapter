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
    assert "Fabrix backend not available or not configured" in response.json()["detail"]
