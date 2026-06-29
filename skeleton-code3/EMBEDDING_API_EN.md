# BGE-m3 Series API Reference

## Embedding API

### Overview

| Item | Value |
|------|------|
| URL | `https://10.124.148.200:13001/v1/embeddings` |
| Method | POST |
| Content-Type | application/json |
| Authentication | mTLS (mutual TLS) |
| Model | bge-m3 |
| Vector Dimension | 1024 |

### Certificate Configuration

| Purpose | File Path |
|---------|-----------|
| Client Certificate | `certs/embedding_cert.pem` |
| Client Private Key | `certs/embedding_key.pem` |

> The project uses the above certificates in `utils.py:153-154`. Server-side certificate verification is disabled by default (`verify=False`).

### Request Format

**Required Fields**:
- `model`: must be `"bge-m3"`
- `input`: a string or an array of strings

**Single Text**:
```json
{
  "model": "bge-m3",
  "input": "Share me more details of Cisco bug CSCwp19413"
}
```

**Batch Texts**:
```json
{
  "model": "bge-m3",
  "input": [
    "Share me more details of Cisco bug CSCwp19413",
    "How to configure BGP on Cisco routers",
    "What is the cause of OSPF adjacency flapping"
  ]
}
```

### Response Format

```json
{
  "id": "embd-8ab044c3619b4239aa82eff5e011a632",
  "object": "list",
  "created": 1780619888,
  "model": "bge-m3",
  "data": [
    {
      "index": 0,
      "object": "embedding",
      "embedding": [0.0123, -0.0456, 0.0789, ...]
    }
  ],
  "usage": {
    "prompt_tokens": 40,
    "total_tokens": 40,
    "completion_tokens": 0,
    "prompt_tokens_details": null
  }
}
```

**Response Fields**:

| Field | Type | Description |
|-------|------|-------------|
| `id` | string | Unique request identifier, format `embd-{hash}` |
| `object` | string | Always `"list"` |
| `created` | int | Unix timestamp |
| `model` | string | Model name `bge-m3` |
| `data[index]` | int | Index corresponding to the input |
| `data[object]` | string | Always `"embedding"` |
| `data[embedding]` | float[] | 1024-dimensional floating-point vector |
| `usage[prompt_tokens]` | int | Number of input tokens |
| `usage[total_tokens]` | int | Total number of tokens |

### Error Response

```json
{
  "error": {
    "message": "...",
    "type": "Bad Request",
    "param": null,
    "code": 400
  }
}
```

### Usage in Project

| File | Line | Format | Purpose |
|------|------|--------|---------|
| `backend/db.py` | 185 | `{"model": "bge-m3", "input": [...]}` | Batch document insertion (2000 per batch) |
| `backend/db.py` | 267 | `{"model": "bge-m3", "input": [text]}` | Single query embedding (with caching) |

---

## Reranker API

### Overview

| Item | Value |
|------|------|
| URL | `https://10.124.148.200:8001/score` |
| Method | POST |
| Content-Type | application/json |
| Authentication | mTLS (mutual TLS) |
| Model | bge-reranker-v2-m3 |

### Certificate Configuration

| Purpose | File Path |
|---------|-----------|
| Client Certificate | `certs/reranking_cert.pem` |
| Client Private Key | `certs/reranking_key.pem` |

> The project uses the above certificates in `utils.py:156-157`. Server-side certificate verification is disabled by default (`verify=False`).

### Request Format

**Required Fields**:
- `model`: must be `"bge-reranker-v2-m3"`
- `text_1`: the query text (string)
- `text_2`: the documents to be ranked (array of strings)

```json
{
  "model": "bge-reranker-v2-m3",
  "text_1": "Share me more details of Cisco bug CSCwp19413",
  "text_2": [
    "Cisco bug CSCwp19413 causes OSPF adjacency flapping on Catalyst 9000 series switches.",
    "How to configure BGP peering on Cisco Nexus 9000 switches step by step.",
    "The bug CSCwp19413 was fixed in IOS XE 17.6.5 release."
  ]
}
```

### Response Format

```json
{
  "id": "score-470b0fa265fc4e8eb061c9f8bb495844",
  "object": "list",
  "created": 1780619089,
  "model": "bge-reranker-v2-m3",
  "data": [
    {
      "index": 0,
      "object": "score",
      "score": 0.787128210067749
    },
    {
      "index": 1,
      "object": "score",
      "score": 2.5758059564395808e-05
    },
    {
      "index": 2,
      "object": "score",
      "score": 0.13579940795898438
    }
  ],
  "usage": {
    "prompt_tokens": 107,
    "total_tokens": 107,
    "completion_tokens": 0,
    "prompt_tokens_details": null
  }
}
```

**Response Fields**:

| Field | Type | Description |
|-------|------|-------------|
| `id` | string | Unique request identifier, format `score-{hash}` |
| `object` | string | Always `"list"` |
| `created` | int | Unix timestamp |
| `model` | string | Model name `bge-reranker-v2-m3` |
| `data[index]` | int | Index corresponding to the document in `text_2` |
| `data[object]` | string | Always `"score"` |
| `data[score]` | float | Relevance score; higher means more relevant |
| `usage[prompt_tokens]` | int | Number of input tokens |
| `usage[total_tokens]` | int | Total number of tokens |

### Usage in Project

| File | Line | Purpose |
|------|------|---------|
| `backend/reranker.py` | 42 | Query re-ranking: scores the top_k_before_rerank=100 candidate documents returned by Milvus, and returns the top_k |

---
