# RAG Router Implementation

A Syft-based RAG (Retrieval-Augmented Generation) router implementation that provides a unified interface for LLM interactions, document retrieval, and embedding management.

## Features

- **Unified LLM Interface**: Support for both chat and completion endpoints
- **Document Retrieval**: Integrated RAG capabilities with vector search
- **Embedding Management**: Automatic document embedding and indexing
- **Event-Driven Architecture**: Built on SyftEvents for scalable processing
- **Rate Limiting**: Configurable rate limits for API endpoints
- **Extensible Design**: Easy to add new LLM providers and retrieval strategies

## Project Structure

```
.
├── pyproject.toml      # Project dependencies and configuration
├── run.sh             # Service startup script
├── router.py          # RAG router implementation
├── server.py          # Server implementation
└── chat_test.py       # Example chat implementation
```

## Prerequisites

- Python 3.12 or higher
- uv (Python package manager)
- Access to a Syft network

## Installation

1. Install uv (if not already installed):
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

2. Clone the repository and install dependencies:
```bash
git clone <repository-url>
cd rag-router-demo
uv pip install -e .
```

## Configuration

The application can be configured through the `pyproject.toml` file:

```toml
[tool.rag-app]
# Rate limiting settings
enable_rate_limiting = true
requests_per_minute = 1
requests_per_hour = 10
requests_per_day = 1000

# Embedding settings
embedder_endpoint = ""
indexer_endpoint = ""

# Retrieval settings
retriever_endpoint = ""
```

## Running the Service

Start the service using the provided script:

```bash
./run.sh
```

The script will:
1. Create a Python virtual environment
2. Install all dependencies
3. Start the server with the specified project name

## API Endpoints

### Document Retrieval
- `POST /retrieve`: Retrieve relevant documents
  ```json
  {
    "query": "string",
    "options": {
      "top_k": 5,
      "score_threshold": 0.7
    }
  }
  ```

### Health Check
- `GET /ping`: Check service health

## Document Embedding

The service automatically watches for new documents in the `{datasite}/embeddings` directory. When new JSON files are added, they are automatically:
1. Chunked into appropriate sizes
2. Embedded using the configured embedder
3. Indexed in the vector database

## Development

### Setting Up Development Environment

1. Create and activate a virtual environment:
```bash
uv venv -p 3.12 .venv
source .venv/bin/activate
```

2. Install development dependencies:
```bash
uv pip install -e ".[dev]"
```

### Adding New LLM Providers

To add a new LLM provider:
1. Create a new class that inherits from `BaseLLMRouter`
2. Implement the required methods:
   - `generate_completion`
   - `generate_chat`
   - `retrieve_documents`
   - `embed_documents`
3. Update the `load_router()` function in `server.py`
