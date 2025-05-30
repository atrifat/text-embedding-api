# Text Embedding API Server

A Text Embedding API server built with FastAPI and Hugging Face Transformers, designed to be compatible with OpenAI's Embeddings API format.

## Features

- **FastAPI Backend**: Built with FastAPI, providing an asynchronous API.
- **Hugging Face Transformers Integration**: Supports various pre-trained transformer models for generating embeddings.
- **Model and Embeddings Caching**: Includes in-memory caches for loaded models, tokenizers, and generated embeddings to help improve response times.
- **Batch Processing**: Processes multiple text inputs in configurable batches.
- **OpenAI API Compatibility**: The `/v1/embeddings` endpoint is designed to be compatible with the OpenAI Embeddings API.

## Architecture Overview

The application is structured around a FastAPI server (`app.py`) that provides endpoints for generating text embeddings and listing available models. Model configurations and aliases are managed in `models_config.py`.

### Key Components:

- **`app.py`**:
  - Initializes the FastAPI application, CORS middleware, and static file serving.
  - Manages application settings via `AppSettings` (Pydantic BaseSettings).
  - Handles application lifecycle events (startup/shutdown) for cache initialization and model warmup.
  - Includes `load_model` for asynchronous model and tokenizer loading with caching.
  - Contains the core `get_embeddings_batch` function for generating embeddings, incorporating caching, batching, and error handling.
  - Defines Pydantic models for API request and response validation.
  - Exposes API endpoints: `/`, `/v1/models`, `/v1/models/{model_id}`, `/api/embed`, and `/v1/embeddings`.
- **`models_config.py`**:
  - Defines `CANONICAL_MODELS` with detailed configurations (dimension, max tokens, instruction prefixes) for each supported Hugging Face model.
  - Defines `MODEL_ALIASES` to map common names to canonical model names.
  - Provides `get_model_config` to resolve model names to their full configurations.

## Setup and Installation

### Prerequisites

- Python 3.10+
- `pip`

### Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/your-repo/text-embedding-api.git
    cd text-embedding-api
    ```
2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate # On Windows, use `venv\Scripts\activate`
    ```
3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
4.  **Download necessary models (optional, models are downloaded on first use):**
    The application will automatically download models from Hugging Face Hub on demand. Ensure you have an active internet connection for the first run of each model.

### Configuration

You can configure the application using environment variables or by creating a `.env` file in the project root.

First, copy the example environment file:

```bash
cp .env.example .env
```

Then, edit the `.env` file to set your desired configuration.

Example `.env` file:

```
DEFAULT_MODEL=text-embedding-3-large
EMBEDDING_BATCH_SIZE=8
EMBEDDINGS_CACHE_ENABLED=True
EMBEDDINGS_CACHE_MAXSIZE=2048
CUDA_CACHE_CLEAR_ENABLED=True
APP_PORT=7860
APP_HOST=0.0.0.0
ENVIRONMENT=development
REPORT_CACHED_TOKENS=False
ALLOWED_ORIGINS=["*"] # Comma-separated list of allowed origins for CORS (e.g., "http://localhost:3000,https://example.com"). Use ["*"] to allow all origins.
```

### Running the Server

```bash
uvicorn app:app --host 0.0.0.0 --port 7860 --reload
```

Or, using the `if __name__ == "__main__":` block:

```bash
python app.py
```

## API Endpoints

### 1. List Available Models

`GET /v1/models`

Returns a list of all embedding models available on the server.

**Example Response:**

```json
{
  "data": [
    {
      "id": "all-MiniLM-L6-v2",
      "object": "model",
      "created": 1678886400,
      "owned_by": "local"
    },
    {
      "id": "text-embedding-3-large",
      "object": "model",
      "created": 1678886400,
      "owned_by": "local"
    }
  ],
  "object": "list"
}
```

### 2. Get Model Information

`GET /v1/models/{model_id}`

Returns information about a specific embedding model.

**Example Response (for `model_id=text-embedding-3-large`):**

```json
{
  "id": "text-embedding-3-large",
  "object": "model",
  "created": 1678886400,
  "owned_by": "local"
}
```

### 3. Create Embeddings

`POST /v1/embeddings` or `POST /api/embed`

Generates embeddings for the given input text(s). Compatible with OpenAI's Embeddings API.

**Request Body:**

```json
{
  "input": "This is an example sentence.",
  "model": "text-embedding-3-large",
  "encoding_format": "float"
}
```

Or for multiple inputs:

```json
{
  "input": ["This is the first sentence.", "This is the second sentence."],
  "model": "text-embedding-3-large",
  "encoding_format": "float"
}
```

**Example Response:**

```json
{
  "data": [
    {
      "object": "embedding",
      "embedding": [0.1, 0.2, ..., 0.9],
      "index": 0
    }
  ],
  "model": "text-embedding-3-large",
  "object": "list",
  "usage": {
    "prompt_tokens": 10,
    "total_tokens": 10
  }
}
```

## Available Models

The following models are configured in `models_config.py`. Please note that while aliases like `text-embedding-3-small` and `text-embedding-3-large` are provided for OpenAI API compatibility, their actual embedding dimensions are determined by the underlying Hugging Face models used in this server, and may differ from OpenAI's native models.

| Canonical Name          | Alias (OpenAI-compatible)              | Dimension | Max Tokens | Instruction Prefix Required | Default Prefix     |
| :---------------------- | :------------------------------------- | :-------- | :--------- | :-------------------------- | :----------------- |
| `all-MiniLM-L6-v2`      | `all-minilm`, `text-embedding-3-small` | 384       | 512        | No                          | -                  |
| `gte-multilingual-base` | `text-embedding-3-large`               | 768       | 8192       | No                          | -                  |
| `nomic-embed-text-v1.5` | `nomic-embed-text`                     | 768       | 8192       | Yes                         | `search_document:` |
| `all-mpnet-base-v2`     | -                                      | 768       | 384        | No                          | -                  |

## Development

### Running Tests

(Assuming `pytest` is installed via `requirements.txt`)

```bash
pytest
```

### Local Development Checks (Pre-commit Hooks)

To ensure code quality, formatting, and test integrity before committing, this project uses `pre-commit` hooks. These hooks run automatically on your staged files before each commit.

1.  **Install `pre-commit` and project dependencies**:
    Ensure you have a virtual environment activated, then install the development dependencies:
    ```bash
    pip install -r requirements.txt
    ```

2.  **Install the Git hooks**:
    Navigate to the project's root directory and run:
    ```bash
    pre-commit install
    ```
    This command sets up the Git hooks based on the configuration in [`pre-commit-config.yaml`](.pre-commit-config.yaml).

Once installed, `pre-commit` will automatically run checks (linting, formatting, testing) on your staged changes before a commit is finalized. If any checks fail, the commit will be aborted, allowing you to fix issues immediately.

To manually run all pre-commit checks on your entire codebase at any time, use:
```bash
pre-commit run --all-files
```

## Continuous Integration

This project uses GitHub Actions for continuous integration. The CI pipeline automatically runs checks on every push and pull request to the `main` branch to ensure code quality, functionality, and security.

The CI workflow is defined in [`python-ci.yml`](.github/workflows/python-ci.yml) and includes the following steps:

*   **Environment Setup**: Configures multiple Python versions (3.8 to 3.12).
*   **Dependency Installation**: Installs project dependencies, linters, and testing tools.
*   **Linting**: Runs `flake8` to enforce code style and identify errors.
*   **Testing**: Executes unit and integration tests using `pytest` and generates a code coverage report.
*   **Security Scan**: Performs a security audit of project dependencies using `pip-audit`.

## License

This project is licensed under the MIT License. See the [`LICENSE`](LICENSE) file for details.
