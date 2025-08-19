import pytest
import torch

from response_formatter import format_ollama_response, format_openai_response
from schemas import EmbeddingResponse, OllamaEmbeddingResponse


@pytest.fixture
def mock_embeddings_tensor():
    """Fixture for a mock embeddings tensor."""
    return torch.tensor([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])


@pytest.fixture
def mock_single_embedding_tensor():
    """Fixture for a mock single embedding tensor."""
    return torch.tensor([[0.7, 0.8, 0.9]])


def test_format_openai_response_single_text(mock_single_embedding_tensor):
    """Test OpenAI response formatting with a single text input."""
    texts = ["test sentence"]
    model_name = "test-model"
    total_tokens = 5

    response = format_openai_response(mock_single_embedding_tensor, total_tokens, texts, model_name)

    assert isinstance(response, EmbeddingResponse)
    assert len(response.data) == 1
    assert response.data[0].embedding == mock_single_embedding_tensor[0].tolist()
    assert response.data[0].index == 0
    assert response.model == model_name
    assert response.object == "list"
    assert response.usage == {"prompt_tokens": total_tokens, "total_tokens": total_tokens}


def test_format_openai_response_multiple_texts(mock_embeddings_tensor):
    """Test OpenAI response formatting with multiple text inputs."""
    texts = ["sentence one", "sentence two"]
    model_name = "another-model"
    total_tokens = 10

    response = format_openai_response(mock_embeddings_tensor, total_tokens, texts, model_name)

    assert isinstance(response, EmbeddingResponse)
    assert len(response.data) == 2
    assert response.data[0].embedding == mock_embeddings_tensor[0].tolist()
    assert response.data[0].index == 0
    assert response.data[1].embedding == mock_embeddings_tensor[1].tolist()
    assert response.data[1].index == 1
    assert response.model == model_name
    assert response.object == "list"
    assert response.usage == {"prompt_tokens": total_tokens, "total_tokens": total_tokens}


def test_format_openai_response_empty_input():
    """Test OpenAI response formatting with empty input."""
    texts = []
    model_name = "empty-model"
    total_tokens = 0
    empty_tensor = torch.empty(0, 0)

    response = format_openai_response(empty_tensor, total_tokens, texts, model_name)

    assert isinstance(response, EmbeddingResponse)
    assert len(response.data) == 0
    assert response.model == model_name
    assert response.object == "list"
    assert response.usage == {"prompt_tokens": 0, "total_tokens": 0}


def test_format_ollama_response_single_embedding(mock_single_embedding_tensor):
    """Test Ollama response formatting with a single embedding."""
    total_tokens = 5

    response = format_ollama_response(mock_single_embedding_tensor, total_tokens)

    assert isinstance(response, OllamaEmbeddingResponse)
    assert response.embeddings == mock_single_embedding_tensor.tolist()
    assert response.usage == {"promptTokens": total_tokens, "totalTokens": total_tokens}


def test_format_ollama_response_multiple_embeddings(mock_embeddings_tensor):
    """Test Ollama response formatting with multiple embeddings."""
    total_tokens = 10

    response = format_ollama_response(mock_embeddings_tensor, total_tokens)

    assert isinstance(response, OllamaEmbeddingResponse)
    assert response.embeddings == mock_embeddings_tensor.tolist()
    assert response.usage == {"promptTokens": total_tokens, "totalTokens": total_tokens}


def test_format_ollama_response_empty_input():
    """Test Ollama response formatting with empty input."""
    total_tokens = 0
    empty_tensor = torch.empty(0, 0)

    response = format_ollama_response(empty_tensor, total_tokens)

    assert isinstance(response, OllamaEmbeddingResponse)
    assert response.embeddings == []
    assert response.usage == {"promptTokens": 0, "totalTokens": 0}
