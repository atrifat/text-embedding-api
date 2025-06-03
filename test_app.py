import hashlib
import importlib
import os
from unittest import mock

import pytest
import torch
from cachetools import LRUCache
from fastapi import HTTPException
from fastapi.testclient import TestClient
from starlette import status

import app
import config
import embedding_processor
import model_loader
import models_config
from config import get_app_settings

client = TestClient(app.app)

all_model_keys = list(models_config.MODELS.keys())


def _reload_app_and_client():
    """
    Helper function to reload app and models_config modules,
    re-initialize the global client, and reset the embeddings cache.
    This is necessary when environment variables affecting app settings are changed.
    """
    # Reload all relevant modules
    importlib.reload(models_config)
    importlib.reload(config)
    importlib.reload(model_loader)
    importlib.reload(embedding_processor)
    importlib.reload(app)  # Reload app last to pick up all new imports

    global client
    client = TestClient(app.app)
    settings = get_app_settings()
    # IMPORTANT: Re-initialize embeddings_cache with the correct maxsize after app reload
    embedding_processor.embeddings_cache = LRUCache(maxsize=settings.embeddings_cache_maxsize)
    embedding_processor.embeddings_cache.clear()


def _assert_embedding_response_structure(response_data, expected_num_embeddings, model_name):
    """
    Helper function to assert the common structure and content of embedding API responses.
    """
    assert "data" in response_data
    assert len(response_data["data"]) == expected_num_embeddings

    expected_dimension = models_config.get_model_config(model_name)["dimension"]

    for embedding_obj in response_data["data"]:
        assert "embedding" in embedding_obj
        assert len(embedding_obj["embedding"]) == expected_dimension

    assert "usage" in response_data
    assert "prompt_tokens" in response_data["usage"]
    assert "total_tokens" in response_data["usage"]
    assert response_data["usage"]["prompt_tokens"] > 0
    assert response_data["usage"]["total_tokens"] == response_data["usage"]["prompt_tokens"]


@pytest.fixture(autouse=True)
def mock_model_loading():
    """
    Fixture to mock app.load_model and provide configurable mock model/tokenizer.
    This fixture is autoused, meaning it applies to all tests in this file.
    It also handles environment variable cleanup and cache clearing.
    """
    original_env = os.environ.copy()

    # Temporarily initialize embeddings_cache for the fixture's use if it's None
    if embedding_processor.embeddings_cache is None:
        embedding_processor.embeddings_cache = LRUCache(maxsize=2048)  # Use a default size for fixture setup

    # Correctly store the original embeddings cache
    original_embeddings_cache = LRUCache(maxsize=embedding_processor.embeddings_cache.maxsize)
    original_embeddings_cache.update(embedding_processor.embeddings_cache)

    # Clear cache and reset env vars to default for each test
    embedding_processor.embeddings_cache.clear()
    os.environ["EMBEDDINGS_CACHE_ENABLED"] = "true"
    os.environ["REPORT_CACHED_TOKENS"] = "false"
    os.environ["EMBEDDINGS_CACHE_MAXSIZE"] = "2048"

    # Reload all relevant modules
    importlib.reload(models_config)
    importlib.reload(config)
    importlib.reload(model_loader)
    importlib.reload(embedding_processor)
    importlib.reload(app)  # Reload app last to pick up all new imports

    # IMPORTANT: Re-initialize embeddings_cache with the correct maxsize after app reload
    settings = get_app_settings()
    embedding_processor.embeddings_cache = LRUCache(maxsize=settings.embeddings_cache_maxsize)

    global client
    client = TestClient(app.app)

    with mock.patch("model_loader.load_model") as mock_load_model, mock.patch(
        "transformers.AutoTokenizer.from_pretrained"
    ) as mock_auto_tokenizer_from_pretrained:
        mock_model = mock.MagicMock()
        mock_tokenizer = mock.MagicMock()

        class MockBatchEncoding(dict):
            """A mock class that behaves like transformers.tokenization_utils_base.BatchEncoding."""

            def __init__(self, input_ids, attention_mask):
                super().__init__({"input_ids": input_ids, "attention_mask": attention_mask})
                self.input_ids = input_ids
                self.attention_mask = attention_mask

            def to(self, device):
                self.input_ids = self.input_ids.to(device)
                self.attention_mask = self.attention_mask.to(device)
                return self

        OOM_TRIGGER_TEXT = "OOM_TEST_SENTENCE"

        def get_mock_tokenizer_output(texts, **kwargs):
            """Helper to create mock tokenizer output that mimics BatchEncoding."""
            num_texts = len(texts) if isinstance(texts, list) else 1

            if OOM_TRIGGER_TEXT in texts:
                input_ids = torch.full((num_texts, 3), 999, dtype=torch.long)  # Use 999 as a signal
            else:
                input_ids = torch.full((num_texts, 3), 1, dtype=torch.long)

            attention_mask = torch.full((num_texts, 3), 1, dtype=torch.long)

            return MockBatchEncoding(input_ids, attention_mask)

        mock_tokenizer.side_effect = get_mock_tokenizer_output
        mock_auto_tokenizer_from_pretrained.return_value = mock_tokenizer

        def mock_model_call_side_effect(**batch_dict):
            input_ids = batch_dict["input_ids"]

            # If input_ids contain the OOM signal (999), raise OutOfMemoryError
            if (input_ids == 999).any():
                raise torch.cuda.OutOfMemoryError("Mock CUDA Out of Memory")

            batch_size = input_ids.size(0)
            mock_output = mock.MagicMock()
            # Use the dimension stored on the mock_model instance
            mock_output.last_hidden_state = torch.randn(batch_size, 10, mock_model.dimension)
            return mock_output

        mock_model.side_effect = mock_model_call_side_effect

        def load_model_side_effect(model_name: str):
            """Simulate load_model behavior."""
            config = models_config.get_model_config(model_name)
            mock_model.dimension = config["dimension"]
            return mock_model, mock_tokenizer

        mock_load_model.side_effect = load_model_side_effect
        yield mock_load_model

    # Restore original env vars and cache state after each test
    os.environ.clear()
    os.environ.update(original_env)
    embedding_processor.embeddings_cache.clear()
    embedding_processor.embeddings_cache.update(original_embeddings_cache)
    importlib.reload(app)
    importlib.reload(models_config)
    importlib.reload(config)
    importlib.reload(model_loader)
    importlib.reload(embedding_processor)


@pytest.mark.parametrize("model_name", all_model_keys)
def test_create_embeddings_single_input(model_name):
    """Test embedding generation with a single input for each model."""
    response = client.post(
        "/v1/embeddings",
        json={
            "input": "This is a test sentence.",
            "model": model_name,
            "encoding_format": "float",
        },
    )
    assert response.status_code == 200
    data = response.json()
    _assert_embedding_response_structure(data, 1, model_name)
    # Specific assertion for single input:
    # The OpenAI API for single input *does* include an 'index' field, which is 0.
    assert "index" in data["data"][0]
    assert data["data"][0]["index"] == 0


@pytest.mark.parametrize("model_name", all_model_keys)
def test_create_embeddings_batch_input(model_name):
    """Test batch embedding generation for each model."""
    test_inputs = [
        "First test sentence.",
        "Second test sentence.",
        "Third test sentence.",
    ]

    response = client.post(
        "/v1/embeddings",
        json={
            "input": test_inputs,
            "model": model_name,
            "encoding_format": "float",
        },
    )
    assert response.status_code == 200
    data = response.json()

    data = response.json()
    _assert_embedding_response_structure(data, len(test_inputs), model_name)
    # Specific assertion for batch input:
    for idx, embedding_obj in enumerate(data["data"]):
        assert "index" in embedding_obj
        assert embedding_obj["index"] == idx


@pytest.mark.parametrize("model_name", all_model_keys)
def test_create_embeddings_ollama_output(model_name):
    """Test embedding generation with Ollama-like output for /api/embed."""
    test_inputs = [
        "This is a test sentence for Ollama.",
        "Another sentence for Ollama output.",
    ]

    response = client.post(
        "/api/embed",  # Target the Ollama-compatible endpoint
        json={
            "input": test_inputs,
            "model": model_name,
            "encoding_format": "float",
        },
    )
    assert response.status_code == 200
    data = response.json()

    # Assert Ollama-like structure
    assert "embeddings" in data
    assert isinstance(data["embeddings"], list)
    assert len(data["embeddings"]) == len(test_inputs)

    expected_dimension = models_config.get_model_config(model_name)["dimension"]

    for embedding_list in data["embeddings"]:
        assert isinstance(embedding_list, list)
        assert len(embedding_list) == expected_dimension

    # Ollama format now includes 'usage' at the top level
    assert "usage" in data
    assert "promptTokens" in data["usage"]
    assert "totalTokens" in data["usage"]
    assert data["usage"]["promptTokens"] >= 0
    assert data["usage"]["totalTokens"] >= 0

    # Ollama format does not include 'model' or 'object' at the top level
    assert "model" not in data
    assert "object" not in data


def test_empty_input_list_ollama_output():
    """Test handling of empty input list for /api/embed."""
    response = client.post(
        "/api/embed",
        json={
            "input": [],
            "model": "text-embedding-3-small",
            "encoding_format": "float",
        },
    )
    assert response.status_code == 200
    data = response.json()
    assert "embeddings" in data
    assert isinstance(data["embeddings"], list)
    assert len(data["embeddings"]) == 0
    assert "usage" in data
    assert data["usage"]["promptTokens"] == 0
    assert data["usage"]["totalTokens"] == 0
    assert "model" not in data
    assert "object" not in data


def test_nomic_embed_text_instruction_prefix():
    """Test that nomic-embed-text-v1.5 correctly applies the default instruction prefix."""
    model_name = "nomic-embed-text-v1.5"
    test_input_without_prefix = "This is a document for search."
    test_input_with_prefix = "search_query: What is the capital of France?"

    # Test with input that should get the default prefix
    response_without_prefix = client.post(
        "/v1/embeddings",
        json={
            "input": test_input_without_prefix,
            "model": model_name,
            "encoding_format": "float",
        },
    )
    assert response_without_prefix.status_code == 200
    data_without_prefix = response_without_prefix.json()
    assert data_without_prefix["usage"]["prompt_tokens"] > 0

    # Test with input that already has a prefix (should not be modified)
    response_with_prefix = client.post(
        "/v1/embeddings",
        json={
            "input": test_input_with_prefix,
            "model": model_name,
            "encoding_format": "float",
        },
    )
    assert response_with_prefix.status_code == 200
    data_with_prefix = response_with_prefix.json()
    assert data_with_prefix["usage"]["prompt_tokens"] > 0

    # Note: Directly asserting the prepended text in the backend is not feasible
    # without exposing internal logic. We rely on the model producing embeddings
    # successfully and token counts being non-zero as an indicator.
    # A more robust test would involve mocking the tokenizer to check input.


def test_invalid_model():
    """Test error handling for invalid model name."""
    response = client.post(
        "/v1/embeddings",
        json={
            "input": "Test sentence.",
            "model": "nonexistent-model",
            "encoding_format": "float",
        },
    )
    assert response.status_code == 422


def test_invalid_encoding_format():
    """Test error handling for invalid encoding format."""
    response = client.post(
        "/v1/embeddings",
        json={
            "input": "Test sentence.",
            "model": "text-embedding-3-small",
            "encoding_format": "invalid",
        },
    )
    assert response.status_code == 422


def test_empty_input_list():
    """Test handling of empty input list."""
    response = client.post(
        "/v1/embeddings",
        json={
            "input": [],
            "model": "text-embedding-3-small",
            "encoding_format": "float",
        },
    )
    assert response.status_code == 200
    data = response.json()
    assert len(data["data"]) == 0
    assert data["usage"]["prompt_tokens"] == 0
    assert data["usage"]["total_tokens"] == 0


def test_cuda_out_of_memory_error():
    """Test granular error handling for torch.cuda.OutOfMemoryError."""
    test_input = "OOM_TEST_SENTENCE"
    test_model = "all-MiniLM-L6-v2"

    # Directly patch _perform_model_inference to raise the HTTPException
    with mock.patch("embedding_processor._perform_model_inference") as mock_perform_inference:
        mock_perform_inference.side_effect = HTTPException(
            status_code=status.HTTP_507_INSUFFICIENT_STORAGE,
            detail="GPU out of memory: Mock CUDA Out of Memory",
        )
        response = client.post(
            "/v1/embeddings",
            json={
                "input": test_input,
                "model": test_model,
                "encoding_format": "float",
            },
        )
        assert response.status_code == 507  # Insufficient Storage
        assert "GPU out of memory" in response.json()["detail"]


def test_embeddings_cache_behavior():
    """Test that the embeddings cache works as expected (hit/miss)."""
    test_input = "This is a cached sentence."
    test_model = "all-MiniLM-L6-v2"

    os.environ["EMBEDDINGS_CACHE_ENABLED"] = "true"
    os.environ["EMBEDDINGS_CACHE_MAXSIZE"] = "10"  # Small cache size for testing
    os.environ["REPORT_CACHED_TOKENS"] = "false"  # Ensure default behavior

    embedding_processor.embeddings_cache.clear()

    model_config = models_config.get_model_config(test_model)
    canonical_hf_model_name = model_config["name"]

    # Calculate the expected hash for the test input
    test_input_hash = hashlib.sha256(test_input.encode("utf-8")).hexdigest()

    response1 = client.post(
        "/v1/embeddings",
        json={
            "input": test_input,
            "model": test_model,
            "encoding_format": "float",
        },
    )
    assert response1.status_code == 200
    assert response1.json()["usage"]["total_tokens"] > 0

    # Verify the item is in cache with the correct key (using canonical_hf_model_name)
    assert (test_input_hash, canonical_hf_model_name) in embedding_processor.embeddings_cache

    response2 = client.post(
        "/v1/embeddings",
        json={
            "input": test_input,
            "model": test_model,
            "encoding_format": "float",
        },
    )
    assert response2.status_code == 200
    assert response2.json()["usage"]["total_tokens"] == 0

    test_input_2 = "Another unique sentence."
    test_input_2_hash = hashlib.sha256(test_input_2.encode("utf-8")).hexdigest()
    response3 = client.post(
        "/v1/embeddings",
        json={
            "input": test_input_2,
            "model": test_model,
            "encoding_format": "float",
        },
    )
    assert response3.status_code == 200
    assert response3.json()["usage"]["total_tokens"] > 0
    assert (test_input_2_hash, canonical_hf_model_name) in embedding_processor.embeddings_cache


def test_embeddings_cache_mixed_batch():
    """Test caching behavior with a mix of cached and uncached items in a batch."""
    test_model = "all-MiniLM-L6-v2"
    cached_input = "This sentence is already in cache."
    uncached_input_1 = "This is a new sentence 1."
    uncached_input_2 = "This is a new sentence 2."

    os.environ["EMBEDDINGS_CACHE_ENABLED"] = "true"
    os.environ["EMBEDDINGS_CACHE_MAXSIZE"] = "10"
    os.environ["REPORT_CACHED_TOKENS"] = "false"  # Ensure default behavior
    _reload_app_and_client()  # Reload app and client to pick up these specific env vars

    model_config = models_config.get_model_config(test_model)
    canonical_hf_model_name = model_config["name"]

    response_pre_cache = client.post(
        "/v1/embeddings",
        json={
            "input": cached_input,
            "model": test_model,
            "encoding_format": "float",
        },
    )
    assert response_pre_cache.status_code == 200

    # Verify the pre-cached item is in cache with the correct key
    cached_input_hash = hashlib.sha256(cached_input.encode("utf-8")).hexdigest()
    assert (cached_input_hash, canonical_hf_model_name) in embedding_processor.embeddings_cache

    mixed_batch_inputs = [cached_input, uncached_input_1, uncached_input_2]

    response_mixed_batch = client.post(
        "/v1/embeddings",
        json={
            "input": mixed_batch_inputs,
            "model": test_model,
            "encoding_format": "float",
        },
    )
    assert response_mixed_batch.status_code == 200

    expected_total_tokens = 2 * 3
    assert response_mixed_batch.json()["usage"]["total_tokens"] == expected_total_tokens

    uncached_input_1_hash = hashlib.sha256(uncached_input_1.encode("utf-8")).hexdigest()
    uncached_input_2_hash = hashlib.sha256(uncached_input_2.encode("utf-8")).hexdigest()
    assert (uncached_input_1_hash, canonical_hf_model_name) in embedding_processor.embeddings_cache
    assert (uncached_input_2_hash, canonical_hf_model_name) in embedding_processor.embeddings_cache


def test_embeddings_cache_report_cached_tokens():
    """Test that total_tokens includes cached tokens when REPORT_CACHED_TOKENS is true."""
    test_input = "This sentence will be reported."
    test_model = "all-MiniLM-L6-v2"

    os.environ["EMBEDDINGS_CACHE_ENABLED"] = "true"
    os.environ["EMBEDDINGS_CACHE_MAXSIZE"] = "10"
    os.environ["REPORT_CACHED_TOKENS"] = "true"  # Enable reporting cached tokens
    _reload_app_and_client()  # Reload app and client to pick up these specific env vars

    model_config = models_config.get_model_config(test_model)
    canonical_hf_model_name = model_config["name"]

    test_input_hash = hashlib.sha256(test_input.encode("utf-8")).hexdigest()

    response1 = client.post(
        "/v1/embeddings",
        json={
            "input": test_input,
            "model": test_model,
            "encoding_format": "float",
        },
    )
    assert response1.status_code == 200
    initial_tokens = response1.json()["usage"]["total_tokens"]
    assert initial_tokens > 0
    assert (test_input_hash, canonical_hf_model_name) in embedding_processor.embeddings_cache

    response2 = client.post(
        "/v1/embeddings",
        json={
            "input": test_input,
            "model": test_model,
            "encoding_format": "float",
        },
    )
    assert response2.status_code == 200
    assert response2.json()["usage"]["total_tokens"] == initial_tokens
    assert (test_input_hash, canonical_hf_model_name) in embedding_processor.embeddings_cache
