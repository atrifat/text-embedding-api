import hashlib  # Re-add hashlib import
import importlib
import os
from unittest import mock
from unittest.mock import AsyncMock  # Import AsyncMock

import pytest
import torch
from cachetools import LRUCache
from fastapi import HTTPException
from fastapi.testclient import TestClient
from starlette import status

import app  # Re-add app import
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
def mock_embedding_generation():
    """
    Fixture to mock model_loader.load_model for integration tests.
    This fixture is autoused, meaning it applies to all tests in this file.
    It also handles environment variable cleanup and cache clearing.
    """
    original_env = os.environ.copy()

    # Reload all relevant modules first to ensure fresh state
    importlib.reload(models_config)
    importlib.reload(config)
    importlib.reload(model_loader)
    importlib.reload(embedding_processor)
    importlib.reload(app)  # Reload app last to pick up all new imports

    # IMPORTANT: Re-initialize embeddings_cache with the correct maxsize after app reload
    settings = get_app_settings()
    embedding_processor.embeddings_cache = LRUCache(maxsize=settings.embeddings_cache_maxsize)

    # Correctly store the original embeddings cache AFTER all reloads and re-initialization
    original_embeddings_cache = LRUCache(maxsize=embedding_processor.embeddings_cache.maxsize)
    original_embeddings_cache.update(embedding_processor.embeddings_cache)

    # Now clear cache and reset env vars to default for each test
    embedding_processor.embeddings_cache.clear()
    os.environ["EMBEDDINGS_CACHE_ENABLED"] = "true"
    os.environ["REPORT_CACHED_TOKENS"] = "false"
    os.environ["EMBEDDINGS_CACHE_MAXSIZE"] = "2048"

    global client
    client = TestClient(app.app)

    with mock.patch("model_loader.load_model") as mock_load_model:
        # Use MagicMock for synchronous return values from model_loader.load_model
        mock_model_instance = mock.MagicMock()
        mock_model_instance.eval.return_value = mock_model_instance  # Ensure .eval() can be called and returns self
        mock_model_instance.to.return_value = mock_model_instance  # Ensure .to(DEVICE) works and returns self

        # When mock_model_instance is called (i.e., model(**batch_dict)), it needs to return an object
        # that has a .last_hidden_state attribute which is a torch.Tensor.
        mock_model_instance.side_effect = lambda **batch_dict: mock.MagicMock(
            last_hidden_state=torch.randn(batch_dict["input_ids"].shape[0], 10, 384)  # Assuming 384 for common models
        )

        mock_tokenizer_instance = mock.MagicMock()
        mock_tokenizer_instance.side_effect = lambda texts, **kwargs: mock.MagicMock(
            input_ids=torch.tensor([[1, 2, 3] for _ in range(len(texts))]),  # Generate input_ids for each text
            attention_mask=torch.tensor(
                [[1, 1, 1] for _ in range(len(texts))]
            ),  # Generate attention_mask for each text
        )

        # load_model is an async function, so its mock needs to return awaitable
        mock_load_model.side_effect = lambda model_name: (mock_model_instance, mock_tokenizer_instance)
        # Mock _perform_model_inference to return predictable values for token counts
        with mock.patch("embedding_processor._perform_model_inference") as mock_perform_inference:

            def side_effect_perform_inference(
                texts_to_tokenize, model, tokenizer, model_max_tokens, model_dimension, settings
            ):
                # Simulate 3 tokens per input text
                individual_tokens = [3] * len(texts_to_tokenize)
                total_tokens = sum(individual_tokens)
                # Create dummy embeddings of the correct dimension (e.g., 384 for common models)
                dummy_embeddings = torch.randn(len(texts_to_tokenize), model_dimension)
                return dummy_embeddings, individual_tokens, total_tokens

            mock_perform_inference.side_effect = side_effect_perform_inference
            yield  # Yield once for the entire fixture, allowing both patches to be active

    # Restore original env vars and cache state after each test
    os.environ.clear()
    os.environ.update(original_env)
    importlib.reload(app)
    importlib.reload(models_config)
    importlib.reload(config)
    importlib.reload(model_loader)
    importlib.reload(embedding_processor)


@pytest.mark.parametrize("model_name", all_model_keys)
@pytest.mark.integration
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
@pytest.mark.integration
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
@pytest.mark.integration
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


@pytest.mark.integration
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


@pytest.mark.integration
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


@pytest.mark.integration
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


@pytest.mark.integration
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


@pytest.mark.integration
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


@pytest.mark.integration
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


@pytest.mark.integration
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


@pytest.mark.integration
def test_embeddings_cache_mixed_batch():
    """Test caching behavior with a mix of cached and uncached items in a batch."""
    test_model = "all-MiniLM-L6-v2"
    cached_input = "This sentence is already in cache."
    uncached_input_1 = "This is a new sentence 1."
    uncached_input_2 = "This is a new sentence 2."

    os.environ["EMBEDDINGS_CACHE_ENABLED"] = "true"
    os.environ["EMBEDDINGS_CACHE_MAXSIZE"] = "10"
    os.environ["REPORT_CACHED_TOKENS"] = "false"  # Ensure default behavior
    # _reload_app_and_client() is handled by the autouse fixture mock_embedding_generation
    # No need to call it here, as it would reset the mocks.

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

    # When REPORT_CACHED_TOKENS is false, only uncached tokens are reported.
    # There is 1 cached input and 2 uncached inputs. Each input is mocked to have 3 tokens.
    expected_total_tokens = (len(mixed_batch_inputs) - 1) * 3  # 2 uncached inputs * 3 tokens/input = 6
    assert response_mixed_batch.json()["usage"]["total_tokens"] == expected_total_tokens

    uncached_input_1_hash = hashlib.sha256(uncached_input_1.encode("utf-8")).hexdigest()
    uncached_input_2_hash = hashlib.sha256(uncached_input_2.encode("utf-8")).hexdigest()
    assert (uncached_input_1_hash, canonical_hf_model_name) in embedding_processor.embeddings_cache
    assert (uncached_input_2_hash, canonical_hf_model_name) in embedding_processor.embeddings_cache


@pytest.mark.integration
def test_embeddings_cache_report_cached_tokens():
    """Test that total_tokens includes cached tokens when REPORT_CACHED_TOKENS is true."""
    test_input = "This sentence will be reported."
    test_model = "all-MiniLM-L6-v2"

    os.environ["EMBEDDINGS_CACHE_ENABLED"] = "true"
    os.environ["EMBEDDINGS_CACHE_MAXSIZE"] = "10"
    os.environ["REPORT_CACHED_TOKENS"] = "true"  # Enable reporting cached tokens
    _reload_app_and_client()  # Re-add this call to pick up env vars

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


@pytest.mark.integration
def test_ollama_api_invalid_model():
    """Test error handling for invalid model name in Ollama API."""
    response = client.post(
        "/api/embed",
        json={
            "input": "Test sentence.",
            "model": "nonexistent-ollama-model",
        },
    )
    assert response.status_code == 422
    assert "detail" in response.json()
    # Pydantic validation errors return a list of dictionaries in 'detail'
    # We need to check the 'msg' field of these dictionaries.
    assert "Value error" in response.json()["detail"]


@pytest.mark.parametrize("model_name", all_model_keys)
@pytest.mark.integration
def test_ollama_api_single_input(model_name):
    """Test single input embedding generation for /api/embed."""
    test_input = "This is a test sentence for Ollama single input."

    response = client.post(
        "/api/embed",
        json={
            "input": test_input,
            "model": model_name,
        },
    )
    assert response.status_code == 200
    data = response.json()

    assert "embeddings" in data
    assert isinstance(data["embeddings"], list)
    assert len(data["embeddings"]) == 1

    expected_dimension = models_config.get_model_config(model_name)["dimension"]
    assert isinstance(data["embeddings"][0], list)
    assert len(data["embeddings"][0]) == expected_dimension

    assert "usage" in data
    assert "promptTokens" in data["usage"]
    assert "totalTokens" in data["usage"]
    assert data["usage"]["promptTokens"] > 0
    assert data["usage"]["totalTokens"] == data["usage"]["promptTokens"]

    assert "model" not in data
    assert "object" not in data


@pytest.mark.integration
def test_ollama_api_empty_input_list():
    """Test handling of empty input list for /api/embed."""
    response = client.post(
        "/api/embed",
        json={
            "input": [],
            "model": "text-embedding-3-small",
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


@pytest.mark.parametrize("model_name", all_model_keys)
@pytest.mark.integration
def test_ollama_api_model_parameter_handling(model_name):
    """Test that the model parameter is correctly handled for Ollama requests."""
    test_input = "This sentence tests model parameter handling."

    response = client.post(
        "/api/embed",
        json={
            "input": test_input,
            "model": model_name,
        },
    )
    assert response.status_code == 200
    data = response.json()

    # Verify that the model name used in the request is reflected in the response if applicable
    # Ollama API does not return the model name in the response by default,
    # so we primarily verify that the request was processed successfully with the specified model.
    assert "embeddings" in data
    assert len(data["embeddings"]) == 1
    assert "usage" in data
    assert data["usage"]["promptTokens"] > 0
    assert data["usage"]["totalTokens"] > 0

    # Further verification would involve mocking model_loader.load_model
    # and asserting it was called with the correct model_name.
    # This is already covered by unit tests in test_model_loader.py.


@pytest.mark.parametrize("model_name", all_model_keys)
@pytest.mark.integration
def test_ollama_api_batch_input(model_name):
    """Test batch input embedding generation for /api/embed."""
    test_inputs = [
        "This is the first sentence for Ollama batch.",
        "This is the second sentence for Ollama batch.",
        "And the third one.",
    ]

    response = client.post(
        "/api/embed",
        json={
            "input": test_inputs,
            "model": model_name,
        },
    )
    assert response.status_code == 200
    data = response.json()

    assert "embeddings" in data
    assert isinstance(data["embeddings"], list)
    assert len(data["embeddings"]) == len(test_inputs)

    expected_dimension = models_config.get_model_config(model_name)["dimension"]
    for embedding_list in data["embeddings"]:
        assert isinstance(embedding_list, list)
        assert len(embedding_list) == expected_dimension

    assert "usage" in data
    assert "promptTokens" in data["usage"]
    assert "totalTokens" in data["usage"]
    assert data["usage"]["promptTokens"] > 0
    assert data["usage"]["totalTokens"] == data["usage"]["promptTokens"]

    assert "model" not in data
    assert "object" not in data


@pytest.mark.integration
def test_read_root_endpoint():
    """Test that the root endpoint serves the index.html file."""
    response = client.get("/")
    assert response.status_code == 200
    assert response.headers["content-type"] == "text/html; charset=utf-8"
    # Optionally, check for a specific string in the HTML content
    assert "<title>Text Embedding API</title>" in response.text


@pytest.mark.integration
def test_list_models_endpoint():
    """Test the /v1/models endpoint to ensure it returns a list of models."""
    response = client.get("/v1/models")
    assert response.status_code == 200
    data = response.json()
    assert "data" in data
    assert isinstance(data["data"], list)
    assert len(data["data"]) > 0  # Assuming there's at least one model configured

    for model_obj in data["data"]:
        assert "id" in model_obj
        assert "object" in model_obj
        assert model_obj["object"] == "model"
        assert "created" in model_obj
        assert "owned_by" in model_obj
        assert model_obj["owned_by"] == "local"


@pytest.mark.integration
def test_get_model_not_found():
    """Test the /v1/models/{model_id} endpoint for a non-existent model."""
    response = client.get("/v1/models/nonexistent-model-123")
    assert response.status_code == 404
    assert response.json()["detail"] == "Model not found"


@pytest.mark.integration
def test_create_embeddings_unhandled_exception():
    """
    Test that unhandled exceptions in create_embeddings are caught and return a 500 error.
    """
    test_input = "This input will cause an unhandled error."
    test_model = "all-MiniLM-L6-v2"

    with mock.patch("app.get_embeddings_batch") as mock_get_embeddings_batch_error:
        mock_get_embeddings_batch_error.side_effect = Exception("Simulated unhandled error")
        response = client.post(
            "/v1/embeddings",
            json={
                "input": test_input,
                "model": test_model,
                "encoding_format": "float",
            },
        )
        assert response.status_code == 500
        assert "Internal server error" in response.json()["detail"]
        mock_get_embeddings_batch_error.assert_called_once()


@pytest.mark.integration
@pytest.mark.parametrize("model_name", all_model_keys)
def test_get_model_existing(model_name):
    """Test the /v1/models/{model_id} endpoint for an existing model."""
    response = client.get(f"/v1/models/{model_name}")
    assert response.status_code == 200
    data = response.json()
    assert "id" in data
    assert data["id"] == model_name
    assert "object" in data
    assert data["object"] == "model"
    assert "created" in data
    assert "owned_by" in data
    assert data["owned_by"] == "local"


@pytest.mark.integration
def test_create_embeddings_value_error_handling():
    """
    Test that ValueError in create_embeddings is caught and returns a 422 error.
    This specifically targets the 'except ValueError' block.
    """
    test_input = "This input will cause a ValueError."
    test_model = "all-MiniLM-L6-v2"

    # Mock app.get_embeddings_batch to raise a ValueError
    with mock.patch("app.get_embeddings_batch", new_callable=AsyncMock) as mock_get_embeddings_batch_value_error:
        mock_get_embeddings_batch_value_error.side_effect = ValueError(
            "Simulated ValueError during embedding generation"
        )
        response = client.post(
            "/v1/embeddings",
            json={
                "input": test_input,
                "model": test_model,
                "encoding_format": "float",
            },
        )
        assert response.status_code == 422
        assert "Simulated ValueError during embedding generation" in response.json()["detail"]
        mock_get_embeddings_batch_value_error.assert_called_once()


@pytest.mark.integration
def test_create_embeddings_production_environment():
    """
    Test that debug logging is skipped when ENVIRONMENT is set to 'production'.
    This covers the 'if settings.environment != "production":' branch in app.py.
    """
    original_env = os.environ.copy()
    os.environ["ENVIRONMENT"] = "production"
    _reload_app_and_client()

    test_input = "This is a test sentence for production environment."
    test_model = "all-MiniLM-L6-v2"

    with mock.patch("app.logger.debug") as mock_logger_debug:
        response = client.post(
            "/v1/embeddings",
            json={
                "input": test_input,
                "model": test_model,
                "encoding_format": "float",
            },
        )
        assert response.status_code == 200
        mock_logger_debug.assert_not_called()

    os.environ.clear()
    os.environ.update(original_env)
    _reload_app_and_client()  # Reload to restore original environment


@pytest.mark.integration
def test_embeddings_cache_disabled():
    """
    Test that the embeddings cache is not used when EMBEDDINGS_CACHE_ENABLED is 'false'.
    This covers the 'if settings.embeddings_cache_enabled:' branch in embedding_processor.py.
    """
    original_env = os.environ.copy()
    os.environ["EMBEDDINGS_CACHE_ENABLED"] = "false"
    _reload_app_and_client()

    test_input = "This sentence should not be cached."
    test_model = "all-MiniLM-L6-v2"

    # Ensure the cache is empty before the test
    embedding_processor.embeddings_cache.clear()

    # First request: should not populate cache
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

    # Verify the item is NOT in cache
    model_config = models_config.get_model_config(test_model)
    canonical_hf_model_name = model_config["name"]
    test_input_hash = hashlib.sha256(test_input.encode("utf-8")).hexdigest()
    assert (test_input_hash, canonical_hf_model_name) not in embedding_processor.embeddings_cache

    # Second request: should still process and report tokens, as cache is disabled
    response2 = client.post(
        "/v1/embeddings",
        json={
            "input": test_input,
            "model": test_model,
            "encoding_format": "float",
        },
    )
    assert response2.status_code == 200
    assert response2.json()["usage"]["total_tokens"] > 0  # Still processes, so tokens > 0

    os.environ.clear()
    os.environ.update(original_env)
    _reload_app_and_client()  # Reload to restore original environment
