# test_e2e.py
import importlib
import os

import pytest
import pytest_asyncio
from cachetools import LRUCache
from fastapi.testclient import TestClient

# Import the actual app instance
import app
import config
import embedding_processor
import model_loader
import models_config
from config import get_app_settings
from embedding_processor import get_embeddings_batch
from test_data.minilm_embeddings import minilm_expected_embedding
from test_data.nomic_embeddings import nomic_embed_text_expected_embedding

# Define models for E2E tests
E2E_TEST_MODELS = ["all-MiniLM-L6-v2", "nomic-embed-text"]

os.environ["WARMUP_ENABLED"] = "true"


@pytest_asyncio.fixture(scope="function")
async def e2e_client(model_name):
    """
    Fixture to provide a TestClient that properly manages the FastAPI app's lifespan
    for end-to-end tests, parameterized by model_name.
    """
    # Set the model for the current test run
    os.environ["DEFAULT_MODEL"] = model_name

    # Reload all relevant modules to apply config changes and reset state
    for module in [config, models_config, model_loader, embedding_processor, app]:
        importlib.reload(module)

    settings = get_app_settings()

    # Clear caches to ensure a fresh state for each test run
    model_loader.model_cache = {}
    model_loader.tokenizer_cache = {}
    embedding_processor.embeddings_cache = LRUCache(maxsize=settings.embeddings_cache_maxsize)

    # Warm up the specific model for this test
    if settings.warmup_enabled:
        await get_embeddings_batch(["warmup"], model_name, settings)

    # Create the TestClient within the app's lifespan context
    async with app.lifespan(app.app):
        yield TestClient(app.app)


# --- EXISTING TEST FOR SINGLE-TEXT ACCURACY ---
@pytest.mark.e2e
@pytest.mark.asyncio
@pytest.mark.parametrize("model_name", E2E_TEST_MODELS)
async def test_e2e_single_text_embedding_accuracy(e2e_client: TestClient, model_name: str):
    """
    Performs an E2E test for a single text input to verify embedding accuracy.
    """
    test_input = "The quick brown fox jumps over the lazy dog."
    response = e2e_client.post("/v1/embeddings", json={"input": test_input, "model": model_name})

    assert response.status_code == 200
    data = response.json()
    assert len(data["data"]) == 1

    # Assertions for structure and content...
    embedding_obj = data["data"][0]
    expected_dimension = models_config.get_model_config(model_name)["dimension"]
    assert len(embedding_obj["embedding"]) == expected_dimension

    if model_name == "all-MiniLM-L6-v2":
        expected_embedding = minilm_expected_embedding
    elif model_name == "nomic-embed-text":
        expected_embedding = nomic_embed_text_expected_embedding

    assert embedding_obj["embedding"] == pytest.approx(expected_embedding, abs=1e-3)


# --- NEW TEST FOR TRUE BATCH PROCESSING ---
@pytest.mark.e2e
@pytest.mark.asyncio
@pytest.mark.parametrize("model_name", E2E_TEST_MODELS)
async def test_e2e_batch_embedding_processing(e2e_client: TestClient, model_name: str):
    """
    Performs a true end-to-end test with a batch of multiple, different-length texts.
    This is the definitive test to validate the model's batching behavior.
    """
    # A true batch of texts with varying lengths.
    test_batch_input = [
        "This is a short sentence.",
        "This sentence is a little bit longer to create a different sequence length.",
        "And this is the longest sentence of them all, designed specifically to test the padding and batching strategy of the model under real-world conditions.",  # noqa: E501
    ]

    # Send the batch request to the API endpoint
    response = e2e_client.post(
        "/v1/embeddings",
        json={
            "input": test_batch_input,
            "model": model_name,
        },
    )

    # --- Primary Assertion ---
    # This is the most important check. If the code has a bug (like the tensor mismatch),
    # it will likely crash and return a 500 error instead of a 200.
    assert response.status_code == 200

    # --- Secondary Assertions for Correctness ---
    data = response.json()
    assert "data" in data
    assert len(data["data"]) == len(test_batch_input)  # Ensure we got 3 embeddings back
    assert data["model"] == model_name

    # Check that each embedding in the batch has the correct structure and dimension
    expected_dimension = models_config.get_model_config(model_name)["dimension"]
    for i, embedding_obj in enumerate(data["data"]):
        assert embedding_obj["index"] == i
        assert "embedding" in embedding_obj
        assert isinstance(embedding_obj["embedding"], list)
        assert len(embedding_obj["embedding"]) == expected_dimension
        assert all(isinstance(val, float) for val in embedding_obj["embedding"])

    # Basic sanity check: ensure the embeddings for different texts are not identical
    embedding_1 = data["data"][0]["embedding"]
    embedding_2 = data["data"][1]["embedding"]
    assert embedding_1 != embedding_2
