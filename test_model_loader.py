import asyncio
import importlib
from unittest import mock

import pytest

import model_loader
import models_config


# Fixture to reset module state and mocks for each test
@pytest.fixture(autouse=True)
def reset_model_loader_state():
    """
    Resets the global caches and locks in model_loader for each test.
    Also reloads the module to ensure a clean state.
    """
    model_loader.model_cache.clear()
    model_loader.tokenizer_cache.clear()
    model_loader.model_load_locks.clear()
    importlib.reload(model_loader)
    importlib.reload(models_config)  # Ensure models_config is also fresh

    with mock.patch("torch.cuda.is_available", return_value=False):
        # Mock AutoModel and AutoTokenizer to prevent actual downloads
        with mock.patch("transformers.AutoModel.from_pretrained") as mock_auto_model, mock.patch(
            "transformers.AutoTokenizer.from_pretrained"
        ) as mock_auto_tokenizer:

            mock_model_instance = mock.MagicMock()  # Use MagicMock for synchronous calls
            mock_model_instance.eval.return_value = mock_model_instance  # Ensure .eval() returns self for chaining
            mock_model_instance.to.return_value = mock_model_instance  # Ensure .to(DEVICE) returns self for chaining
            mock_auto_model.return_value = mock_model_instance  # AutoModel.from_pretrained is synchronous

            mock_tokenizer_instance = mock.MagicMock()  # Use MagicMock for synchronous calls
            mock_auto_tokenizer.return_value = mock_tokenizer_instance  # AutoTokenizer.from_pretrained is synchronous

            yield mock_auto_model, mock_auto_tokenizer, mock_model_instance, mock_tokenizer_instance


@pytest.mark.asyncio
@pytest.mark.unit
async def test_load_model_success(reset_model_loader_state):
    """Test successful loading and caching of a model."""
    mock_auto_model, mock_auto_tokenizer, mock_model_instance, mock_tokenizer_instance = reset_model_loader_state

    model_name = "all-MiniLM-L6-v2"
    canonical_name = models_config.get_model_config(model_name)["name"]

    # First call: should load model
    model, tokenizer = await model_loader.load_model(model_name)

    mock_auto_model.assert_called_once_with(canonical_name, trust_remote_code=False)
    mock_auto_tokenizer.assert_called_once_with(canonical_name)
    assert model is mock_model_instance
    assert tokenizer is mock_tokenizer_instance
    assert model_loader.model_cache[canonical_name] is mock_model_instance
    assert model_loader.tokenizer_cache[canonical_name] is mock_tokenizer_instance

    # Second call: should return from cache without reloading
    mock_auto_model.reset_mock()
    mock_auto_tokenizer.reset_mock()
    model_2, tokenizer_2 = await model_loader.load_model(model_name)

    mock_auto_model.assert_not_called()
    mock_auto_tokenizer.assert_not_called()
    assert model_2 is mock_model_instance
    assert tokenizer_2 is mock_tokenizer_instance


@pytest.mark.asyncio
@pytest.mark.unit
async def test_load_model_trust_remote_code(reset_model_loader_state):
    """Test that trust_remote_code is correctly passed for models requiring it."""
    mock_auto_model, _, _, _ = reset_model_loader_state

    model_name = "nomic-embed-text-v1.5"
    canonical_name = models_config.get_model_config(model_name)["name"]

    await model_loader.load_model(model_name)
    mock_auto_model.assert_called_once_with(canonical_name, trust_remote_code=True)


@pytest.mark.asyncio
@pytest.mark.unit
async def test_load_model_unknown_model():
    """Test error handling for an unknown model name."""
    with pytest.raises(ValueError, match="Model 'nonexistent-model' .* is not a recognized model."):
        await model_loader.load_model("nonexistent-model")


async def _concurrent_load_task(model_name, results):
    """Helper for concurrent loading test."""
    model, tokenizer = await model_loader.load_model(model_name)
    results.append((model, tokenizer))


@pytest.mark.asyncio
@pytest.mark.unit
async def test_load_model_concurrent_loading(reset_model_loader_state):
    """Test that concurrent calls to load_model for the same model are serialized."""
    mock_auto_model, mock_auto_tokenizer, mock_model_instance, mock_tokenizer_instance = reset_model_loader_state

    model_name = "all-MiniLM-L6-v2"

    # Simulate a delay in model loading
    def delayed_from_pretrained(*args, **kwargs):  # This function should not be async
        # Simulate I/O delay, but the return value is synchronous
        return mock_model_instance  # Return the MagicMock instance

    mock_auto_model.side_effect = delayed_from_pretrained

    results = []
    tasks = [
        _concurrent_load_task(model_name, results),
        _concurrent_load_task(model_name, results),
        _concurrent_load_task(model_name, results),
    ]

    await asyncio.gather(*tasks)

    # AutoModel.from_pretrained should only be called once
    mock_auto_model.assert_called_once()
    mock_auto_tokenizer.assert_called_once()

    # All tasks should receive the same model and tokenizer instances
    assert len(results) == 3
    for model, tokenizer in results:
        assert model is mock_model_instance
        assert tokenizer is mock_tokenizer_instance


@pytest.mark.asyncio
@pytest.mark.unit
async def test_load_model_device_assignment(reset_model_loader_state):
    """Test that the loaded model is moved to the correct device."""
    mock_auto_model, _, mock_model_instance, _ = reset_model_loader_state

    model_name = "all-MiniLM-L6-v2"

    await model_loader.load_model(model_name)

    # Verify that .to(DEVICE) was called on the mock model
    mock_model_instance.to.assert_called_once_with(model_loader.DEVICE)
    mock_model_instance.eval.assert_called_once()
