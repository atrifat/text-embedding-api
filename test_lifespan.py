import importlib
import os
from unittest import mock
from unittest.mock import AsyncMock

import pytest
import torch
from cachetools import LRUCache

import app
import config
import embedding_processor
import model_loader
import models_config


@pytest.fixture(autouse=True)
def reset_modules_and_cache():
    """
    Resets the global embeddings_cache in embedding_processor and reloads modules
    to ensure a clean state before each test in this file.
    """
    original_env = os.environ.copy()

    # Reload modules first to ensure fresh state for settings and caches
    importlib.reload(models_config)
    importlib.reload(model_loader)
    importlib.reload(embedding_processor)

    yield  # Run the test

    # Teardown: Restore original env vars. Module reloads and cache re-initialization
    # for the *next* test will be handled by the next fixture setup.
    os.environ.clear()
    os.environ.update(original_env)


# Fixture to manage environment variables for specific tests
@pytest.fixture
def set_env_vars(request):
    original_env = os.environ.copy()
    # Clear existing environment variables to ensure a clean slate for the test's specific settings
    os.environ.clear()
    if hasattr(request, "param") and request.param is not None:
        for key, value in request.param.items():
            os.environ[key] = value

    # Reload config to pick up new environment variables
    importlib.reload(config)
    importlib.reload(app)  # Reload app to pick up new config settings

    # Re-initialize embeddings_cache after config and app reload to pick up new maxsize
    settings = config.get_app_settings()
    embedding_processor.embeddings_cache = LRUCache(maxsize=settings.embeddings_cache_maxsize)
    embedding_processor.embeddings_cache.clear()

    yield

    # Restore original environment variables after the test
    os.environ.clear()
    os.environ.update(original_env)

    # Reload config and app, and re-initialize cache again to ensure original settings are restored for subsequent tests
    importlib.reload(config)
    importlib.reload(app)
    settings = config.get_app_settings()
    embedding_processor.embeddings_cache = LRUCache(maxsize=settings.embeddings_cache_maxsize)
    embedding_processor.embeddings_cache.clear()


@pytest.mark.integration
@pytest.mark.parametrize(
    "set_env_vars",
    [
        {
            "DEFAULT_MODEL": "all-MiniLM-L6-v2",
            "WARMUP_ENABLED": "true",
            "EMBEDDINGS_CACHE_MAXSIZE": "50",
            "ENVIRONMENT": "development",
        }
    ],
    indirect=True,
)
async def test_lifespan_initialization_and_warmup(set_env_vars):
    """
    Test that the embeddings cache is initialized and the default model is warmed up during lifespan startup.
    """
    # Patch app.get_embeddings_batch directly, as app reloads its imports.
    with mock.patch("app.get_embeddings_batch", new_callable=AsyncMock) as mock_get_embeddings_batch:
        mock_get_embeddings_batch.return_value = (torch.empty(0, 384), 0)
        async with app.lifespan(app.app):
            assert isinstance(embedding_processor.embeddings_cache, LRUCache)
            assert embedding_processor.embeddings_cache.maxsize == 50
            mock_get_embeddings_batch.assert_called_once_with(["warmup"], "all-MiniLM-L6-v2", mock.ANY)


@pytest.mark.integration
@pytest.mark.integration
@pytest.mark.parametrize("set_env_vars", [{"WARMUP_ENABLED": "false"}], indirect=True)
async def test_lifespan_no_warmup_when_disabled(set_env_vars):
    """
    Test that warmup does not occur when WARMUP_ENABLED is set to false, using the set_env_vars fixture.
    """
    with mock.patch("app.get_embeddings_batch", new_callable=AsyncMock) as mock_get_embeddings_batch_no_warmup:
        async with app.lifespan(app.app):
            mock_get_embeddings_batch_no_warmup.assert_not_called()


@pytest.mark.integration
@pytest.mark.parametrize(
    "set_env_vars", [{"DEFAULT_MODEL": "nonexistent-model", "WARMUP_ENABLED": "true"}], indirect=True
)
async def test_lifespan_default_model_not_found_error(set_env_vars):
    """
    Test that a ValueError is raised if the default model is not configured.
    """
    with pytest.raises(ValueError, match="Default model 'nonexistent-model' is not configured in MODELS."):
        async with app.lifespan(app.app):
            pass


@pytest.mark.integration
async def test_lifespan_no_env_vars_set(set_env_vars):  # Use set_env_vars without param to cover else branch
    """
    Test that the lifespan context manager works correctly when no environment variables
    are explicitly set via the fixture (covering the 'else' branch of set_env_vars).
    """
    with mock.patch("app.get_embeddings_batch", new_callable=AsyncMock) as mock_get_embeddings_batch_no_env:
        async with app.lifespan(app.app):
            # By default, WARMUP_ENABLED is True, so it should be called.
            mock_get_embeddings_batch_no_env.assert_called_once()
            assert isinstance(embedding_processor.embeddings_cache, LRUCache)
            settings = config.get_app_settings()
            assert embedding_processor.embeddings_cache.maxsize == settings.embeddings_cache_maxsize


@pytest.mark.integration
@pytest.mark.parametrize(
    "set_env_vars", [{"DEFAULT_MODEL": "all-MiniLM-L6-v2", "WARMUP_ENABLED": "true"}], indirect=True
)
async def test_lifespan_warmup_failure_handled(set_env_vars):
    """
    Test that warmup failures are handled gracefully (logged but not raised to the client).
    """
    with mock.patch("app.get_embeddings_batch", new_callable=AsyncMock) as mock_get_embeddings_batch_failure:
        mock_get_embeddings_batch_failure.side_effect = Exception("Mock Warmup Failure")
        async with app.lifespan(app.app):
            mock_get_embeddings_batch_failure.assert_called_once()
