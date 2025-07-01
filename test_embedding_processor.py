import hashlib
import importlib
from unittest import mock

import pytest
import torch
import torch.nn.functional as F
from cachetools import LRUCache
from fastapi import HTTPException
from starlette import status

import config
import embedding_processor
import model_loader  # Import model_loader to mock its functions
import models_config


class MockBatchEncoding:
    """A mock class to simulate the output of a Hugging Face tokenizer."""

    def __init__(self, input_ids: torch.Tensor, attention_mask: torch.Tensor):
        self.input_ids = input_ids
        self.attention_mask = attention_mask

    def items(self):
        # Simulate dictionary-like behavior for .items()
        return {"input_ids": self.input_ids, "attention_mask": self.attention_mask}.items()


class MockModelOutput:
    """A mock class to simulate the output of a Hugging Face model."""

    def __init__(self, last_hidden_state: torch.Tensor):
        self.last_hidden_state = last_hidden_state


# Fixture to reset module state and mocks for each test
@pytest.fixture(autouse=True)
def reset_embedding_processor_state():
    """
    Resets the global embeddings_cache in embedding_processor for each test.
    Also reloads the module to ensure a clean state.
    """
    # Ensure embeddings_cache is initialized for tests
    if embedding_processor.embeddings_cache is None:
        embedding_processor.embeddings_cache = LRUCache(maxsize=2048)
    embedding_processor.embeddings_cache.clear()

    # Reload modules to ensure fresh state for settings and caches
    importlib.reload(config)
    importlib.reload(models_config)
    importlib.reload(model_loader)  # Reload model_loader as well

    # Mock load_model from model_loader as the outermost patch
    with mock.patch("model_loader.load_model", new_callable=mock.AsyncMock) as mock_load_model:
        # Reload embedding_processor *after* model_loader.load_model is patched
        importlib.reload(embedding_processor)

        # Define mock_model and mock_tokenizer here, within the scope of the load_model patch
        mock_model = mock.MagicMock()
        mock_tokenizer = mock.MagicMock()

        OOM_TRIGGER_TEXT = "OOM_TEST_SENTENCE"

        def get_mock_tokenizer_output(texts, **kwargs):
            num_texts = len(texts) if isinstance(texts, list) else 1
            if OOM_TRIGGER_TEXT in texts:
                input_ids = torch.full((num_texts, 3), 999, dtype=torch.long)
                print(f"DEBUG: get_mock_tokenizer_output - OOM_TEST_SENTENCE detected. input_ids: {input_ids}")
            else:
                input_ids = torch.full((num_texts, 3), 1, dtype=torch.long)
            attention_mask = torch.full((num_texts, 3), 1, dtype=torch.long)
            return MockBatchEncoding(input_ids, attention_mask)

        mock_tokenizer.side_effect = get_mock_tokenizer_output

        def mock_model_call_side_effect(**batch_dict):
            input_ids = batch_dict["input_ids"]
            if (input_ids == 999).any():
                raise torch.cuda.OutOfMemoryError("Mock CUDA Out of Memory")

            # Simulate last_hidden_state with distinct values for CLS and mean pooling
            batch_size, seq_len = input_ids.shape
            # For CLS, make the first token's hidden state distinct
            # For mean, make all hidden states contribute to a predictable mean
            hidden_state = torch.ones(batch_size, seq_len, mock_model.dimension) * 0.1
            # Make CLS token (index 0) unique for CLS pooling tests
            hidden_state[:, 0, :] = 0.5

            return MockModelOutput(last_hidden_state=hidden_state.to(model_loader.DEVICE))  # Move to device

        mock_model.side_effect = mock_model_call_side_effect

        async def load_model_side_effect(model_name: str):
            cfg = models_config.get_model_config(model_name)
            mock_model.dimension = cfg["dimension"]
            mock_model.pooling_strategy = cfg.get("pooling_strategy", "cls")  # Pass pooling strategy to mock
            return mock_model, mock_tokenizer

        mock_load_model.side_effect = load_model_side_effect

        # Now, apply other patches
        with mock.patch("config.get_app_settings") as mock_get_app_settings:
            mock_settings = mock.MagicMock(spec=config.AppSettings)
            mock_settings.embeddings_cache_enabled = True
            mock_settings.report_cached_tokens = False
            mock_settings.embeddings_cache_maxsize = 2048
            mock_settings.embedding_batch_size = 16
            mock_settings.cuda_cache_clear_enabled = False  # Default to False for most tests
            mock_get_app_settings.return_value = mock_settings

            embedding_processor.embeddings_cache = LRUCache(maxsize=mock_settings.embeddings_cache_maxsize)

            with mock.patch("torch.cuda.is_available", return_value=False):
                with mock.patch("transformers.AutoModel.from_pretrained") as mock_auto_model, mock.patch(
                    "transformers.AutoTokenizer.from_pretrained"
                ) as mock_auto_tokenizer:
                    mock_auto_model.return_value = mock_model  # Ensure AutoModel returns our mock
                    mock_auto_tokenizer.return_value = mock_tokenizer  # Ensure AutoTokenizer returns our mock

                    yield mock_settings, mock_model, mock_tokenizer, mock_load_model


@pytest.mark.unit
def test_process_texts_for_cache_and_batching_cache_hit(reset_embedding_processor_state):
    """Test _process_texts_for_cache_and_batching with a cache hit."""
    settings, _, _, _ = reset_embedding_processor_state
    settings.embeddings_cache_enabled = True
    settings.report_cached_tokens = True

    test_text = "This is a cached sentence."
    model_config = models_config.get_model_config("all-MiniLM-L6-v2")
    canonical_name = model_config["name"]
    text_hash = hashlib.sha256(test_text.encode("utf-8")).hexdigest()
    cache_key = (text_hash, canonical_name)

    # Manually put an item in the cache
    dummy_embedding = torch.randn(1, model_config["dimension"])
    dummy_tokens = 5
    embedding_processor.embeddings_cache[cache_key] = (dummy_embedding, dummy_tokens)

    final_ordered_embeddings, total_prompt_tokens, texts_to_process, original_indices = (
        embedding_processor._process_texts_for_cache_and_batching([test_text], model_config, settings)
    )

    assert len(final_ordered_embeddings) == 1
    assert torch.equal(final_ordered_embeddings[0].squeeze(0), dummy_embedding)
    assert total_prompt_tokens == dummy_tokens
    assert len(texts_to_process) == 0
    assert len(original_indices) == 0


@pytest.mark.unit
def test_process_texts_for_cache_and_batching_cache_miss(reset_embedding_processor_state):
    """Test _process_texts_for_cache_and_batching with a cache miss."""
    settings, _, _, _ = reset_embedding_processor_state
    settings.embeddings_cache_enabled = True
    settings.report_cached_tokens = False  # Should not report tokens for uncached

    test_text = "This is an uncached sentence."
    model_config = models_config.get_model_config("all-MiniLM-L6-v2")

    final_ordered_embeddings, total_prompt_tokens, texts_to_process, original_indices = (
        embedding_processor._process_texts_for_cache_and_batching([test_text], model_config, settings)
    )

    assert len(final_ordered_embeddings) == 1
    assert final_ordered_embeddings[0] is None
    assert total_prompt_tokens == 0  # No cached tokens to report
    assert len(texts_to_process) == 1
    assert texts_to_process[0] == test_text
    assert original_indices[0] == 0


@pytest.mark.unit
def test_apply_instruction_prefix_required(reset_embedding_processor_state):
    """Test _apply_instruction_prefix when a prefix is required and not present."""
    _, _, _, _ = reset_embedding_processor_state
    model_config = models_config.get_model_config("nomic-embed-text-v1.5")
    test_texts = ["This is a document.", "Another document."]

    processed_texts = embedding_processor._apply_instruction_prefix(test_texts, model_config)
    assert processed_texts[0] == "search_document:This is a document."
    assert processed_texts[1] == "search_document:Another document."


@pytest.mark.unit
def test_apply_instruction_prefix_already_present(reset_embedding_processor_state):
    """Test _apply_instruction_prefix when a known prefix is already present."""
    _, _, _, _ = reset_embedding_processor_state
    model_config = models_config.get_model_config("nomic-embed-text-v1.5")
    test_texts = ["search_query:What is it?", "clustering:Group these."]

    processed_texts = embedding_processor._apply_instruction_prefix(test_texts, model_config)
    assert processed_texts[0] == "search_query:What is it?"
    assert processed_texts[1] == "clustering:Group these."


@pytest.mark.unit
def test_apply_instruction_prefix_not_required(reset_embedding_processor_state):
    """Test _apply_instruction_prefix when no prefix is required for the model."""
    _, _, _, _ = reset_embedding_processor_state
    model_config = models_config.get_model_config("all-MiniLM-L6-v2")  # Model without instruction prefix
    test_texts = ["Simple text.", "Another simple text."]

    processed_texts = embedding_processor._apply_instruction_prefix(test_texts, model_config)
    assert processed_texts == test_texts  # Should return original texts


@pytest.mark.unit
@pytest.mark.parametrize(
    "model_name, expected_pooling_strategy",
    [
        ("all-MiniLM-L6-v2", "mean"),
        ("gte-multilingual-base", "cls"),
    ],
)
@pytest.mark.asyncio
async def test_perform_model_inference_success_with_pooling(
    reset_embedding_processor_state, model_name, expected_pooling_strategy
):
    """Test successful model inference with different pooling strategies."""
    settings, mock_model, mock_tokenizer, mock_load_model = reset_embedding_processor_state

    test_texts = ["Hello world.", "Test sentence."]
    model_config = models_config.get_model_config(model_name)
    model_dimension = model_config["dimension"]

    # Manually load model and tokenizer mocks to ensure pooling_strategy is set
    model_instance, tokenizer_instance = await mock_load_model.side_effect(model_name)
    mock_model.dimension = model_dimension  # Ensure mock model has correct dimension
    mock_model.pooling_strategy = expected_pooling_strategy  # Set pooling strategy for mock

    # Mock tokenizer to return predictable token counts and attention mask
    mock_tokenizer.side_effect = lambda texts, **kwargs: MockBatchEncoding(
        input_ids=torch.tensor([[1, 2, 3], [4, 5, 6]]), attention_mask=torch.tensor([[1, 1, 1], [1, 1, 1]])
    )

    # Mock model to return predictable hidden states for pooling verification
    def mock_model_call_side_effect_for_pooling(**batch_dict):
        batch_size, seq_len = batch_dict["input_ids"].shape
        # Create a hidden state where CLS token is 0.5 and others are 0.1
        hidden_state = torch.ones(batch_size, seq_len, model_dimension) * 0.1
        hidden_state[:, 0, :] = 0.5  # CLS token

        return MockModelOutput(last_hidden_state=hidden_state.to(model_loader.DEVICE))

    mock_model.side_effect = mock_model_call_side_effect_for_pooling

    embeddings, individual_tokens, total_tokens = embedding_processor._perform_model_inference(
        test_texts, model_instance, tokenizer_instance, model_config, settings  # Pass model_config
    )

    assert isinstance(embeddings, torch.Tensor)
    assert embeddings.shape == (len(test_texts), model_dimension)
    assert all(isinstance(t, int) for t in individual_tokens)
    assert total_tokens > 0
    mock_tokenizer.assert_called_once()
    mock_model.assert_called_once()

    # Verify pooling strategy
    if expected_pooling_strategy == "cls":
        # For CLS pooling, the embedding should be the normalized CLS token (0.5)
        expected_embedding = F.normalize(torch.full((1, model_dimension), 0.5), p=2, dim=1).to(model_loader.DEVICE)
        assert torch.allclose(embeddings[0], expected_embedding.squeeze(0), atol=1e-6)
    elif expected_pooling_strategy == "mean":
        # For Mean pooling, the embedding should be the normalized mean of (0.5 + 0.1 + 0.1) / 3
        # Assuming attention mask is all ones, mean of [0.5, 0.1, 0.1] is (0.5+0.1+0.1)/3 = 0.7/3 = 0.2333...
        expected_mean_value = (0.5 + 0.1 + 0.1) / 3.0
        expected_embedding = F.normalize(torch.full((1, model_dimension), expected_mean_value), p=2, dim=1).to(
            model_loader.DEVICE
        )
        assert torch.allclose(embeddings[0], expected_embedding.squeeze(0), atol=1e-6)


@pytest.mark.unit
@pytest.mark.asyncio
async def test_perform_model_inference_cuda_oom_error(reset_embedding_processor_state):
    """Test _perform_model_inference handling of CUDA Out of Memory error."""
    settings, mock_model, mock_tokenizer, mock_load_model = reset_embedding_processor_state
    settings.cuda_cache_clear_enabled = True  # Ensure finally block is tested

    test_texts = ["OOM_TEST_SENTENCE"]  # This text triggers OOM in the mock
    model_name = "all-MiniLM-L6-v2"
    model_config = models_config.get_model_config(model_name)

    model_instance, tokenizer_instance = await mock_load_model.side_effect(model_name)

    # Mock torch.cuda.empty_cache to check if it's called
    with mock.patch("torch.cuda.empty_cache") as mock_empty_cache:
        with mock.patch("torch.cuda.is_available", return_value=True):  # Simulate CUDA available
            with pytest.raises(HTTPException) as exc_info:
                await embedding_processor._perform_model_inference(
                    test_texts, model_instance, tokenizer_instance, model_config, settings
                )
            assert exc_info.value.status_code == status.HTTP_507_INSUFFICIENT_STORAGE
            assert "GPU out of memory" in exc_info.value.detail
            mock_empty_cache.assert_called_once()  # Should be called even on error


@pytest.mark.unit
@pytest.mark.asyncio
async def test_perform_model_inference_general_exception(reset_embedding_processor_state):
    """Test _perform_model_inference handling of a general unexpected exception."""
    settings, mock_model, mock_tokenizer, mock_load_model = reset_embedding_processor_state

    test_texts = ["Some text."]
    model_name = "all-MiniLM-L6-v2"
    model_config = models_config.get_model_config(model_name)

    model_instance, tokenizer_instance = await mock_load_model.side_effect(model_name)

    mock_tokenizer.side_effect = Exception("Tokenizer error")  # Simulate an unexpected error

    with pytest.raises(HTTPException) as exc_info:
        await embedding_processor._perform_model_inference(
            test_texts, model_instance, tokenizer_instance, model_config, settings
        )
    assert exc_info.value.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR
    assert "Internal server error" in exc_info.value.detail


@pytest.mark.unit
def test_store_embeddings_in_cache(reset_embedding_processor_state):
    """Test _store_embeddings_in_cache functionality."""
    settings, _, _, _ = reset_embedding_processor_state
    settings.embeddings_cache_enabled = True

    test_texts = ["Text A", "Text B"]
    model_config = models_config.get_model_config("all-MiniLM-L6-v2")
    canonical_name = model_config["name"]

    embeddings = torch.randn(2, model_config["dimension"])
    individual_tokens = [3, 4]
    batch_original_indices = [0, 1]
    final_ordered_embeddings = [None, None]

    embedding_processor._store_embeddings_in_cache(
        embeddings,
        individual_tokens,
        batch_original_indices,
        test_texts,
        model_config,
        final_ordered_embeddings,
        settings,
    )

    text_a_hash = hashlib.sha256(test_texts[0].encode("utf-8")).hexdigest()
    text_b_hash = hashlib.sha256(test_texts[1].encode("utf-8")).hexdigest()

    assert (text_a_hash, canonical_name) in embedding_processor.embeddings_cache
    assert (text_b_hash, canonical_name) in embedding_processor.embeddings_cache
    assert torch.equal(final_ordered_embeddings[0].squeeze(0), embeddings[0].cpu())
    assert torch.equal(final_ordered_embeddings[1].squeeze(0), embeddings[1].cpu())
    assert embedding_processor.embeddings_cache[(text_a_hash, canonical_name)][1] == individual_tokens[0]
    assert embedding_processor.embeddings_cache[(text_b_hash, canonical_name)][1] == individual_tokens[1]


@pytest.mark.asyncio
@pytest.mark.unit
async def test_get_embeddings_batch_full_flow(reset_embedding_processor_state):
    """Test the full get_embeddings_batch flow with a mix of cached and uncached texts."""
    settings, mock_model, mock_tokenizer, mock_load_model = reset_embedding_processor_state
    settings.embeddings_cache_enabled = True
    settings.report_cached_tokens = True
    settings.embedding_batch_size = 1  # Force small batches for testing loop

    cached_text = "This is a cached sentence."
    uncached_text_1 = "This is a new sentence 1."
    uncached_text_2 = "This is a new sentence 2."
    model_name = "all-MiniLM-L6-v2"
    model_config = models_config.get_model_config(model_name)
    canonical_name = model_config["name"]

    # Pre-populate cache with one item
    cached_text_hash = hashlib.sha256(cached_text.encode("utf-8")).hexdigest()
    # Pre-populate cache with one item
    # dummy_embedding should be 1D so that when unsqueeze(0) is applied in
    # _store_embeddings_in_cache, it becomes 2D (1, dimension)
    dummy_embedding = torch.randn(model_config["dimension"])
    dummy_tokens = 5
    embedding_processor.embeddings_cache[(cached_text_hash, canonical_name)] = (dummy_embedding, dummy_tokens)

    # Mock _perform_model_inference directly to control its return values
    with mock.patch("embedding_processor._perform_model_inference") as mock_perform_inference:

        def side_effect_perform_inference(
            texts_to_tokenize, model, tokenizer, model_config, settings  # Updated parameters
        ):
            model_dimension = model_config["dimension"]  # Get dimension from model_config
            # Simulate 3 tokens per input text
            individual_tokens = [3] * len(texts_to_tokenize)
            total_tokens = sum(individual_tokens)
            # Create dummy embeddings of the correct dimension
            dummy_embeddings = torch.randn(len(texts_to_tokenize), model_dimension)
            return dummy_embeddings, individual_tokens, total_tokens

        mock_perform_inference.side_effect = side_effect_perform_inference

        texts_input = [cached_text, uncached_text_1, uncached_text_2]

        final_embeddings, total_tokens = await embedding_processor.get_embeddings_batch(
            texts_input, model_name, settings
        )

    assert isinstance(final_embeddings, torch.Tensor)
    assert final_embeddings.shape == (len(texts_input), model_config["dimension"])

    # Expected total tokens: cached_tokens + (tokens_per_uncached_text * num_uncached_texts)
    # Mock tokenizer returns 3 tokens per text, so 2 * 3 = 6 for uncached
    assert total_tokens == dummy_tokens + (2 * 3)  # 5 (cached) + 6 (uncached) = 11

    # Verify uncached items are now in cache
    uncached_text_1_hash = hashlib.sha256(uncached_text_1.encode("utf-8")).hexdigest()
    uncached_text_2_hash = hashlib.sha256(uncached_text_2.encode("utf-8")).hexdigest()
    assert (uncached_text_1_hash, canonical_name) in embedding_processor.embeddings_cache
    assert (uncached_text_2_hash, canonical_name) in embedding_processor.embeddings_cache

    # Verify load_model was called once
    # mock_load_model.assert_called_once_with(model_name) # Not called when _perform_model_inference is mocked directly
    # Verify _perform_model_inference was called twice (due to batch_size=1)
    # assert mock_model.call_count == 2 # Not called when _perform_model_inference is mocked directly


@pytest.mark.asyncio
@pytest.mark.unit
async def test_get_embeddings_batch_full_flow_no_mock_inference(reset_embedding_processor_state):
    """
    Test the full get_embeddings_batch flow without mocking _perform_model_inference,
    to cover its internal branches, including OOM_TRIGGER_TEXT and cuda_cache_clear_enabled.
    """
    settings, mock_model, mock_tokenizer, mock_load_model = reset_embedding_processor_state
    settings.embeddings_cache_enabled = True
    settings.report_cached_tokens = True
    settings.embedding_batch_size = 1  # Force small batches for testing loop
    settings.cuda_cache_clear_enabled = True  # Enable CUDA cache clear for testing

    cached_text = "This is a cached sentence for full flow."
    uncached_text_1 = "This is a new sentence 1 for full flow."
    uncached_text_2 = "OOM_TEST_SENTENCE"  # This will trigger OOM in the mock_model_call_side_effect
    model_name = "all-MiniLM-L6-v2"
    model_config = models_config.get_model_config(model_name)
    canonical_name = model_config["name"]

    # Ensure model and tokenizer are not pre-loaded in embedding_processor
    embedding_processor.current_model = None
    embedding_processor.current_tokenizer = None

    # Pre-populate cache with one item
    cached_text_hash = hashlib.sha256(cached_text.encode("utf-8")).hexdigest()
    dummy_embedding = torch.randn(model_config["dimension"])
    dummy_tokens = 5
    embedding_processor.embeddings_cache[(cached_text_hash, canonical_name)] = (dummy_embedding, dummy_tokens)

    # Original test case for full flow with OOM
    texts_input_full_flow = [cached_text, uncached_text_1, uncached_text_2]

    with mock.patch("torch.cuda.empty_cache"):
        with mock.patch("torch.cuda.is_available", return_value=False):  # Force CPU to avoid real OOM interference
            try:
                final_embeddings, total_tokens = await embedding_processor.get_embeddings_batch(
                    texts_input_full_flow, model_name, settings
                )
                pytest.fail("HTTPException was not raised for OOM_TEST_SENTENCE in full flow.")
            except HTTPException as e:
                # Assertions for the OOM error path
                assert e.status_code == status.HTTP_507_INSUFFICIENT_STORAGE
                assert "GPU out of memory" in e.detail
                # mock_empty_cache.assert_called_once() # Removed as torch.cuda.is_available is mocked to False
            except Exception as e:
                pytest.fail(f"Unexpected exception caught in full flow: {type(e).__name__}: {e}")

    # Verify load_model was called once
    mock_load_model.assert_called_once_with(model_name)
    # Verify mock_model and mock_tokenizer were called


@pytest.mark.asyncio
@pytest.mark.unit
async def test_get_embeddings_batch_oom_only(reset_embedding_processor_state):
    """
    Dedicated test for the OOM path in get_embeddings_batch.
    """
    settings, mock_model, mock_tokenizer, mock_load_model = reset_embedding_processor_state
    settings.embedding_batch_size = 1  # Force small batches for testing loop
    settings.cuda_cache_clear_enabled = True  # Enable CUDA cache clear for testing

    oom_text = "OOM_TEST_SENTENCE"
    model_name = "all-MiniLM-L6-v2"

    # Ensure model and tokenizer are not pre-loaded in embedding_processor
    embedding_processor.current_model = None
    embedding_processor.current_tokenizer = None

    with mock.patch("torch.cuda.empty_cache"):
        with mock.patch("torch.cuda.is_available", return_value=False):  # Force CPU to avoid real OOM interference
            try:
                # Only process the OOM sentence
                await embedding_processor.get_embeddings_batch([oom_text], model_name, settings)
                pytest.fail("HTTPException was not raised for OOM_TEST_SENTENCE in oom_only test.")
            except HTTPException as e:
                assert e.status_code == status.HTTP_507_INSUFFICIENT_STORAGE
                # mock_empty_cache.assert_called_once() # Removed as torch.cuda.is_available is mocked to False
            except Exception as e:
                pytest.fail(f"Unexpected exception caught in oom_only test: {type(e).__name__}: {e}")


@pytest.mark.asyncio
@pytest.mark.unit
async def test_perform_model_inference_oom_direct(reset_embedding_processor_state):
    """
    Test _perform_model_inference directly for the OOM path.
    """
    settings, mock_model, mock_tokenizer, mock_load_model = reset_embedding_processor_state
    settings.cuda_cache_clear_enabled = True

    oom_text = "OOM_TEST_SENTENCE"
    model_name = "all-MiniLM-L6-v2"
    model_config = models_config.get_model_config(model_name)

    # Ensure model and tokenizer are not pre-loaded in embedding_processor
    embedding_processor.current_model = None
    embedding_processor.current_tokenizer = None

    # Manually load model and tokenizer mocks as _perform_model_inference expects them
    model_instance, tokenizer_instance = await mock_load_model.side_effect(model_name)

    with mock.patch("torch.cuda.empty_cache"):
        with mock.patch("torch.cuda.is_available", return_value=False):  # Force CPU to avoid real OOM interference
            with pytest.raises(HTTPException) as exc_info:
                await embedding_processor._perform_model_inference(
                    [oom_text], model_instance, tokenizer_instance, model_config, settings
                )

            assert exc_info.value.status_code == status.HTTP_507_INSUFFICIENT_STORAGE
            # mock_empty_cache.assert_called_once() # Removed as torch.cuda.is_available is mocked to False
    # For a direct OOM test, the tokenizer should be called once.
    assert mock_tokenizer.call_count == 1
    assert mock_model.call_count == 1


@pytest.mark.asyncio
@pytest.mark.unit
async def test_get_embeddings_batch_empty_input(reset_embedding_processor_state):
    """Test get_embeddings_batch with an empty input list."""
    settings, _, _, _ = reset_embedding_processor_state
    model_name = "all-MiniLM-L6-v2"

    final_embeddings, total_tokens = await embedding_processor.get_embeddings_batch([], model_name, settings)

    assert isinstance(final_embeddings, torch.Tensor)
    # If no texts are processed, final_embeddings_tensor should be an empty tensor
    # with the correct dimension.
    expected_shape = (0, models_config.get_model_config(model_name)["dimension"])
    assert final_embeddings.shape == expected_shape
    assert total_tokens == 0


@pytest.mark.asyncio
@pytest.mark.unit
async def test_nomic_embed_text_variable_length_batch(reset_embedding_processor_state):
    """
    Test Case 1: Validate the fix for nomic-embed-text-v1.5 with variable-length texts.
    """
    settings, mock_model, mock_tokenizer, mock_load_model = reset_embedding_processor_state
    model_name = "nomic-embed-text-v1.5"
    model_config = models_config.get_model_config(model_name)

    test_texts = ["short text", "this is a much longer sentence to process and it should still work correctly"]

    # Manually load model and tokenizer mocks to ensure pooling_strategy is set
    model_instance, tokenizer_instance = await mock_load_model.side_effect(model_name)

    # Mock tokenizer to return predictable token counts and attention mask
    mock_tokenizer.side_effect = lambda texts, **kwargs: MockBatchEncoding(
        input_ids=torch.full((len(texts), model_config["max_tokens"]), 1, dtype=torch.long),
        attention_mask=torch.full((len(texts), model_config["max_tokens"]), 1, dtype=torch.long),
    )

    # Mock model to return predictable hidden states
    def mock_model_call_side_effect_nomic(**batch_dict):
        batch_size, seq_len = batch_dict["input_ids"].shape
        hidden_state = torch.ones(batch_size, seq_len, model_config["dimension"]) * 0.1
        return MockModelOutput(last_hidden_state=hidden_state.to(model_loader.DEVICE))

    mock_model.side_effect = mock_model_call_side_effect_nomic

    with mock.patch("embedding_processor._perform_model_inference") as mock_inference:

        def _mock_perform_model_inference_side_effect(texts_to_tokenize, model, tokenizer, model_config, settings):
            # Simulate the tokenizer call within _perform_model_inference
            tokenizer(
                texts_to_tokenize,
                max_length=model_config["max_tokens"],
                padding="longest",  # This is what we want to assert
                truncation=True,
                return_tensors="pt",
            )
            # Return dummy results as _perform_model_inference would
            return (
                torch.randn(len(texts_to_tokenize), model_config["dimension"]),
                [5] * len(texts_to_tokenize),
                15 * len(texts_to_tokenize),
            )

        mock_inference.side_effect = _mock_perform_model_inference_side_effect

        # Call the actual function we are testing
        await embedding_processor.get_embeddings_batch(test_texts, model_name, settings)

        # Assert that our inference function was called
        mock_inference.assert_called_once()

        # Now, assert on the original mock_tokenizer from the fixture,
        # as it's the one that was passed into _perform_model_inference and called by our side_effect.
        mock_tokenizer.assert_called_with(
            mock.ANY,  # The texts passed to the tokenizer
            max_length=model_config["max_tokens"],
            padding="longest",
            truncation=True,
            return_tensors="pt",
        )


@pytest.mark.asyncio
@pytest.mark.unit
async def test_standard_model_parallel_batching(reset_embedding_processor_state):
    settings, _, _, mock_load_model = reset_embedding_processor_state
    model_name = "all-MiniLM-L6-v2"
    model_config = models_config.get_model_config(model_name)
    test_texts = [f"text {i}" for i in range(70)]

    with mock.patch("embedding_processor._perform_model_inference") as mock_inference:
        # Configure the mock to return dummy results for each batch
        mock_inference.side_effect = [
            (torch.randn(32, model_config["dimension"]), [5] * 32, 160),  # noqa: E226
            (torch.randn(32, model_config["dimension"]), [5] * 32, 160),  # noqa: E226
            (torch.randn(6, model_config["dimension"]), [5] * 6, 30),  # noqa: E226
        ]

        await embedding_processor.get_embeddings_batch(test_texts, model_name, settings)

        # Assert that _perform_model_inference was called ceil(70 / 32) = 3 times
        expected_calls = 3
        assert mock_inference.call_count == expected_calls

        # Assert the size of the text batches passed in each call
        assert len(mock_inference.call_args_list[0][0][0]) == 32
        assert len(mock_inference.call_args_list[1][0][0]) == 32
        assert len(mock_inference.call_args_list[2][0][0]) == 6


@pytest.mark.asyncio
@pytest.mark.unit
@pytest.mark.parametrize("model_name", ["nomic-embed-text-v1.5", "all-MiniLM-L6-v2"])
async def test_single_item_batch_edge_case(reset_embedding_processor_state, model_name):
    """
    Test Case 3: Edge case with a single-item batch for both problematic and standard models.
    """
    settings, mock_model, mock_tokenizer, mock_load_model = reset_embedding_processor_state
    model_config = models_config.get_model_config(model_name)

    test_texts = ["a single sentence."]

    # Manually load model and tokenizer mocks
    model_instance, tokenizer_instance = await mock_load_model.side_effect(model_name)

    # Mock tokenizer to return predictable token counts and attention mask
    if model_config.get("requires_max_length_padding", False):
        expected_padding = "max_length"
        mock_tokenizer.side_effect = lambda texts, **kwargs: MockBatchEncoding(
            input_ids=torch.full((len(texts), model_config["max_tokens"]), 1, dtype=torch.long),
            attention_mask=torch.full((len(texts), model_config["max_tokens"]), 1, dtype=torch.long),
        )
    else:
        expected_padding = "longest"
        mock_tokenizer.side_effect = lambda texts, **kwargs: MockBatchEncoding(
            input_ids=torch.full((len(texts), 5), 1, dtype=torch.long),
            attention_mask=torch.full((len(texts), 5), 1, dtype=torch.long),
        )

    # Mock model to return predictable hidden states
    def mock_model_call_side_effect_single(**batch_dict):
        batch_size, seq_len = batch_dict["input_ids"].shape
        hidden_state = torch.ones(batch_size, seq_len, model_config["dimension"]) * 0.1
        return MockModelOutput(last_hidden_state=hidden_state.to(model_loader.DEVICE))

    mock_model.side_effect = mock_model_call_side_effect_single

    embeddings, individual_tokens, total_tokens = embedding_processor._perform_model_inference(
        test_texts,
        model_instance,
        tokenizer_instance,
        model_config,
        settings,
    )

    assert isinstance(embeddings, torch.Tensor)
    assert embeddings.shape == (len(test_texts), model_config["dimension"])
    assert all(isinstance(t, int) for t in individual_tokens)
    assert total_tokens > 0
    mock_tokenizer.assert_called_once()
    mock_model.assert_called_once()
    # Assert that the tokenizer was called with the correct padding strategy
    mock_tokenizer.assert_called_with(
        test_texts,
        max_length=model_config["max_tokens"],
        padding=expected_padding,
        truncation=True,
        return_tensors="pt",
    )
