# embedding_processor.py
import hashlib
import logging
from typing import List, Tuple, Union

import torch
import torch.nn.functional as F
from cachetools import LRUCache
from fastapi import HTTPException
from starlette import status

from config import AppSettings
from model_loader import DEVICE, load_model
from models_config import get_model_config

logger = logging.getLogger(__name__)

# Initialize global embeddings cache (size determined by settings)
embeddings_cache: Union[LRUCache, None] = None  # Will be initialized on startup based on settings


def _process_texts_for_cache_and_batching(
    texts: List[str], model_config: dict, settings: AppSettings
) -> Tuple[List[torch.Tensor], int, List[str], List[int]]:
    """
    Checks cache for each text and prepares texts for model processing.
    Returns cached embeddings, total cached tokens, texts to process, and their original indices.
    """
    final_ordered_embeddings = [None] * len(texts)
    total_prompt_tokens = 0
    texts_to_process_in_model = []
    original_indices_for_model_output = []

    canonical_hf_model_name = model_config["name"]

    for i, text in enumerate(texts):
        text_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
        cache_key = (text_hash, canonical_hf_model_name)

        if settings.embeddings_cache_enabled and cache_key in embeddings_cache:
            cached_embedding, cached_tokens = embeddings_cache[cache_key]
            final_ordered_embeddings[i] = cached_embedding.unsqueeze(0)
            if settings.report_cached_tokens:
                total_prompt_tokens += cached_tokens
            logger.debug(f"Cache hit for text at index {i}")
        else:
            texts_to_process_in_model.append(text)
            original_indices_for_model_output.append(i)
            logger.debug(f"Cache miss for text at index {i}")
    return final_ordered_embeddings, total_prompt_tokens, texts_to_process_in_model, original_indices_for_model_output


def _apply_instruction_prefix(texts: List[str], model_config: dict) -> List[str]:
    """
    Applies instruction prefixes to texts if required by the model configuration.
    """
    if model_config.get("instruction_prefix_required", False):
        processed_texts = []
        default_prefix = model_config.get("default_instruction_prefix", "")
        known_prefixes = model_config.get("known_instruction_prefixes", [])
        for text in texts:
            if not any(text.startswith(prefix) for prefix in known_prefixes):
                processed_texts.append(f"{default_prefix}{text}")
            else:
                processed_texts.append(text)
        return processed_texts
    return texts


def _perform_model_inference(
    texts_to_tokenize: List[str], model, tokenizer, model_config: dict, settings: AppSettings  # Changed parameters
) -> Tuple[torch.Tensor, List[int], int]:
    """
    Performs model inference for a batch of texts and returns embeddings,
    individual token counts, and total prompt tokens for the batch.
    Handles CUDA Out of Memory errors.
    """
    try:
        model_max_tokens = model_config.get("max_tokens", 8192)
        model_dimension = model_config["dimension"]
        pooling_strategy = model_config.get("pooling_strategy", "cls")

        # Determine the correct padding strategy based on the model's needs.
        if model_config.get("requires_max_length_padding", False):
            padding_strategy = "max_length"
        else:
            padding_strategy = "longest"

        batch_dict = tokenizer(
            texts_to_tokenize,
            max_length=model_max_tokens,
            padding=padding_strategy,
            truncation=True,
            return_tensors="pt",
        )

        individual_tokens_in_batch = [int(torch.sum(mask).item()) for mask in batch_dict.attention_mask]

        prompt_tokens_current_batch = int(torch.sum(batch_dict.attention_mask).item())

        batch_dict = {k: v.to(DEVICE) for k, v in batch_dict.items()}

        with torch.no_grad():
            outputs = model(**batch_dict)

        if pooling_strategy == "mean":
            last_hidden = outputs.last_hidden_state
            attention_mask = batch_dict["attention_mask"]
            mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden.size()).float()
            sum_embeddings = torch.sum(last_hidden * mask_expanded, 1)
            sum_mask = torch.clamp(mask_expanded.sum(1), min=1e-9)
            embeddings = sum_embeddings / sum_mask
        else:  # "cls" pooling
            embeddings = outputs.last_hidden_state[:, 0]

        # All models benefit from normalization
        embeddings = F.normalize(embeddings, p=2, dim=1)

        # Truncate to model's specified dimension (for models like nomic-embed-text)
        embeddings = embeddings[:, :model_dimension]

        return embeddings, individual_tokens_in_batch, prompt_tokens_current_batch
    except torch.cuda.OutOfMemoryError as e:
        logger.error(
            f"CUDA Out of Memory Error during embedding generation: {e}. "
            "Consider reducing EMBEDDING_BATCH_SIZE or using a smaller model.",
            exc_info=True,
        )
        raise HTTPException(
            status_code=status.HTTP_507_INSUFFICIENT_STORAGE,
            detail=f"GPU out of memory: {e}. Please try with a smaller batch size or a different model.",
        )
    except Exception as e:
        logger.error(f"An unexpected error occurred during batch embedding generation: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal server error during embedding generation: {str(e)}",
        )
    finally:
        if settings.cuda_cache_clear_enabled and torch.cuda.is_available():
            torch.cuda.empty_cache()
            logger.debug("CUDA cache cleared after processing chunk.")


def _store_embeddings_in_cache(
    embeddings: torch.Tensor,
    individual_tokens_in_batch: List[int],
    batch_original_indices: List[int],
    texts: List[str],
    model_config: dict,
    final_ordered_embeddings: List[Union[torch.Tensor, None]],
    settings: AppSettings,
):
    """
    Stores newly generated embeddings in the cache and updates the final ordered embeddings list.
    """
    canonical_hf_model_name = model_config["name"]
    for j, original_idx in enumerate(batch_original_indices):
        current_text = texts[original_idx]
        current_embedding = embeddings[j].cpu()
        current_tokens = individual_tokens_in_batch[j]

        current_text_hash = hashlib.sha256(current_text.encode("utf-8")).hexdigest()
        if settings.embeddings_cache_enabled:
            embeddings_cache[(current_text_hash, canonical_hf_model_name)] = (current_embedding, current_tokens)
        final_ordered_embeddings[original_idx] = current_embedding.unsqueeze(0)


async def get_embeddings_batch(texts: List[str], model_name: str, settings: AppSettings) -> Tuple[torch.Tensor, int]:
    """
    Generates embeddings for a batch of texts using the specified model.
    Handles potential CUDA out of memory errors by processing texts in chunks.
    Includes an in-memory cache for individual text-model pairs.

    Args:
        texts (List[str]): The list of input texts to embed.
        model_name (str): The name of the model to use.
        settings (AppSettings): Application settings.
    """
    model_config = get_model_config(model_name)
    model, tokenizer = await load_model(model_name)

    model_dimension = model_config["dimension"]

    # Use the model-specific batch size from the config, falling back to the global setting.
    max_batch_size = model_config.get("max_batch_size", settings.embedding_batch_size)

    final_ordered_embeddings, total_prompt_tokens, texts_to_process_in_model, original_indices_for_model_output = (
        _process_texts_for_cache_and_batching(texts, model_config, settings)
    )

    if texts_to_process_in_model:
        processing_strategy = model_config.get("processing_strategy", "parallel")

        if processing_strategy == "sequential":
            # Process texts one by one for models requiring sequential processing (e.g., nomic-embed-text)
            for i, text_to_process in enumerate(texts_to_process_in_model):
                original_idx = original_indices_for_model_output[i]

                texts_to_tokenize = _apply_instruction_prefix([text_to_process], model_config)

                embeddings, individual_tokens_in_batch, prompt_tokens_current_batch = _perform_model_inference(
                    texts_to_tokenize, model, tokenizer, model_config, settings
                )

                total_prompt_tokens += prompt_tokens_current_batch

                _store_embeddings_in_cache(
                    embeddings,
                    individual_tokens_in_batch,
                    [original_idx],  # Pass single original index
                    texts,
                    model_config,
                    final_ordered_embeddings,
                    settings,
                )
        else:  # Default to parallel batching
            for i in range(0, len(texts_to_process_in_model), max_batch_size):
                batch_texts = texts_to_process_in_model[i : i + max_batch_size]
                batch_original_indices = original_indices_for_model_output[i : i + max_batch_size]

                texts_to_tokenize = _apply_instruction_prefix(batch_texts, model_config)

                embeddings, individual_tokens_in_batch, prompt_tokens_current_batch = _perform_model_inference(
                    texts_to_tokenize, model, tokenizer, model_config, settings
                )

                total_prompt_tokens += prompt_tokens_current_batch

                _store_embeddings_in_cache(
                    embeddings,
                    individual_tokens_in_batch,
                    batch_original_indices,
                    texts,
                    model_config,
                    final_ordered_embeddings,
                    settings,
                )

    # If no texts were processed by the model and no cached embeddings were found,
    # return an empty tensor with the correct dimension.
    if not any(e is not None for e in final_ordered_embeddings):
        return torch.empty(0, model_dimension), 0

    final_embeddings_tensor = torch.cat([e for e in final_ordered_embeddings if e is not None], dim=0)
    return final_embeddings_tensor, total_prompt_tokens
