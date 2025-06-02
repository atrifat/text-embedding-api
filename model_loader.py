# model_loader.py
import asyncio
import logging

import torch
from transformers import AutoModel, AutoTokenizer

from models_config import get_model_config

logger = logging.getLogger(__name__)

# Set up device configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Initialize model cache to avoid reloading models on every request
model_cache = {}
tokenizer_cache = {}

# Initialize global dictionary for model loading locks
model_load_locks = {}


async def load_model(model_name: str):
    """
    Load model and tokenizer if not already loaded, with asynchronous locking.

    Args:
        model_name (str): The name of the model to load.

    Returns:
        tuple: A tuple containing the loaded model and tokenizer.
    """
    config = get_model_config(model_name)
    canonical_hf_model_name = config["name"]

    async with model_load_locks.setdefault(canonical_hf_model_name, asyncio.Lock()):
        if canonical_hf_model_name not in model_cache:
            logger.info(f"Using device: {DEVICE}")
            logger.info(f"Loading model: {canonical_hf_model_name}")
            model_path = config["name"]
            trust_remote = config.get("requires_remote_code", False)
            # `trust_remote_code=True` is required for some models
            # because they use custom code (e.g., custom AutoModel implementations) that is not part of the
            # standard Hugging Face Transformers library. This flag allows the library to load and execute
            # that custom code from the model's repository.

            model_cache[canonical_hf_model_name] = AutoModel.from_pretrained(
                model_path, trust_remote_code=trust_remote
            ).to(DEVICE)
            model_cache[canonical_hf_model_name].eval()

            tokenizer_cache[canonical_hf_model_name] = AutoTokenizer.from_pretrained(model_path)
            logger.info(f"Model loaded: {canonical_hf_model_name}")
        return model_cache[canonical_hf_model_name], tokenizer_cache[canonical_hf_model_name]
