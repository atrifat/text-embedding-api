# models_config.py

CANONICAL_MODELS = {
    "all-MiniLM-L6-v2": {
        "name": "sentence-transformers/all-MiniLM-L6-v2",
        "dimension": 384,
        "pooling_strategy": "mean",
        "requires_remote_code": False,
        "max_tokens": 512,
    },
    "gte-multilingual-base": {
        "name": "Alibaba-NLP/gte-multilingual-base",
        "dimension": 768,
        "pooling_strategy": "cls",
        "requires_remote_code": True,
        "max_tokens": 8192,
    },
    "nomic-embed-text-v1.5": {
        "name": "nomic-ai/nomic-embed-text-v1.5",
        "dimension": 768,
        "pooling_strategy": "mean",
        "requires_remote_code": True,
        "max_tokens": 8192,
        "instruction_prefix_required": True,
        "default_instruction_prefix": "search_document:",
        "known_instruction_prefixes": [
            "search_document:",
            "search_query:",
            "clustering:",
            "classification:",
        ],
    },
    "all-mpnet-base-v2": {
        "name": "sentence-transformers/all-mpnet-base-v2",
        "dimension": 768,
        "pooling_strategy": "mean",
        "requires_remote_code": False,
        "max_tokens": 384,
    },
}

# Mapping of aliases to their canonical model names
MODEL_ALIASES = {
    "all-minilm": "all-MiniLM-L6-v2",
    "text-embedding-3-small": "all-MiniLM-L6-v2",
    "text-embedding-3-large": "gte-multilingual-base",
    "nomic-embed-text": "nomic-embed-text-v1.5",
}

# This global MODELS dictionary will be used for listing available models and validation.
# It combines canonical names and aliases for easy lookup.
MODELS = {
    **CANONICAL_MODELS,
    **{alias: CANONICAL_MODELS[canonical] for alias, canonical in MODEL_ALIASES.items()},
}


def get_model_config(requested_model_name: str) -> dict:
    """
    Resolves a requested model name (which might be an alias) to its canonical
    configuration. Raises ValueError if the model is not found.
    """
    canonical_name = MODEL_ALIASES.get(requested_model_name, requested_model_name)

    if canonical_name not in CANONICAL_MODELS:
        raise ValueError(f"Model '{requested_model_name}' (canonical: '{canonical_name}') is not a recognized model.")

    return CANONICAL_MODELS[canonical_name]
