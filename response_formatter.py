import logging
from typing import List

import torch

from schemas import EmbeddingObject, EmbeddingResponse, OllamaEmbeddingResponse

logger = logging.getLogger(__name__)


def format_openai_response(
    embeddings_tensor: torch.Tensor, total_tokens: int, texts: List[str], model_name: str
) -> EmbeddingResponse:
    """
    Constructs an OpenAI-compatible EmbeddingResponse.
    """
    data = [EmbeddingObject(embedding=embeddings_tensor[i].tolist(), index=i) for i in range(len(texts))]
    usage = {
        "prompt_tokens": total_tokens,
        "total_tokens": total_tokens,
    }
    return EmbeddingResponse(data=data, model=model_name, object="list", usage=usage)


def format_ollama_response(embeddings_tensor: torch.Tensor, total_tokens: int) -> OllamaEmbeddingResponse:
    """
    Constructs an Ollama-compatible OllamaEmbeddingResponse.
    """
    usage_data = {"promptTokens": total_tokens, "totalTokens": total_tokens}
    return OllamaEmbeddingResponse(embeddings=embeddings_tensor.tolist(), usage=usage_data)
