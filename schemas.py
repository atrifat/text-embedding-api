from typing import List, Optional, Union

from pydantic import BaseModel, Field, field_validator

from models_config import CANONICAL_MODELS, MODEL_ALIASES, MODELS


class EmbeddingRequest(BaseModel):
    """
    Represents a request for generating embeddings.

    Attributes:
        input (Union[str, List[str]]): The input text to embed, can be a single string or a list of strings.
        model (str): The name of the model to use for embedding.
        encoding_format (str): The format of the embeddings. Currently only 'float' is supported.
    """

    input: Union[str, List[str]] = Field(
        ...,
        description="The input text to embed, can be a single string or a list of strings.",
        json_schema_extra={"example": "This is an example sentence."},
    )
    model: str = Field(
        "text-embedding-3-large",
        description=(
            "The name of the model to use for embedding. Supports both original model names "
            "and OpenAI-compatible names."
        ),
        json_schema_extra={"example": "text-embedding-3-large"},
    )
    encoding_format: str = Field(
        "float",
        description="The format of the embeddings. Currently only 'float' is supported.",
        json_schema_extra={"example": "float"},
    )

    @field_validator("model")
    @classmethod
    def validate_model(cls, value: str) -> str:
        if value not in MODELS:
            valid_models = list(CANONICAL_MODELS.keys()) + list(MODEL_ALIASES.keys())
            raise ValueError(f"Model must be one of: {', '.join(sorted(valid_models))}")
        return value

    @field_validator("encoding_format")
    @classmethod
    def validate_encoding_format(cls, value: str) -> str:
        if value != "float":
            raise ValueError("Only 'float' encoding format is supported")
        return value


class EmbeddingObject(BaseModel):
    """
    Represents an embedding object.

    Attributes:
        object (str): The type of object, which is "embedding".
        embedding (List[float]): The embedding vector.
        index (int): The index of the embedding.
    """

    object: str = "embedding"
    embedding: List[float]
    index: int


class EmbeddingResponse(BaseModel):
    """
    Represents the response containing a list of embedding objects.
    """

    data: List[EmbeddingObject]
    model: str
    object: str = "list"
    usage: dict


class OllamaEmbeddingResponse(BaseModel):
    """
    Represents the response containing a list of embedding vectors in Ollama's format,
    with an optional usage field.
    """

    embeddings: List[List[float]]
    usage: Optional[dict] = {}


class ModelObject(BaseModel):
    """
    Represents a single model object in the list of models.
    """

    id: str
    object: str = "model"
    created: int
    owned_by: str


class ListModelsResponse(BaseModel):
    """
    Represents the response containing a list of available models.
    """

    data: List[ModelObject]
    object: str = "list"


class OllamaModelDetails(BaseModel):
    format: str = "gguf"  # Placeholder, as HF models are not necessarily GGUF
    family: str
    families: List[str]
    parameter_size: str
    quantization_level: str


class OllamaModelObject(BaseModel):
    name: str
    modified_at: str
    size: int
    digest: str
    details: OllamaModelDetails


class OllamaTagsResponse(BaseModel):
    models: List[OllamaModelObject]
