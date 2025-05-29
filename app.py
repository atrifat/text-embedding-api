import os
import time
import logging
from typing import List, Union, Tuple
from cachetools import LRUCache
import hashlib
import asyncio
from functools import lru_cache
from contextlib import asynccontextmanager

os.environ["TOKENIZERS_PARALLELISM"] = (
    "false"  # Suppress a common warning from Hugging Face Tokenizers when processes are forked, which can occur in web servers like FastAPI. This prevents potential deadlocks or runtime warnings.
)

logging.basicConfig(
    level=(
        logging.DEBUG if os.environ.get("ENVIRONMENT") != "production" else logging.INFO
    ),
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

from fastapi import FastAPI, HTTPException, Request, Depends
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.exceptions import RequestValidationError
from pydantic import BaseModel, Field, field_validator, ConfigDict  # Import ConfigDict
from pydantic_settings import BaseSettings
from transformers import AutoModel, AutoTokenizer
import torch
import torch.nn.functional as F
import uvicorn
from starlette import status


from models_config import MODELS, get_model_config, CANONICAL_MODELS, MODEL_ALIASES


class AppSettings(BaseSettings):
    """
    Application settings loaded from environment variables.
    """

    cuda_cache_clear_enabled: bool = Field(
        True,
        json_schema_extra={"env": "CUDA_CACHE_CLEAR_ENABLED"},
        description="Enable CUDA cache clearing after each batch.",
    )
    default_model: str = Field(
        "text-embedding-3-large",
        json_schema_extra={"env": "DEFAULT_MODEL"},
        description="Default embedding model to use.",
    )
    warmup_enabled: bool = Field(
        True,
        json_schema_extra={"env": "WARMUP_ENABLED"},
        description="Enable model warmup on startup.",
    )
    app_port: int = Field(
        7860,
        json_schema_extra={"env": "APP_PORT"},
        description="Port for the FastAPI application.",
    )
    app_host: str = Field(
        "0.0.0.0",
        json_schema_extra={"env": "APP_HOST"},
        description="Host for the FastAPI application.",
    )
    embedding_batch_size: int = Field(
        8,
        json_schema_extra={"env": "EMBEDDING_BATCH_SIZE"},
        description="Batch size for embedding generation.",
    )
    embeddings_cache_enabled: bool = Field(
        True,
        json_schema_extra={"env": "EMBEDDINGS_CACHE_ENABLED"},
        description="Enable in-memory embeddings cache.",
    )
    report_cached_tokens: bool = Field(
        False,
        json_schema_extra={"env": "REPORT_CACHED_TOKENS"},
        description="Report token count for cached embeddings.",
    )
    embeddings_cache_maxsize: int = Field(
        2048,
        json_schema_extra={"env": "EMBEDDINGS_CACHE_MAXSIZE"},
        description="Maximum size of the embeddings cache.",
    )
    environment: str = Field(
        "development",
        json_schema_extra={"env": "ENVIRONMENT"},
        description="Application environment (e.g., 'production', 'development').",
    )
    allowed_origins: List[str] = Field(
        ["*"],
        json_schema_extra={"env": "ALLOWED_ORIGINS"},
        description="List of allowed origins for CORS. Use comma-separated values in .env (e.g., 'http://localhost:3000,https://example.com').",
    )

    model_config = ConfigDict(env_file=".env")  # Use ConfigDict instead of class Config


@lru_cache()  # Cache the settings instance for performance
def get_app_settings():
    return AppSettings()


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {DEVICE}")

# Initialize global embeddings cache (will be set in lifespan)
embeddings_cache: LRUCache | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Handles application startup and shutdown events.
    Initializes the embeddings cache and warms up the default model.
    """
    settings = get_app_settings()
    global embeddings_cache
    embeddings_cache = LRUCache(maxsize=settings.embeddings_cache_maxsize)
    logger.info(
        f"Embeddings cache initialized with max size: {settings.embeddings_cache_maxsize}"
    )

    default_model = settings.default_model
    if default_model not in MODELS:
        logger.error(f"Default model '{default_model}' is not configured in MODELS.")
        raise ValueError(
            f"Default model '{default_model}' is not configured in MODELS."
        )
    if settings.warmup_enabled:
        logger.info(f"Warming up default model: {default_model}...")
        try:
            await get_embeddings_batch(["warmup"], default_model, settings)
            logger.info("Model warmup complete.")
        except Exception as e:
            logger.error(f"Model warmup failed for {default_model}: {e}", exc_info=True)

    yield

    logger.info("Application shutdown.")


app = FastAPI(
    title="Embedding API",
    description="API for generating embeddings using a transformer model.",
    version="0.1.0",
    lifespan=lifespan,
)

# Add CORS middleware
settings = get_app_settings()
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allowed_origins,  # Use configurable origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory="static"), name="static")

# Initialize model cache
# to avoid reloading models on every request
model_cache = {}
tokenizer_cache = {}

# Global dictionary for model loading locks
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
            logger.info(f"Loading model: {canonical_hf_model_name}")
            model_path = config["name"]
            trust_remote = config.get("requires_remote_code", False)
            # `trust_remote_code=True` is required for some models (e.g., 'gte-multilingual-base', 'nomic-embed-text-v1.5')
            # because they use custom code (e.g., custom AutoModel implementations) that is not part of the
            # standard Hugging Face Transformers library. This flag allows the library to load and execute
            # that custom code from the model's repository.

            model_cache[canonical_hf_model_name] = AutoModel.from_pretrained(
                model_path, trust_remote_code=trust_remote
            ).to(DEVICE)
            model_cache[canonical_hf_model_name].eval()

            tokenizer_cache[canonical_hf_model_name] = AutoTokenizer.from_pretrained(
                model_path
            )
            logger.info(f"Model loaded: {canonical_hf_model_name}")
        return (
            model_cache[canonical_hf_model_name],
            tokenizer_cache[canonical_hf_model_name],
        )


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
        description="The name of the model to use for embedding. Supports both original model names and OpenAI-compatible names.",
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

        # Access embeddings_cache only if it's not None
        if (
            settings.embeddings_cache_enabled
            and embeddings_cache is not None
            and cache_key in embeddings_cache
        ):
            cached_embedding, cached_tokens = embeddings_cache[cache_key]
            final_ordered_embeddings[i] = cached_embedding.unsqueeze(0)
            if settings.report_cached_tokens:
                total_prompt_tokens += cached_tokens
            logger.debug(f"Cache hit for text at index {i}")
        else:
            texts_to_process_in_model.append(text)
            original_indices_for_model_output.append(i)
            logger.debug(f"Cache miss for text at index {i}")
    return (
        final_ordered_embeddings,
        total_prompt_tokens,
        texts_to_process_in_model,
        original_indices_for_model_output,
    )


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
    texts_to_tokenize: List[str],
    model,
    tokenizer,
    model_max_tokens: int,
    model_dimension: int,
    settings: AppSettings,
) -> Tuple[torch.Tensor, List[int], int]:
    """
    Performs model inference for a batch of texts and returns embeddings,
    individual token counts, and total prompt tokens for the batch.
    Handles CUDA Out of Memory errors.
    """
    try:
        batch_dict = tokenizer(
            texts_to_tokenize,
            max_length=model_max_tokens,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )

        individual_tokens_in_batch = [
            int(torch.sum(mask).item()) for mask in batch_dict["attention_mask"]
        ]

        prompt_tokens_current_batch = int(
            torch.sum(batch_dict["attention_mask"]).item()
        )

        batch_dict = {k: v.to(DEVICE) for k, v in batch_dict.items()}

        with torch.no_grad():
            outputs = model(**batch_dict)

        embeddings = outputs.last_hidden_state[:, 0]
        embeddings = embeddings[:, :model_dimension]
        embeddings = F.normalize(embeddings, p=2, dim=1)

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
        logger.error(
            f"An unexpected error occurred during batch embedding generation: {e}",
            exc_info=True,
        )
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
        if settings.embeddings_cache_enabled and embeddings_cache is not None:
            embeddings_cache[(current_text_hash, canonical_hf_model_name)] = (
                current_embedding,
                current_tokens,
            )
        final_ordered_embeddings[original_idx] = current_embedding.unsqueeze(0)


async def get_embeddings_batch(
    texts: List[str], model_name: str, settings: AppSettings = Depends(get_app_settings)
) -> Tuple[torch.Tensor, int]:
    """
    Generates embeddings for a batch of texts using the specified model.
    Handles potential CUDA out of memory errors by processing texts in chunks.
    Includes an in-memory cache for individual text-model pairs.

    Args:
        texts (List[str]): The list of input texts to embed.
        model_name (str): The name of the model to use.
        settings (AppSettings): Application settings injected via FastAPI's Depends.
    """
    config = get_model_config(model_name)
    model, tokenizer = await load_model(model_name)

    model_max_tokens = config.get("max_tokens", 8192)
    model_dimension = config["dimension"]
    max_batch_size = settings.embedding_batch_size

    (
        final_ordered_embeddings,
        total_prompt_tokens,
        texts_to_process_in_model,
        original_indices_for_model_output,
    ) = _process_texts_for_cache_and_batching(texts, config, settings)

    if texts_to_process_in_model:
        for i in range(0, len(texts_to_process_in_model), max_batch_size):
            batch_texts = texts_to_process_in_model[i : i + max_batch_size]
            batch_original_indices = original_indices_for_model_output[
                i : i + max_batch_size
            ]

            texts_to_tokenize = _apply_instruction_prefix(batch_texts, config)

            embeddings, individual_tokens_in_batch, prompt_tokens_current_batch = (
                _perform_model_inference(
                    texts_to_tokenize,
                    model,
                    tokenizer,
                    model_max_tokens,
                    model_dimension,
                    settings,
                )
            )

            total_prompt_tokens += prompt_tokens_current_batch

            _store_embeddings_in_cache(
                embeddings,
                individual_tokens_in_batch,
                batch_original_indices,
                texts,
                config,
                final_ordered_embeddings,
                settings,
            )

    final_embeddings_tensor = torch.cat(
        [e for e in final_ordered_embeddings if e is not None], dim=0
    )
    return final_embeddings_tensor, total_prompt_tokens


@app.get("/", response_class=FileResponse)
async def read_root():
    """
    Serve the static index.html file at the root route.
    """
    return FileResponse("static/index.html")


@app.get("/v1/models", response_model=ListModelsResponse)
async def list_models():
    """
    Lists the available embedding models.
    Returns:
        ListModelsResponse: The response containing a list of model objects.
    """
    model_list = []
    current_time = int(time.time())
    for model_name in MODELS.keys():
        model_list.append(
            ModelObject(
                id=model_name,
                created=current_time,
                owned_by="local",
            )
        )

    return ListModelsResponse(data=model_list)


@app.get("/v1/models/{model_id}", response_model=ModelObject)
async def get_model(model_id: str):
    """
    Retrieves information about a specific embedding model.
    Args:
        model_id (str): The ID of the model to retrieve.
    """
    if model_id in MODELS:
        current_time = int(time.time())
        return ModelObject(
            id=model_id,
            created=current_time,
            owned_by="local",
        )
    else:
        raise HTTPException(status_code=404, detail="Model not found")


@app.post(
    "/api/embed", response_model=EmbeddingResponse
)  # Route for compatibility with Ollama's API
@app.post(
    "/v1/embeddings", response_model=EmbeddingResponse
)  # Route for compatibility with OpenAI's API
async def create_embeddings(
    request: EmbeddingRequest, settings: AppSettings = Depends(get_app_settings)
):
    """
    Generates embeddings for the given input text(s) using batch processing.
    Compatible with OpenAI's Embeddings API format.
    The input can be a single string or a list of strings.
    Returns a list of embedding objects, each containing the embedding vector.
    """
    try:
        start_time = time.time()

        if isinstance(request.input, str):
            texts = [request.input]
        else:
            texts = request.input

        if not texts:
            return EmbeddingResponse(
                data=[],
                model=request.model,
                object="list",
                usage={"prompt_tokens": 0, "total_tokens": 0},
            )

        embeddings_tensor, total_tokens = await get_embeddings_batch(
            texts, request.model, settings
        )

        data = [
            EmbeddingObject(embedding=embeddings_tensor[i].tolist(), index=i)
            for i in range(len(texts))
        ]

        usage = {
            "prompt_tokens": total_tokens,
            "total_tokens": total_tokens,
        }

        end_time = time.time()
        processing_time = end_time - start_time

        if settings.environment != "production":
            logger.debug(
                f"Processed {len(texts)} inputs in {processing_time:.4f} seconds. "
                f"Model: {request.model}. Tokens: {total_tokens}."
            )

        return EmbeddingResponse(
            data=data, model=request.model, object="list", usage=usage
        )

    except ValueError as e:
        logger.error(f"Validation error in /v1/embeddings: {e}", exc_info=True)
        raise HTTPException(status_code=422, detail=str(e))
    except HTTPException as e:
        logger.error(f"HTTPException in /v1/embeddings: {e.detail}", exc_info=True)
        raise e
    except Exception as e:
        logger.error(f"Unhandled error in /v1/embeddings: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    logger.error(f"Validation error for request to {request.url}: {exc.errors()}")
    raise HTTPException(status_code=422, detail=str(exc.errors()))


if __name__ == "__main__":
    uvicorn.run(app, host=get_app_settings().app_host, port=get_app_settings().app_port)
