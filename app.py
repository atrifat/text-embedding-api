import logging
import os
import time
from contextlib import asynccontextmanager
from datetime import datetime, timezone

import torch
import uvicorn
from cachetools import LRUCache
from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

import embedding_processor
from config import AppSettings, get_app_settings
from embedding_processor import get_embeddings_batch
from models_config import MODELS, get_model_config
from response_formatter import format_ollama_response, format_openai_response
from schemas import (
    EmbeddingRequest,
    EmbeddingResponse,
    ListModelsResponse,
    ModelObject,
    OllamaEmbeddingResponse,
    OllamaModelDetails,
    OllamaModelObject,
    OllamaTagsResponse,
)

# Suppress a common warning from Hugging Face Tokenizers when processes are forked,
# which can occur in web servers like FastAPI. This prevents potential deadlocks or runtime warnings.
os.environ["TOKENIZERS_PARALLELISM"] = "false"

logging.basicConfig(
    level=(logging.DEBUG if os.environ.get("ENVIRONMENT") != "production" else logging.INFO),
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Handles application startup and shutdown events.
    Initializes the embeddings cache and warms up the default model.
    """
    settings = get_app_settings()
    embedding_processor.embeddings_cache = LRUCache(maxsize=settings.embeddings_cache_maxsize)
    logger.info(f"Embeddings cache initialized with max size: {settings.embeddings_cache_maxsize}")

    default_model = settings.default_model
    if default_model not in MODELS:
        logger.error(f"Default model '{default_model}' is not configured in MODELS.")
        raise ValueError(f"Default model '{default_model}' is not configured in MODELS.")
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


@app.get("/api/tags", response_model=OllamaTagsResponse)
async def list_ollama_models():
    """
    Lists the available embedding models in an Ollama-compatible /api/tags format.
    Note: Some fields like 'size', 'digest', and 'details' are placeholders
    as they are not directly available from Hugging Face model metadata.
    """
    ollama_models_list = []
    current_time_iso = datetime.now(timezone.utc).isoformat(timespec="seconds") + "Z"  # For modified_at

    # Iterate through all models (canonical and aliases) from the combined MODELS dictionary
    # This ensures all supported models are listed without complex de-duplication logic.
    for model_name_or_alias in MODELS.keys():
        model_config = get_model_config(model_name_or_alias)  # Use get_model_config to resolve aliases

        family = model_config["name"].split("/")[0] if "/" in model_config["name"] else "unknown"
        parameter_size = f"{model_config['dimension']}D"
        quantization_level = "Q8_0"  # Placeholder

        ollama_models_list.append(
            OllamaModelObject(
                name=model_name_or_alias,  # Use the name/alias as requested by Ollama format
                modified_at=current_time_iso,
                size=100000000,  # Placeholder size (e.g., 100MB)
                digest="placeholder_digest",
                details=OllamaModelDetails(
                    format="gguf",  # Placeholder format
                    family=family,
                    families=[family],
                    parameter_size=parameter_size,
                    quantization_level=quantization_level,
                ),
            )
        )
    return OllamaTagsResponse(models=ollama_models_list)


@app.post("/api/embed", response_model=OllamaEmbeddingResponse)
@app.post("/v1/embeddings", response_model=EmbeddingResponse)
async def create_embeddings(
    embedding_request: EmbeddingRequest,
    request: Request,  # Inject the FastAPI Request object
    settings: AppSettings = Depends(get_app_settings),
):
    """
    Generates embeddings for the given input text(s) using batch processing.
    Compatible with OpenAI's Embeddings API format for /v1/embeddings and
    Ollama's API format for /api/embed.
    The input can be a single string or a list of strings.
    Returns a list of embedding objects, each containing the embedding vector.
    """
    try:
        start_time = time.time()

        if isinstance(embedding_request.input, str):
            texts = [embedding_request.input]
        else:
            texts = embedding_request.input

        if not texts:
            # Handle empty input for both response types
            if request.url.path == "/api/embed":
                return format_ollama_response(embeddings_tensor=torch.empty(0, 0), total_tokens=0)
            else:  # Default to OpenAI format for /v1/embeddings
                return format_openai_response(
                    embeddings_tensor=torch.empty(0, 0),
                    total_tokens=0,
                    texts=[],
                    model_name=embedding_request.model,
                )

        embeddings_tensor, total_tokens = await get_embeddings_batch(texts, embedding_request.model, settings)

        if request.url.path == "/api/embed":
            return format_ollama_response(embeddings_tensor=embeddings_tensor, total_tokens=total_tokens)
        else:
            response = format_openai_response(
                embeddings_tensor=embeddings_tensor,
                total_tokens=total_tokens,
                texts=texts,
                model_name=embedding_request.model,
            )
            end_time = time.time()
            processing_time = end_time - start_time

            if settings.environment != "production":
                logger.debug(
                    f"Processed {len(texts)} inputs in {processing_time:.4f} seconds. "
                    f"Model: {embedding_request.model}. Tokens: {total_tokens}."
                )
            return response

    except ValueError as e:
        logger.error(f"Validation error in embeddings endpoint: {e}", exc_info=True)
        raise HTTPException(status_code=422, detail=str(e))
    except HTTPException as e:
        logger.error(f"HTTPException in embeddings endpoint: {e.detail}", exc_info=True)
        raise e
    except Exception as e:
        logger.error(f"Unhandled error in embeddings endpoint: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    logger.error(f"Validation error for request to {request.url}: {exc.errors()}")
    raise HTTPException(status_code=422, detail=str(exc.errors()))


if __name__ == "__main__":
    uvicorn.run(app, host=get_app_settings().app_host, port=get_app_settings().app_port)
