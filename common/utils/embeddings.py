# app/common/utils/embeddings.py
"""
Module responsible for generating text embeddings.
Provides functions for token validation and embedding generation.
"""
from vertexai.language_models import TextEmbeddingModel
import tiktoken
from typing import List, Dict, Optional
import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from ...common.config import (
    EMBEDDING_MODEL,
    MAX_TOKENS_PER_TEXT,
    MAX_RETRY_ATTEMPTS,
    RETRY_DELAY_SECONDS,
    EMBEDDING_BATCH_SIZE
)

logger = logging.getLogger(__name__)

def count_tokens(text: str, encoding_name: str = "cl100k_base") -> int:
    """Count the number of tokens in a text

    Args:
        text: Text to count tokens for
        encoding_name: Name of the encoding to use

    Returns:
        Number of tokens in the text

    Raises:
        ValueError: If the encoding is not supported
    """
    try:
        encoding = tiktoken.get_encoding(encoding_name)
        return len(encoding.encode(text))
    except Exception as e:
        error_msg = f"Failed to count tokens: {str(e)}"
        logger.error(error_msg)
        raise ValueError(error_msg) from e

def validate_token_count_per_text(text_info_list: List[Dict[str, str]]) -> None:
    """Validates the number of tokens in each text

    Args:
        text_info_list: List of text information to validate. Each element should be
                        a dictionary with 'filename' and 'content' keys.

    Raises:
        ValueError: If any text exceeds the token limit
    """
    total_tokens = 0
    for text_info in text_info_list:
        filename = text_info['filename']
        text = text_info['content']

        try:
            num_tokens = count_tokens(text)
            logger.info(f"Number of tokens in {filename}: {num_tokens}")

            if num_tokens > MAX_TOKENS_PER_TEXT:
                error_msg = (
                    f"Text in {filename} exceeds token limit. "
                    f"Limit: {MAX_TOKENS_PER_TEXT}, "
                    f"Actual: {num_tokens}, "
                    f"Text beginning: '{text[:100]}...'"
                )
                logger.error(error_msg)
                raise ValueError(error_msg)

            total_tokens += num_tokens

        except Exception as e:
            error_msg = f"Token validation failed for {filename}: {str(e)}"
            logger.error(error_msg)
            raise ValueError(error_msg) from e

    logger.info(f"Total number of tokens in all texts: {total_tokens}")

def embed_single_text(text: str,
                        model: TextEmbeddingModel,
                        retry_attempts: int = MAX_RETRY_ATTEMPTS) -> List[float]:
    """Generate embedding for a single text with retry logic

    Args:
        text: Text to generate embedding for
        model: The embedding model to use
        retry_attempts: Number of retry attempts for failed requests

    Returns:
        List of embedding values

    Raises:
        Exception: If embedding generation fails after all retries
    """
    for attempt in range(retry_attempts):
        try:
            embedding = model.get_embeddings([text])[0]
            return embedding.values

        except Exception as e:
            if attempt == retry_attempts - 1:  # Last attempt
                error_msg = (
                    f"Failed to generate embedding after {retry_attempts} attempts. "
                    f"Text beginning: '{text[:100]}...', Error: {str(e)}"
                )
                logger.error(error_msg)
                raise Exception(error_msg) from e

            logger.warning(
                f"Embedding generation attempt {attempt + 1} failed. "
                f"Retrying in {RETRY_DELAY_SECONDS} seconds..."
            )
            time.sleep(RETRY_DELAY_SECONDS)

def process_batch(texts: List[str],
                    model: TextEmbeddingModel,
                    start_idx: int) -> List[List[float]]:
    """Process a batch of texts to generate embeddings

    Args:
        texts: List of texts to generate embeddings for
        model: The embedding model to use
        start_idx: Starting index of the batch (for logging)

    Returns:
        List of embedding vectors

    Raises:
        Exception: If embedding generation fails for the batch
    """
    try:
        embeddings = model.get_embeddings(texts)
        logger.info(f"Successfully processed batch starting at index {start_idx}")
        return [embedding.values for embedding in embeddings]
    except Exception as e:
        error_msg = f"Failed to process batch starting at index {start_idx}: {str(e)}"
        logger.error(error_msg)
        raise Exception(error_msg) from e

def embed_texts(text_info_list: List[Dict[str, str]],
                model_name: str = EMBEDDING_MODEL) -> List[List[float]]:
    """Generate embeddings for multiple texts with batching and parallel processing

    Args:
        text_info_list: List of text information. Each element should be a dictionary
                        with 'filename' and 'content' keys.
        model_name: Name of the embedding model to use

    Returns:
        List of embedding vectors

    Raises:
        Exception: If embedding generation fails
    """
    try:
        # Validate token counts
        validate_token_count_per_text(text_info_list)

        # Initialize the model
        model = TextEmbeddingModel.from_pretrained(model_name)
        logger.info(f"Initialized embedding model: {model_name}")

        # Extract text content for processing
        texts = [info['content'] for info in text_info_list]
        total_texts = len(texts)

        # Prepare batches
        batches = [
            texts[i:i + EMBEDDING_BATCH_SIZE]
            for i in range(0, total_texts, EMBEDDING_BATCH_SIZE)
        ]

        all_embeddings = []
        start_time = time.time()

        # Process batches with parallel execution
        with ThreadPoolExecutor() as executor:
            future_to_batch = {
                executor.submit(process_batch, batch, model, i * EMBEDDING_BATCH_SIZE): i
                for i, batch in enumerate(batches)
            }

            for future in as_completed(future_to_batch):
                batch_idx = future_to_batch[future]
                try:
                    batch_embeddings = future.result()
                    all_embeddings.extend(batch_embeddings)
                    logger.info(
                        f"Completed batch {batch_idx + 1}/{len(batches)}, "
                        f"Total progress: {len(all_embeddings)}/{total_texts}"
                    )
                except Exception as e:
                    error_msg = f"Batch {batch_idx + 1} failed: {str(e)}"
                    logger.error(error_msg)
                    raise Exception(error_msg) from e

        # Verify results
        if len(all_embeddings) != total_texts:
            raise ValueError(
                f"Embedding count mismatch. Expected: {total_texts}, "
                f"Got: {len(all_embeddings)}"
            )

        total_time = time.time() - start_time
        logger.info(
            f"Successfully generated {len(all_embeddings)} embeddings "
            f"in {total_time:.2f} seconds"
        )
        return all_embeddings

    except Exception as e:
        error_msg = f"Embedding generation failed: {str(e)}"
        logger.error(error_msg)
        raise Exception(error_msg) from e
