# app/common/utils/embeddings.py
"""
Module responsible for generating text embeddings.
Provides functions for token validation and embedding generation.
"""
from vertexai.language_models import TextEmbeddingModel
import tiktoken
from typing import List, Dict
import logging
from ...common.config import (
    EMBEDDING_MODEL,
    MAX_TOKENS_PER_TEXT,
)

logger = logging.getLogger(__name__)

def validate_token_count_per_text(text_info_list: List[Dict[str, str]]) -> None:
    """Validates the number of tokens in the input text (per text unit)

    Args:
        text_info_list: List of text information to be validated. Each element is a dictionary of {'filename': 'filename', 'content': 'text content'}.

    Raises:
        ValueError: If the number of tokens exceeds the limit
    """
    encoding = tiktoken.get_encoding("cl100k_base")

    total_tokens = 0
    for text_info in text_info_list:
        filename = text_info['filename']
        text = text_info['content']
        num_tokens = len(encoding.encode(text))
        logger.info(f"Number of tokens in {filename}: {num_tokens}")
        if num_tokens > MAX_TOKENS_PER_TEXT:
            raise ValueError(
                f"The number of tokens in the text exceeds the limit. Limit: {MAX_TOKENS_PER_TEXT}, "
                f"Actual: {num_tokens}, File: {filename}, Text beginning: '{text[:50]}...'"
            )
        total_tokens += num_tokens

    logger.info(f"Total number of tokens in all texts: {total_tokens}")

def embed_single_text(text: str, model: TextEmbeddingModel) -> List[float]:
    """Converts a single text into an embedding

    Args:
        text: Text to be converted
        model: Embedding model to use

    Returns:
        List of embedding values
    """
    try:
        embedding = model.get_embeddings([text])[0]
        return embedding.values
    except Exception as e:
        logger.error(f"Embedding generation error - Text: '{text[:50]}...': {str(e)}")
        raise

def embed_texts(text_info_list: List[Dict[str, str]], model_name: str = EMBEDDING_MODEL) -> List[List[float]]:
    """Converts multiple texts into embeddings

    Args:
        text_info_list: List of text information to be converted. Each element is a dictionary of {'filename': 'filename', 'content': 'text content'}.
        model_name: Name of the embedding model to use

    Returns:
        List of lists of embedding values
    """
    try:
        # Validate the number of tokens (per text unit)
        validate_token_count_per_text(text_info_list)

        # Initialize the model
        model = TextEmbeddingModel.from_pretrained(model_name)

        # List to store the results
        result = []

        # Process each text individually
        for text_info in text_info_list:
            filename = text_info['filename']
            text = text_info['content']
            logger.info(f"Start generating embedding for {filename}: '{text[:50]}...'")
            embedding = model.get_embeddings([text])[0]  # Vectorize each text individually
            result.append(embedding.values)
            logger.info(f"Embedding generation completed for {filename}: {len(embedding.values)} dimensions")

        logger.info(f"Embedding generation completed for a total of {len(result)} texts")
        return result

    except ValueError as ve:
        logger.error(f"Token count validation error: {str(ve)}")
        raise
    except Exception as e:
        logger.error(f"Embedding generation process error: {str(e)}")
        raise

def get_embedding_dimension(model_name: str = EMBEDDING_MODEL) -> int:
    model = TextEmbeddingModel.from_pretrained(model_name)
    sample_embedding = model.get_embeddings(["test"])[0]
    return len(sample_embedding.values)
