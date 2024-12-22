# app/common/utils/embeddings.py
"""
Module responsible for generating text embeddings.
Provides comprehensive functionality for token validation and embedding generation
with robust error handling and parallel processing capabilities.
"""
from vertexai.language_models import TextEmbeddingModel
import tiktoken
from typing import List, Dict, Any, Optional, Union
import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from enum import Enum
from ...common.config import (
    EMBEDDING_MODEL,
    MAX_TOKENS_PER_TEXT,
    MAX_RETRY_ATTEMPTS,
    RETRY_DELAY_SECONDS,
    EMBEDDING_BATCH_SIZE
)

logger = logging.getLogger(__name__)

class EmbeddingError(Exception):
    """Base exception class for embedding operations"""
    pass

class TokenizationError(EmbeddingError):
    """Exception raised for tokenization-related errors"""
    pass

class EmbeddingGenerationError(EmbeddingError):
    """Exception raised when embedding generation fails"""
    pass

class BatchProcessingError(EmbeddingError):
    """Exception raised when batch processing fails"""
    pass

class EncodingType(Enum):
    """Supported encoding types for tokenization"""
    CL100K_BASE = "cl100k_base"
    P50K_BASE = "p50k_base"
    R50K_BASE = "r50k_base"

@dataclass
class TokenValidationResult:
    """Results of token validation"""
    is_valid: bool
    token_count: int
    error_message: Optional[str] = None

@dataclass
class EmbeddingConfig:
    """Configuration for embedding generation"""
    model_name: str = EMBEDDING_MODEL
    max_tokens: int = MAX_TOKENS_PER_TEXT
    batch_size: int = EMBEDDING_BATCH_SIZE
    retry_attempts: int = MAX_RETRY_ATTEMPTS
    retry_delay: int = RETRY_DELAY_SECONDS
    encoding_type: EncodingType = EncodingType.CL100K_BASE

class TextTokenizer:
    """Class to handle text tokenization operations"""

    def __init__(self, encoding_type: EncodingType = EncodingType.CL100K_BASE):
        """Initialize tokenizer with specified encoding

        Args:
            encoding_type: Type of encoding to use for tokenization

        Raises:
            TokenizationError: If encoding initialization fails
        """
        try:
            self.encoding = tiktoken.get_encoding(encoding_type.value)
            logger.debug(f"Initialized tokenizer with encoding: {encoding_type.value}")
        except Exception as e:
            error_msg = f"Failed to initialize encoding {encoding_type.value}: {str(e)}"
            logger.error(error_msg)
            raise TokenizationError(error_msg) from e

    def count_tokens(self, text: str) -> int:
        """Count tokens in a text string

        Args:
            text: Text to count tokens for

        Returns:
            Number of tokens in the text

        Raises:
            TokenizationError: If token counting fails
        """
        try:
            return len(self.encoding.encode(text))
        except Exception as e:
            error_msg = f"Failed to count tokens: {str(e)}"
            logger.error(error_msg)
            raise TokenizationError(error_msg) from e

    def validate_token_count(self,
                            text: str,
                            max_tokens: int) -> TokenValidationResult:
        """Validate token count against maximum limit

        Args:
            text: Text to validate
            max_tokens: Maximum allowed tokens

        Returns:
            TokenValidationResult object with validation results
        """
        try:
            token_count = self.count_tokens(text)
            is_valid = token_count <= max_tokens
            error_message = None if is_valid else (
                f"Token count {token_count} exceeds limit {max_tokens}"
            )

            return TokenValidationResult(
                is_valid=is_valid,
                token_count=token_count,
                error_message=error_message
            )
        except TokenizationError as e:
            return TokenValidationResult(
                is_valid=False,
                token_count=0,
                error_message=str(e)
            )

class EmbeddingGenerator:
    """Class to handle embedding generation operations"""

    def __init__(self, config: Optional[EmbeddingConfig] = None):
        """Initialize embedding generator

        Args:
            config: Configuration for embedding generation
        """
        self.config = config or EmbeddingConfig()
        self.tokenizer = TextTokenizer(self.config.encoding_type)
        self.model = TextEmbeddingModel.from_pretrained(self.config.model_name)
        logger.info(f"Initialized embedding generator with model: {self.config.model_name}")

    def _generate_single_embedding(self, text: str) -> List[float]:
        """Generate embedding for a single text with retry logic

        Args:
            text: Text to generate embedding for

        Returns:
            List of embedding values

        Raises:
            EmbeddingGenerationError: If embedding generation fails after all retries
        """
        for attempt in range(self.config.retry_attempts):
            try:
                embedding = self.model.get_embeddings([text])[0]
                return embedding.values

            except Exception as e:
                if attempt == self.config.retry_attempts - 1:
                    error_msg = (
                        f"Failed to generate embedding after {self.config.retry_attempts} attempts. "
                        f"Text beginning: '{text[:100]}...', Error: {str(e)}"
                    )
                    logger.error(error_msg)
                    raise EmbeddingGenerationError(error_msg) from e

                logger.warning(
                    f"Embedding generation attempt {attempt + 1} failed. "
                    f"Retrying in {self.config.retry_delay} seconds..."
                )
                time.sleep(self.config.retry_delay)

    def _process_batch(self,
                        texts: List[str],
                        start_idx: int) -> List[List[float]]:
        """Process a batch of texts to generate embeddings

        Args:
            texts: List of texts to process
            start_idx: Starting index of the batch (for logging)

        Returns:
            List of embedding vectors

        Raises:
            BatchProcessingError: If batch processing fails
        """
        try:
            embeddings = self.model.get_embeddings(texts)
            logger.debug(f"Successfully processed batch starting at index {start_idx}")
            return [embedding.values for embedding in embeddings]
        except Exception as e:
            error_msg = f"Failed to process batch starting at index {start_idx}: {str(e)}"
            logger.error(error_msg)
            raise BatchProcessingError(error_msg) from e

    def validate_and_prepare_texts(self,
                                    text_info_list: List[Dict[str, str]]) -> List[str]:
        """Validate and prepare texts for embedding generation

        Args:
            text_info_list: List of text information dictionaries

        Returns:
            List of validated texts

        Raises:
            TokenizationError: If any text fails validation
        """
        prepared_texts = []
        total_tokens = 0

        for text_info in text_info_list:
            filename = text_info['filename']
            text = text_info['content']

            validation_result = self.tokenizer.validate_token_count(
                text,
                self.config.max_tokens
            )

            if not validation_result.is_valid:
                raise TokenizationError(
                    f"Validation failed for {filename}: {validation_result.error_message}"
                )

            total_tokens += validation_result.token_count
            prepared_texts.append(text)

        logger.info(f"Total tokens in all texts: {total_tokens}")
        return prepared_texts

    def generate_embeddings(self,
                            text_info_list: List[Dict[str, str]]) -> List[List[float]]:
        """Generate embeddings for multiple texts with batching and parallel processing

        Args:
            text_info_list: List of text information dictionaries

        Returns:
            List of embedding vectors

        Raises:
            EmbeddingError: If embedding generation fails
        """
        try:
            # Validate and prepare texts
            texts = self.validate_and_prepare_texts(text_info_list)
            total_texts = len(texts)

            # Prepare batches
            batches = [
                texts[i:i + self.config.batch_size]
                for i in range(0, total_texts, self.config.batch_size)
            ]

            all_embeddings = []
            start_time = time.time()

            # Process batches with parallel execution
            with ThreadPoolExecutor() as executor:
                future_to_batch = {
                    executor.submit(self._process_batch, batch, i * self.config.batch_size): i
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
                    except BatchProcessingError as e:
                        error_msg = f"Batch {batch_idx + 1} failed: {str(e)}"
                        logger.error(error_msg)
                        raise EmbeddingError(error_msg) from e

            # Verify results
            if len(all_embeddings) != total_texts:
                raise EmbeddingError(
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
            raise EmbeddingError(error_msg) from e

def embed_texts(text_info_list: List[Dict[str, str]],
                config: Optional[EmbeddingConfig] = None) -> List[List[float]]:
    """Convenience function to generate embeddings with default configuration

    Args:
        text_info_list: List of text information dictionaries
        config: Optional custom configuration

    Returns:
        List of embedding vectors

    Raises:
        EmbeddingError: If embedding generation fails
    """
    generator = EmbeddingGenerator(config)
    return generator.generate_embeddings(text_info_list)
