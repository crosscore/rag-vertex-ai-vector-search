# app/rag/search.py
"""
Module for performing semantic search operations using Vector Search.
Provides functionality to search for relevant table metadata based on user queries.
"""
from typing import List, Dict, Any
import logging
from datetime import datetime
import os

from ..common.config import (
    PROJECT_ID,
    REGION,
    DEPLOYED_INDEX_ID,
    ENDPOINT_DISPLAY_NAME,
    FIRESTORE_COLLECTION
)
from ..common.utils.vector_search import VectorSearchClient, format_search_results
from ..common.utils.embeddings import embed_texts
from ..vector_store.utils.firestore_ops import FirestoreManager

logger = logging.getLogger(__name__)

class SemanticSearcher:
    """Class to perform semantic search operations"""

    def __init__(self):
        """Initialize the semantic searcher with necessary clients"""
        self.project_id = PROJECT_ID
        self.region = REGION
        self.deployed_index_id = DEPLOYED_INDEX_ID
        self.vector_client = VectorSearchClient(project_id=self.project_id, location=self.region)
        self.firestore_manager = FirestoreManager()
        self._initialize_clients()

    def _initialize_clients(self) -> None:
        """Initialize Vector Search and Firestore clients"""
        try:
            # Initialize Vector Search endpoint
            endpoint_name = f"projects/{self.project_id}/locations/{self.region}/indexEndpoints/{ENDPOINT_DISPLAY_NAME}"
            self.vector_client.initialize_endpoint(endpoint_name)
            logger.info("Vector Search client initialized successfully")

        except Exception as e:
            error_msg = f"Failed to initialize clients: {str(e)}"
            logger.error(error_msg)
            raise RuntimeError(error_msg) from e

    def search(self,
                questions: List[str],
                num_results: int = 10,
                min_similarity_score: float = 0.0) -> List[Dict[str, Any]]:
        """Perform semantic search for multiple questions

        Args:
            questions: List of question strings to search for
            num_results: Number of results to return per question
            min_similarity_score: Minimum similarity score threshold

        Returns:
            List of dictionaries containing search results for each question
        """
        try:
            # Convert questions to expected format for embedding
            questions_info = [{"filename": f"question_{i}", "content": q}
                            for i, q in enumerate(questions)]

            # Generate embeddings for questions
            logger.info(f"Generating embeddings for {len(questions)} questions")
            query_embeddings = embed_texts(questions_info)

            # Perform vector similarity search
            logger.info("Performing vector similarity search")
            search_results = self.vector_client.find_neighbors(
                deployed_index_id=self.deployed_index_id,
                queries=query_embeddings,
                num_neighbors=num_results
            )

            # Format search results
            formatted_results = format_search_results(search_results)

            # Fetch metadata from Firestore and prepare final results
            final_results = []
            for question, question_results in zip(questions, formatted_results):
                result_entries = []
                for result in question_results:
                    # Skip results below similarity threshold
                    similarity_score = 1.0 - (result['distance'] / 2.0)  # Convert distance to similarity
                    if similarity_score < min_similarity_score:
                        continue

                    # Fetch metadata from Firestore
                    metadata = self.firestore_manager.get_text_metadata(
                        FIRESTORE_COLLECTION,
                        result['id']
                    )

                    if metadata:
                        result_entry = {
                            'question': question,
                            'data_point_id': result['id'],
                            'similarity_score': similarity_score,
                            'metadata': metadata
                        }
                        result_entries.append(result_entry)

                final_results.append({
                    'question': question,
                    'results': result_entries
                })

            logger.info(f"Search completed. Found results for {len(final_results)} questions")
            return final_results

        except Exception as e:
            error_msg = f"Search operation failed: {str(e)}"
            logger.error(error_msg)
            raise RuntimeError(error_msg) from e

def setup_logging(log_dir: str = 'app/log') -> None:
    """Configure logging settings

    Args:
        log_dir: Directory to store log files
    """
    os.makedirs(log_dir, exist_ok=True)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_filename = os.path.join(log_dir, f'semantic_search_{timestamp}.log')

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_filename, mode='w', encoding='utf-8')
        ]
    )

def main() -> None:
    """Main execution function"""
    setup_logging()
    logger = logging.getLogger(__name__)

    try:
        # Test questions
        test_questions = [
            "Find tables containing customer purchase history",
            "Which tables have information about product inventory?",
            "Show me tables related to sales transactions"
        ]

        # Initialize searcher and perform search
        searcher = SemanticSearcher()
        results = searcher.search(
            questions=test_questions,
            num_results=10,
            min_similarity_score=0.5
        )

        # Log results
        for question_result in results:
            question = question_result['question']
            matches = question_result['results']

            logger.info(f"\nResults for question: {question}")
            for i, match in enumerate(matches, 1):
                logger.info(f"Match {i}:")
                logger.info(f"  Data Point ID: {match['data_point_id']}")
                logger.info(f"  Similarity Score: {match['similarity_score']:.4f}")
                if 'filename' in match['metadata']:
                    logger.info(f"  Filename: {match['metadata']['filename']}")
                if 'content' in match['metadata']:
                    content_preview = match['metadata']['content'][:100] + "..."
                    logger.info(f"  Content Preview: {content_preview}")

    except Exception as e:
        logger.error(f"Main execution failed: {str(e)}")
        raise

if __name__ == "__main__":
    main()
