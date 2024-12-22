# app/rag/search.py
"""
Module for performing semantic search operations using Vector Search.
Integrates vector search, result formatting, and metadata management for comprehensive search functionality.
"""
from typing import List, Dict, Any
import logging
from datetime import datetime
import os
from dataclasses import dataclass

from ..common.config import (
    PROJECT_ID,
    REGION,
    DEPLOYED_INDEX_ID,
    ENDPOINT_DISPLAY_NAME,
    FIRESTORE_COLLECTION
)
from ..common.utils.vector_search import (
    VectorSearchClient,
    SearchConfiguration,
    VectorSearchError
)
from ..common.utils.embeddings import embed_texts
from ..common.utils.result_formatter import (
    ResultFormatter,
    ResultAnalyzer,
    ResultType,
    StatisticalSummary
)
from ..vector_store.utils.firestore_ops import FirestoreManager

logger = logging.getLogger(__name__)

@dataclass
class SearchParameters:
    """Configuration parameters for semantic search"""
    num_results: int = 10
    min_similarity_score: float = 0.0
    include_metadata: bool = True
    compute_statistics: bool = True

class SearchError(Exception):
    """Base exception class for search operations"""
    pass

class SemanticSearcher:
    """Class to perform semantic search operations"""

    def __init__(self):
        """Initialize the semantic searcher with necessary clients"""
        self.project_id = PROJECT_ID
        self.region = REGION
        self.deployed_index_id = DEPLOYED_INDEX_ID

        # Initialize clients
        self.vector_client = VectorSearchClient(project_id=self.project_id, location=self.region)
        self.firestore_manager = FirestoreManager()
        self.result_formatter = ResultFormatter(result_type=ResultType.SEARCH)
        self.result_analyzer = ResultAnalyzer()

        self._initialize_clients()

    def _initialize_clients(self) -> None:
        """Initialize Vector Search and Firestore clients

        Raises:
            SearchError: If client initialization fails
        """
        try:
            # Initialize Vector Search endpoint
            endpoint_name = (f"projects/{self.project_id}/locations/{self.region}/"
                           f"indexEndpoints/{ENDPOINT_DISPLAY_NAME}")
            self.vector_client.initialize_endpoint(endpoint_name)
            logger.info("Clients initialized successfully")

        except VectorSearchError as e:
            error_msg = f"Failed to initialize vector search client: {str(e)}"
            logger.error(error_msg)
            raise SearchError(error_msg) from e
        except Exception as e:
            error_msg = f"Unexpected error during client initialization: {str(e)}"
            logger.error(error_msg)
            raise SearchError(error_msg) from e

    def _generate_embeddings(self, questions: List[str]) -> List[List[float]]:
        """Generate embeddings for search questions

        Args:
            questions: List of questions to generate embeddings for

        Returns:
            List of embedding vectors

        Raises:
            SearchError: If embedding generation fails
        """
        try:
            questions_info = [
                {"filename": f"question_{i}", "content": q}
                for i, q in enumerate(questions)
            ]

            embeddings = embed_texts(questions_info)
            logger.info(f"Generated embeddings for {len(questions)} questions")
            return embeddings

        except Exception as e:
            error_msg = f"Failed to generate embeddings: {str(e)}"
            logger.error(error_msg)
            raise SearchError(error_msg) from e

    def _process_search_results(self,
                              raw_results: List[List[Any]],
                              questions: List[str],
                              min_similarity_score: float) -> List[Dict[str, Any]]:
        """Process raw search results and fetch metadata

        Args:
            raw_results: Raw search results from vector search
            questions: Original search questions
            min_similarity_score: Minimum similarity score threshold

        Returns:
            List of processed search results with metadata

        Raises:
            SearchError: If result processing fails
        """
        try:
            processed_results = []

            for question, question_results in zip(questions, raw_results):
                result_entries = []

                for match in question_results:
                    # Convert distance to similarity score
                    similarity_score = 1.0 - (match.distance / 2.0)
                    if similarity_score < min_similarity_score:
                        continue

                    # Fetch metadata from Firestore
                    metadata = self.firestore_manager.get_text_metadata(
                        FIRESTORE_COLLECTION,
                        match.id
                    )

                    if metadata:
                        result_entry = {
                            'data_point_id': match.id,
                            'similarity_score': similarity_score,
                            'metadata': metadata
                        }
                        result_entries.append(result_entry)

                processed_results.append({
                    'question': question,
                    'results': result_entries
                })

            logger.info(f"Processed {len(processed_results)} search results")
            return processed_results

        except Exception as e:
            error_msg = f"Failed to process search results: {str(e)}"
            logger.error(error_msg)
            raise SearchError(error_msg) from e

    def search(self,
               questions: List[str],
               params: SearchParameters = SearchParameters()) -> Dict[str, Any]:
        """Perform semantic search for multiple questions

        Args:
            questions: List of question strings to search for
            params: Search configuration parameters

        Returns:
            Dictionary containing:
                - results: Formatted search results
                - statistics: Statistical analysis of results (if enabled)
                - summary: Human-readable summary of results

        Raises:
            SearchError: If search operation fails
        """
        try:
            # Generate embeddings for questions
            query_embeddings = self._generate_embeddings(questions)

            # Configure and perform vector search
            search_config = SearchConfiguration(num_neighbors=params.num_results)
            raw_results = self.vector_client.find_neighbors(
                deployed_index_id=self.deployed_index_id,
                queries=query_embeddings,
                config=search_config
            )

            # Process results and fetch metadata
            processed_results = self._process_search_results(
                raw_results,
                questions,
                params.min_similarity_score
            )

            # Format results
            formatted_results = self.result_formatter.to_detailed_dict(
                processed_results,
                include_metadata=params.include_metadata
            )

            # Prepare response
            response = {
                'results': formatted_results,
                'summary': self.result_formatter.to_summary_text(processed_results)
            }

            # Add statistics if enabled
            if params.compute_statistics:
                stats: StatisticalSummary = self.result_analyzer.calculate_statistics(
                    processed_results
                )
                response['statistics'] = stats.to_dict()

            logger.info("Search completed successfully")
            return response

        except Exception as e:
            error_msg = f"Search operation failed: {str(e)}"
            logger.error(error_msg)
            raise SearchError(error_msg) from e

def setup_logging(log_dir: str = 'app/log') -> None:
    """Configure logging settings

    Args:
        log_dir: Directory to store log files
    """
    try:
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

        logger.info(f"Logging initialized: {log_filename}")

    except Exception as e:
        print(f"Failed to setup logging: {str(e)}")
        raise

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

        # Configure search parameters
        search_params = SearchParameters(
            num_results=10,
            min_similarity_score=0.5,
            include_metadata=True,
            compute_statistics=True
        )

        # Initialize searcher and perform search
        searcher = SemanticSearcher()
        search_output = searcher.search(
            questions=test_questions,
            params=search_params
        )

        # Log results
        logger.info("\n=== Search Statistics ===")
        for key, value in search_output['statistics'].items():
            logger.info(f"{key}: {value}")

        logger.info("\n=== Search Results ===")
        logger.info(search_output['summary'])

    except Exception as e:
        logger.error(f"Main execution failed: {str(e)}")
        raise

if __name__ == "__main__":
    main()
