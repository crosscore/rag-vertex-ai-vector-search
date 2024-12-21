# app/common/utils/vector_search.py
"""
Module that provides basic functions for Vector Search operations.
Implements basic operations such as search, add, and delete data.
"""
from google.cloud import aiplatform
from google.api_core.exceptions import GoogleAPIError
from typing import List, Dict, Any, Optional
import logging
from ...common.utils.embeddings import embed_texts
from ...common.config import EMBEDDING_MODEL

logger = logging.getLogger(__name__)

class VectorSearchClient:
    """Client class that manages Vector Search operations"""

    def __init__(self,
                    project_id: str,
                    location: str,
                    index_endpoint: aiplatform.MatchingEngineIndexEndpoint):
        """
        Args:
            project_id: Project ID
            location: Region
            index_endpoint: Index endpoint to use
        """
        self.project_id = project_id
        self.location = location
        self.index_endpoint = index_endpoint
        aiplatform.init(project=project_id, location=location)

    def upsert_data_points(self,
                            deployed_index_id: str,
                            data_points: List[Dict[str, Any]]) -> None:
        """Add or update data points in the index
        Args:
            deployed_index_id: ID of the deployed index
            data_points: List of data points to add or update
                        Each data point is in the format {'id': str, 'embedding': List[float]}
        """
        try:
            logger.info(f"Start updating {len(data_points)} data points")
            self.index_endpoint.upsert_datapoints(
                deployed_index_id=deployed_index_id,
                datapoints=data_points
            )
            logger.info("Data points update completed")
        except GoogleAPIError as e:
            logger.error(f"Data points update error: {str(e)}")
            raise

    def remove_data_points(self,
                            deployed_index_id: str,
                            ids: List[str]) -> None:
        """Remove data points from the index
        Args:
            deployed_index_id: ID of the deployed index
            ids: List of IDs of data points to remove
        """
        try:
            logger.info(f"Start deleting {len(ids)} data points")
            self.index_endpoint.remove_datapoints(
                deployed_index_id=deployed_index_id,
                datapoint_ids=ids
            )
            logger.info("Data points deletion completed")
        except GoogleAPIError as e:
            logger.error(f"Data points deletion error: {str(e)}")
            raise

    def search(self,
                deployed_index_id: str,
                query: str,
                num_neighbors: int = 5,
                filter_expr: Optional[str] = None) -> List[Dict[str, Any]]:
        """Search for data points similar to the query
        Args:
            deployed_index_id: ID of the deployed index
            query: Search query string
            num_neighbors: Number of neighbors to retrieve
            filter_expr: Filter expression (optional)

        Returns:
            List of search results. Each element is of the form:
            {
                'id': Data point ID,
                'distance': Similarity score,
                'neighbor_count': Number of neighbors
            }
        """
        try:
            # Generate embedding of the query
            query_embedding = embed_texts([{'filename': 'query', 'content': query}], EMBEDDING_MODEL)[0]
            logger.info(f"Search query embedding generated: {len(query_embedding)} dimensions")

            # Execute nearest neighbor search
            response = self.index_endpoint.find_neighbors(
                deployed_index_id=deployed_index_id,
                queries=[query_embedding],
                num_neighbors=num_neighbors,
                filter=filter_expr
            )

            # Format the response
            results = []
            for neighbor in response[0].nearest_neighbors:
                results.append({
                    'id': neighbor.id,
                    'distance': neighbor.distance,
                    'neighbor_count': len(response[0].nearest_neighbors)
                })

            logger.info(f"Search completed: {len(results)} results found")
            return results

        except GoogleAPIError as e:
            logger.error(f"Search execution error: {str(e)}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error: {str(e)}")
            raise

    def get_index_stats(self, deployed_index_id: str) -> Dict[str, Any]:
        """Get index statistics

        Args:
            deployed_index_id: ID of the deployed index

        Returns:
            Dictionary containing statistics
        """
        try:
            stats = self.index_endpoint.get_index_stats(deployed_index_id)
            logger.info(f"Index statistics retrieved: {deployed_index_id}")
            return {
                'total_data_points': stats.total_data_points,
                'updated_at': stats.updated_at,
                'deployed_index_id': deployed_index_id
            }
        except GoogleAPIError as e:
            logger.error(f"Statistics retrieval error: {str(e)}")
            raise
