# app/common/utils/vector_search.py
"""Module for handling vector search operations using Vertex AI Matching Engine."""

from typing import List, Dict, Any, Optional
import logging
import google.auth
from google.api_core import exceptions as core_exceptions
from google.cloud import aiplatform
from google.cloud.aiplatform.matching_engine import (
    MatchingEngineIndexEndpoint,
    MatchNeighbor,
)

logger = logging.getLogger(__name__)

class VectorSearchClient:
    """Client class for vector similarity search operations"""

    def __init__(
        self,
        project_id: str,
        location: str,
        endpoint_name: str,
    ):
        """Initialize VectorSearchClient.

        Args:
            project_id (str): Google Cloud project ID
            location (str): Google Cloud region
            endpoint_name (str): Full resource name of the endpoint or endpoint ID
        """
        try:
            self.credentials, _ = google.auth.default()
            self.project_id = project_id
            self.location = location

            # Initialize Vertex AI
            aiplatform.init(project=project_id, location=location)

            # Get the endpoint
            self.endpoint = MatchingEngineIndexEndpoint(
                index_endpoint_name=endpoint_name,
                project=project_id,
                location=location,
            )
            logger.info(f"Successfully initialized VectorSearchClient with endpoint: {endpoint_name}")

        except core_exceptions.GoogleAPIError as e:
            logger.error(f"Failed to initialize VectorSearchClient: {str(e)}")
            raise

    def find_neighbors(
        self,
        deployed_index_id: str,
        queries: List[List[float]],
        num_neighbors: int = 10,
        filter_expr: Optional[str] = None,
    ) -> List[List[MatchNeighbor]]:
        """Search for nearest neighbors.

        Args:
            deployed_index_id (str): ID of the deployed index
            queries (List[List[float]]): List of query vectors
            num_neighbors (int, optional): Number of neighbors to return. Defaults to 10.
            filter_expr (Optional[str]): Filter expression for results

        Returns:
            List[List[MatchNeighbor]]: List of nearest neighbors for each query

        Raises:
            core_exceptions.GoogleAPIError: If search operation fails
        """
        try:
            results = self.endpoint.find_neighbors(
                deployed_index_id=deployed_index_id,
                queries=queries,
                num_neighbors=num_neighbors,
            )
            logger.info(f"Successfully found {len(results)} results")
            return results

        except core_exceptions.GoogleAPIError as e:
            logger.error(f"Search operation failed: {str(e)}")
            raise

    def get_datapoints(
        self,
        deployed_index_id: str,
        datapoint_ids: List[str],
    ) -> List[Dict[str, Any]]:
        """Retrieve specific datapoints by their IDs.

        Args:
            deployed_index_id (str): ID of the deployed index
            datapoint_ids (List[str]): List of datapoint IDs to retrieve

        Returns:
            List[Dict[str, Any]]: Retrieved datapoints

        Raises:
            core_exceptions.GoogleAPIError: If retrieval operation fails
        """
        try:
            datapoints = self.endpoint.read_index_datapoints(
                deployed_index_id=deployed_index_id,
                ids=datapoint_ids,
            )
            return [
                {
                    'id': dp.datapoint_id,
                    'vector': dp.feature_vector,
                }
                for dp in datapoints
            ]

        except core_exceptions.GoogleAPIError as e:
            logger.error(f"Failed to retrieve datapoints: {str(e)}")
            raise
