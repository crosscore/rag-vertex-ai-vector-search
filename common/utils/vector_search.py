# app/common/utils/vector_search.py
"""
Module that provides vector similarity search functionality using Vertex AI Matching Engine.
Handles initialization and search operations for vector similarity search.
"""
from typing import List, Dict, Any, Optional
import logging
from google.api_core import exceptions as core_exceptions
from google.cloud import aiplatform
from google.cloud.aiplatform.matching_engine.matching_engine_index_endpoint import (
    MatchingEngineIndexEndpoint,
    MatchNeighbor,
    Namespace,
)
from ...common.config import PROJECT_ID, REGION

logger = logging.getLogger(__name__)

class VectorSearchClient:
    """Class to manage vector similarity search operations"""

    def __init__(self, project_id: str = PROJECT_ID, location: str = REGION):
        """Initialize Vector Search client

        Args:
            project_id: Google Cloud project ID
            location: Google Cloud region
        """
        self.project_id = project_id
        self.location = location
        self.endpoint = None

    def initialize_endpoint(self, endpoint_name: str) -> None:
        """Initialize endpoint connection

        Args:
            endpoint_name: Full resource name of the endpoint or endpoint ID
        """
        try:
            # Initialize Vertex AI with project and location
            aiplatform.init(project=self.project_id, location=self.location)

            # Get the endpoint instance
            self.endpoint = MatchingEngineIndexEndpoint(
                index_endpoint_name=endpoint_name,
                project=self.project_id,
                location=self.location,
            )
            logger.info(f"Successfully initialized endpoint: {endpoint_name}")

        except core_exceptions.GoogleAPIError as e:
            logger.error(f"Failed to initialize endpoint: {str(e)}")
            raise

    def find_neighbors(self,
                        deployed_index_id: str,
                        queries: List[List[float]],
                        num_neighbors: int = 10,
                        filter_namespaces: Optional[List[Namespace]] = None) -> List[List[MatchNeighbor]]:
        """Perform vector similarity search

        Args:
            deployed_index_id: ID of the deployed index
            queries: List of query vectors
            num_neighbors: Number of nearest neighbors to return
            filter_namespaces: Optional filters to apply

        Returns:
            List of nearest neighbors for each query
        """
        if not self.endpoint:
            raise ValueError("Endpoint not initialized. Call initialize_endpoint first.")

        try:
            results = self.endpoint.find_neighbors(
                deployed_index_id=deployed_index_id,
                queries=queries,
                num_neighbors=num_neighbors,
                filter=filter_namespaces
            )
            logger.info(f"Found neighbors for {len(queries)} queries")
            return results

        except core_exceptions.GoogleAPIError as e:
            logger.error(f"Search operation failed: {str(e)}")
            raise

    def get_datapoints(self,
                        deployed_index_id: str,
                        datapoint_ids: List[str]) -> List[Dict[str, Any]]:
        """Retrieve specific datapoints by their IDs

        Args:
            deployed_index_id: ID of the deployed index
            datapoint_ids: List of datapoint IDs to retrieve

        Returns:
            List of retrieved datapoints
        """
        if not self.endpoint:
            raise ValueError("Endpoint not initialized. Call initialize_endpoint first.")

        try:
            datapoints = self.endpoint.read_index_datapoints(
                deployed_index_id=deployed_index_id,
                ids=datapoint_ids,
            )

            # Format the response
            formatted_datapoints = []
            for dp in datapoints:
                point_dict = {
                    'id': dp.datapoint_id,
                    'vector': dp.feature_vector,
                }

                if hasattr(dp, 'crowding_tag') and dp.crowding_tag:
                    point_dict['crowding_tag'] = dp.crowding_tag
                if hasattr(dp, 'restricts') and dp.restricts:
                    point_dict['restricts'] = dp.restricts

                formatted_datapoints.append(point_dict)

            logger.info(f"Retrieved {len(formatted_datapoints)} datapoints")
            return formatted_datapoints

        except core_exceptions.GoogleAPIError as e:
            logger.error(f"Datapoint retrieval failed: {str(e)}")
            raise

class SearchResult:
    """Helper class to format and manage search results"""

    def __init__(self, match_neighbor: MatchNeighbor):
        """Initialize search result

        Args:
            match_neighbor: MatchNeighbor object from search results
        """
        self.id = match_neighbor.id
        self.distance = match_neighbor.distance
        self.crowding_tag = getattr(match_neighbor, 'crowding_tag', None)
        self.restricts = getattr(match_neighbor, 'restricts', None)

    def to_dict(self) -> Dict[str, Any]:
        """Convert search result to dictionary

        Returns:
            Dictionary representation of the search result
        """
        result = {
            'id': self.id,
            'distance': float(self.distance),
        }

        if self.crowding_tag:
            result['crowding_tag'] = self.crowding_tag
        if self.restricts:
            result['restricts'] = self.restricts

        return result

def format_search_results(results: List[List[MatchNeighbor]]) -> List[List[Dict[str, Any]]]:
    """Format search results into a more usable structure

    Args:
        results: Raw search results from find_neighbors

    Returns:
        Formatted search results
    """
    formatted_results = []
    for result_group in results:
        group_results = [SearchResult(match).to_dict() for match in result_group]
        formatted_results.append(group_results)
    return formatted_results
