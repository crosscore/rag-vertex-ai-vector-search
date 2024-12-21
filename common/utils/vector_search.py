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

logger = logging.getLogger(__name__)

def get_matching_engine_endpoint(
    project_id: str,
    location: str,
    endpoint_name: str,
) -> MatchingEngineIndexEndpoint:
    """Get a MatchingEngineIndexEndpoint instance.

    Args:
        project_id (str): Google Cloud project ID
        location (str): Google Cloud region
        endpoint_name (str): Full resource name of the endpoint or endpoint ID.
        Format: 'projects/{project}/locations/{location}/indexEndpoints/{endpoint_id}' or just 'endpoint_id'

    Returns:
        MatchingEngineIndexEndpoint: Initialized endpoint instance

    Raises:
        core_exceptions.GoogleAPIError: If initialization fails
    """
    try:
        # Initialize Vertex AI with project and location
        aiplatform.init(project=project_id, location=location)

        # Get the endpoint instance
        endpoint = MatchingEngineIndexEndpoint(
            index_endpoint_name=endpoint_name,
            project=project_id,
            location=location,
        )
        logger.info(f"Successfully initialized endpoint: {endpoint_name}")
        return endpoint

    except core_exceptions.GoogleAPIError as e:
        error_msg = f"Failed to initialize endpoint: {str(e)}"
        logger.error(error_msg)
        raise core_exceptions.GoogleAPIError(error_msg) from e


def get_neighbors(
    endpoint: MatchingEngineIndexEndpoint,
    deployed_index_id: str,
    queries: List[List[float]],
    num_neighbors: int = 10,
    filter_namespaces: Optional[List[Namespace]] = None,
) -> List[List[MatchNeighbor]]:
    """Convenience function to perform vector similarity search.

    Args:
        endpoint (MatchingEngineIndexEndpoint): The initialized endpoint instance
        deployed_index_id (str): ID of the deployed index to search
        queries (List[List[float]]): List of query vectors to search for
        num_neighbors (int): Number of nearest neighbors to return for each query. Defaults to 10.
        filter_namespaces (Optional[List[Namespace]]): Optional filters to apply to the search.
            Each Namespace can contain allow_tokens and deny_tokens.

    Returns:
        List[List[MatchNeighbor]]: List of nearest neighbors for each query.
            Each MatchNeighbor contains:
            - id: The datapoint ID
            - distance: Distance score to the query
            - crowding_tag: Optional crowding tag if specified
            - restricts: Optional restrictions if specified

    Raises:
        core_exceptions.GoogleAPIError: If the search operation fails
    """
    try:
        results = endpoint.find_neighbors(
            deployed_index_id=deployed_index_id,
            queries=queries,
            num_neighbors=num_neighbors,
            filter=filter_namespaces
        )
        logger.info(
            f"Successfully found neighbors for {len(queries)} queries, "
            f"with up to {num_neighbors} neighbors each"
        )
        return results

    except core_exceptions.GoogleAPIError as e:
        error_msg = f"Failed to perform similarity search: {str(e)}"
        logger.error(error_msg)
        raise core_exceptions.GoogleAPIError(error_msg) from e


def get_datapoints(
    endpoint: MatchingEngineIndexEndpoint,
    deployed_index_id: str,
    datapoint_ids: List[str],
) -> List[Dict[str, Any]]:
    """Retrieve specific datapoints by their IDs.

    Args:
        endpoint (MatchingEngineIndexEndpoint): The initialized endpoint instance
        deployed_index_id (str): ID of the deployed index
        datapoint_ids (List[str]): List of datapoint IDs to retrieve

    Returns:
        List[Dict[str, Any]]: List of retrieved datapoints.
            Each datapoint contains:
            - id: The datapoint ID
            - vector: The feature vector
            - crowding_tag: Optional crowding tag if specified
            - restricts: Optional restrictions if specified

    Raises:
        core_exceptions.GoogleAPIError: If the retrieval operation fails
    """
    try:
        datapoints = endpoint.read_index_datapoints(
            deployed_index_id=deployed_index_id,
            ids=datapoint_ids,
        )

        # Convert to dictionary format for easier consumption
        formatted_datapoints = []
        for dp in datapoints:
            point_dict = {
                'id': dp.datapoint_id,
                'vector': dp.feature_vector,
            }

            # Add optional fields if they exist
            if hasattr(dp, 'crowding_tag') and dp.crowding_tag:
                point_dict['crowding_tag'] = dp.crowding_tag
            if hasattr(dp, 'restricts') and dp.restricts:
                point_dict['restricts'] = dp.restricts

            formatted_datapoints.append(point_dict)

        logger.info(f"Successfully retrieved {len(formatted_datapoints)} datapoints")
        return formatted_datapoints

    except core_exceptions.GoogleAPIError as e:
        error_msg = f"Failed to retrieve datapoints: {str(e)}"
        logger.error(error_msg)
        raise core_exceptions.GoogleAPIError(error_msg) from e
