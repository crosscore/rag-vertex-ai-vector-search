# app/common/utils/vector_search.py
"""
Module that provides vector similarity search functionality using Vertex AI Matching Engine.
Handles low-level vector search operations and index endpoint management.
"""
from typing import List, Dict, Any, Optional
import logging
from dataclasses import dataclass
from google.api_core import exceptions as core_exceptions
from google.cloud import aiplatform
from google.cloud.aiplatform.matching_engine.matching_engine_index_endpoint import (
    MatchingEngineIndexEndpoint,
    MatchNeighbor,
    Namespace,
)

from ...common.config import PROJECT_ID, REGION

logger = logging.getLogger(__name__)

@dataclass
class SearchConfiguration:
    """Configuration for vector search operations"""
    num_neighbors: int = 10
    distance_measure: str = "DOT_PRODUCT_DISTANCE"
    namespace_filters: Optional[List[Namespace]] = None

class VectorSearchError(Exception):
    """Base exception class for vector search operations"""
    pass

class EndpointInitializationError(VectorSearchError):
    """Exception raised when endpoint initialization fails"""
    pass

class SearchOperationError(VectorSearchError):
    """Exception raised when search operation fails"""
    pass

class DatapointOperationError(VectorSearchError):
    """Exception raised when datapoint operation fails"""
    pass

class VectorSearchClient:
    """Class to manage vector similarity search operations"""

    def __init__(self, project_id: str = PROJECT_ID, location: str = REGION):
        """Initialize Vector Search client

        Args:
            project_id: Google Cloud project ID
            location: Google Cloud region

        Attributes:
            project_id: Stored project ID
            location: Stored region
            endpoint: MatchingEngineIndexEndpoint instance (initialized later)
        """
        self.project_id = project_id
        self.location = location
        self.endpoint: Optional[MatchingEngineIndexEndpoint] = None
        logger.info(f"Vector Search Client initialized for project {project_id} in {location}")

    def initialize_endpoint(self, endpoint_name: str) -> None:
        """Initialize endpoint connection

        Args:
            endpoint_name: Full resource name of the endpoint or endpoint ID

        Raises:
            EndpointInitializationError: If endpoint initialization fails
        """
        try:
            # Initialize Vertex AI
            aiplatform.init(project=self.project_id, location=self.location)

            # Get the endpoint instance
            self.endpoint = MatchingEngineIndexEndpoint(
                index_endpoint_name=endpoint_name,
                project=self.project_id,
                location=self.location,
            )
            logger.info(f"Successfully initialized endpoint: {endpoint_name}")

        except core_exceptions.GoogleAPIError as e:
            error_msg = f"Failed to initialize endpoint: {str(e)}"
            logger.error(error_msg)
            raise EndpointInitializationError(error_msg) from e
        except Exception as e:
            error_msg = f"Unexpected error during endpoint initialization: {str(e)}"
            logger.error(error_msg)
            raise EndpointInitializationError(error_msg) from e

    def find_neighbors(self,
                        deployed_index_id: str,
                        queries: List[List[float]],
                        config: Optional[SearchConfiguration] = None) -> List[List[MatchNeighbor]]:
        """Perform vector similarity search

        Args:
            deployed_index_id: ID of the deployed index
            queries: List of query vectors
            config: Optional search configuration parameters

        Returns:
            List of MatchNeighbor lists for each query

        Raises:
            SearchOperationError: If search operation fails
            ValueError: If endpoint is not initialized
        """
        if not self.endpoint:
            error_msg = "Endpoint not initialized. Call initialize_endpoint first."
            logger.error(error_msg)
            raise ValueError(error_msg)

        try:
            if config is None:
                config = SearchConfiguration()

            results = self.endpoint.find_neighbors(
                deployed_index_id=deployed_index_id,
                queries=queries,
                num_neighbors=config.num_neighbors,
                filter=config.namespace_filters
            )

            logger.info(f"Successfully found neighbors for {len(queries)} queries")
            logger.debug(f"Search configuration: {config}")

            return results

        except core_exceptions.GoogleAPIError as e:
            error_msg = f"Search operation failed: {str(e)}"
            logger.error(error_msg)
            raise SearchOperationError(error_msg) from e
        except Exception as e:
            error_msg = f"Unexpected error during search: {str(e)}"
            logger.error(error_msg)
            raise SearchOperationError(error_msg) from e

    def get_datapoints(self,
                        deployed_index_id: str,
                        datapoint_ids: List[str]) -> List[Dict[str, Any]]:
        """Retrieve specific datapoints by their IDs

        Args:
            deployed_index_id: ID of the deployed index
            datapoint_ids: List of datapoint IDs to retrieve

        Returns:
            List of datapoint dictionaries

        Raises:
            DatapointOperationError: If datapoint retrieval fails
            ValueError: If endpoint is not initialized
        """
        if not self.endpoint:
            error_msg = "Endpoint not initialized. Call initialize_endpoint first."
            logger.error(error_msg)
            raise ValueError(error_msg)

        try:
            datapoints = self.endpoint.read_index_datapoints(
                deployed_index_id=deployed_index_id,
                ids=datapoint_ids,
            )

            formatted_datapoints = []
            for dp in datapoints:
                point_dict = {
                    'id': dp.datapoint_id,
                    'vector': dp.feature_vector
                }

                # Add optional attributes if present
                if hasattr(dp, 'restricts') and dp.restricts:
                    point_dict['restricts'] = [
                        {
                            'namespace': r.namespace,
                            'allow_list': list(r.allow_list),
                            'deny_list': list(r.deny_list)
                        } for r in dp.restricts
                    ]

                if hasattr(dp, 'crowding_tag') and dp.crowding_tag:
                    point_dict['crowding_tag'] = {
                        'crowding_attribute': dp.crowding_tag.crowding_attribute
                    }

                formatted_datapoints.append(point_dict)

            logger.info(f"Successfully retrieved {len(formatted_datapoints)} datapoints")
            return formatted_datapoints

        except core_exceptions.GoogleAPIError as e:
            error_msg = f"Datapoint retrieval failed: {str(e)}"
            logger.error(error_msg)
            raise DatapointOperationError(error_msg) from e
        except Exception as e:
            error_msg = f"Unexpected error during datapoint retrieval: {str(e)}"
            logger.error(error_msg)
            raise DatapointOperationError(error_msg) from e
