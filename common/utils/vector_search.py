# app/common/utils/vector_search.py
from google.cloud.aiplatform.matching_engine.matching_engine_index_endpoint import MatchingEngineIndexEndpoint
import ssl
import grpc
import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from google.api_core import exceptions as core_exceptions

from ...common.config import PROJECT_ID, REGION, ENDPOINT_RESOURCE_ID

logger = logging.getLogger(__name__)

@dataclass
class SearchConfiguration:
    """Configuration for vector search operations"""
    num_neighbors: int = 10
    distance_measure: str = "DOT_PRODUCT_DISTANCE"
    namespace_filters: Optional[List[Any]] = None

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
        self.endpoint_resource_id = ENDPOINT_RESOURCE_ID
        endpoint_name = f"projects/{self.project_id}/locations/{self.location}/indexEndpoints/{self.endpoint_resource_id}"
        self.endpoint = MatchingEngineIndexEndpoint(
            index_endpoint_name=endpoint_name,
            project=self.project_id,
            location=self.location
        )
        logger.info(f"Vector Search Client initialized for project {project_id} in {location}")

    def _initialize_ssl_context(self) -> None:
        """Initialize SSL context for secure connections"""
        try:
            ssl_context = ssl.create_default_context()
            ssl_context.check_hostname = False
            ssl_context.verify_mode = ssl.CERT_NONE

            # gRPCチャンネルオプションの設定
            self.channel_credentials = grpc.ssl_channel_credentials()

            logger.info("SSL context initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize SSL context: {str(e)}")
            raise

    def find_neighbors(self,
                    deployed_index_id: str,
                    queries: List[List[float]],
                    config: Optional[SearchConfiguration] = None) -> List[List[Any]]:
        """Perform vector similarity search

        Args:
            deployed_index_id: ID of the deployed index
            queries: List of query vectors
            config: Optional search configuration parameters

        Returns:
            List of MatchNeighbor lists for each query

        Raises:
            SearchOperationError: If search operation fails
        """
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
                ids=datapoint_ids
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
