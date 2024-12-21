# app/vector_store/utils/index_manager.py
from google.cloud.aiplatform_v1 import (
    IndexServiceClient,
    IndexEndpointServiceClient,
    Index,
    IndexEndpoint,
)
from google.cloud.aiplatform_v1.types import Index
from google.api_core.exceptions import GoogleAPIError
from google.api_core.operation import Operation
import time
from typing import Optional, Dict, Any
import logging
from ...common.config import (
    PROJECT_ID,
    REGION,
    INDEX_CONFIG,
    DEPLOYMENT_CONFIG,
    DEPLOYMENT_TIMEOUT_MINUTES,
    DEPLOYMENT_CHECK_INTERVAL
)

logger = logging.getLogger(__name__)

class IndexManager:
    """Class to manage Vector Search indexes using low-level Vertex AI API"""

    def __init__(self, project_id: str = PROJECT_ID, region: str = REGION):
        """Initialize the Index Manager

        Args:
            project_id: Project ID
            region: Region
        """
        self.project_id = project_id
        self.region = region
        self.parent = f"projects/{project_id}/locations/{region}"

        # Initialize clients with proper endpoint configuration
        client_options = {"api_endpoint": f"{region}-aiplatform.googleapis.com"}
        self.index_client = IndexServiceClient(client_options=client_options)
        self.endpoint_client = IndexEndpointServiceClient(client_options=client_options)

    def create_index(self,
                    display_name: str,
                    dimension: int,
                    description: Optional[str] = None) -> Operation:
        """Create a new index

        Args:
            display_name: Display name of the index
            dimension: Number of dimensions of the vector
            description: Description of the index (optional)

        Returns:
            Operation: The long-running operation for index creation

        Raises:
            GoogleAPIError: If index creation fails
        """
        try:
            # Prepare index configuration
            config = INDEX_CONFIG.copy()
            config['dimensions'] = dimension

            # Create index with StreamUpdate enabled
            index = Index(
                display_name=display_name,
                description=description or f"Vector search index created at {time.strftime('%Y-%m-%d %H:%M:%S')}",
                metadata_schema_uri="gs://google-cloud-aiplatform/schema/matchingengine/metadata/nearest_neighbor_search_1.0.0.yaml",
                metadata={
                    "config": config
                },
                index_update_method=Index.IndexUpdateMethod.STREAM_UPDATE
            )

            # Execute index creation operation
            operation = self.index_client.create_index(
                parent=self.parent,
                index=index
            )

            logger.info(f"Index creation started: {display_name}")
            return operation

        except GoogleAPIError as e:
            error_msg = f"Failed to create index: {str(e)}"
            logger.error(error_msg)
            raise GoogleAPIError(error_msg) from e

    def create_endpoint(self,
                        display_name: str,
                        description: Optional[str] = None) -> Operation:
        """Create a new endpoint

        Args:
            display_name: Display name of the endpoint
            description: Description of the endpoint (optional)

        Returns:
            Operation: The long-running operation for endpoint creation

        Raises:
            GoogleAPIError: If endpoint creation fails
        """
        try:
            endpoint = IndexEndpoint(
                display_name=display_name,
                description=description or f"Vector search endpoint created at {time.strftime('%Y-%m-%d %H:%M:%S')}",
                public_endpoint_enabled=True
            )

            operation = self.endpoint_client.create_index_endpoint(
                parent=self.parent,
                index_endpoint=endpoint
            )

            logger.info(f"Endpoint creation started: {display_name}")
            return operation

        except GoogleAPIError as e:
            error_msg = f"Failed to create endpoint: {str(e)}"
            logger.error(error_msg)
            raise GoogleAPIError(error_msg) from e

    def deploy_index(self,
                    index_name: str,
                    endpoint_name: str,
                    deployed_index_id: str) -> Operation:
        """Deploy an index to an endpoint

        Args:
            index_name: Name of the index to deploy
            endpoint_name: Name of the endpoint to deploy to
            deployed_index_id: ID for the deployed index

        Returns:
            Operation: The long-running operation for index deployment

        Raises:
            GoogleAPIError: If deployment fails
        """
        try:
            deploy_request = {
                "index_endpoint": endpoint_name,
                "deployed_index": {
                    "id": deployed_index_id,
                    "index": index_name,
                    "display_name": f"Deployed index {deployed_index_id}",
                    "dedicated_resources": DEPLOYMENT_CONFIG
                }
            }

            operation = self.endpoint_client.deploy_index(request=deploy_request)
            logger.info(f"Index deployment started: {deployed_index_id}")
            return operation

        except GoogleAPIError as e:
            error_msg = f"Failed to deploy index: {str(e)}"
            logger.error(error_msg)
            raise GoogleAPIError(error_msg) from e

    def wait_for_operation(self,
                            operation: Operation,
                            timeout_minutes: int = DEPLOYMENT_TIMEOUT_MINUTES) -> Any:
        """Wait for the operation to complete

        Args:
            operation: Operation to wait for
            timeout_minutes: Timeout in minutes

        Returns:
            Any: Result of the operation

        Raises:
            TimeoutError: If the operation does not complete within the specified time
            GoogleAPIError: If the operation fails
        """
        try:
            start_time = time.time()
            while True:
                if operation.done():
                    logger.info("Operation completed successfully")
                    return operation.result()

                if time.time() - start_time > timeout_minutes * 60:
                    error_msg = f"Operation timed out after {timeout_minutes} minutes"
                    logger.error(error_msg)
                    raise TimeoutError(error_msg)

                logger.debug("Waiting for operation to complete...")
                time.sleep(DEPLOYMENT_CHECK_INTERVAL)

        except GoogleAPIError as e:
            error_msg = f"Operation failed: {str(e)}"
            logger.error(error_msg)
            raise GoogleAPIError(error_msg) from e

    def get_deployment_state(self,
                            endpoint_name: str,
                            deployed_index_id: str) -> Dict[str, Any]:
        """Get the deployment state

        Args:
            endpoint_name: Name of the endpoint
            deployed_index_id: ID of the deployed index

        Returns:
            Dict[str, Any]: Dictionary containing deployment state information

        Raises:
            GoogleAPIError: If state retrieval fails
        """
        try:
            endpoint = self.endpoint_client.get_index_endpoint(name=endpoint_name)

            for deployed_index in endpoint.deployed_indexes:
                if deployed_index.id == deployed_index_id:
                    # Check index_sync_time to determine deployment state
                    is_synced = hasattr(deployed_index, 'index_sync_time')

                    state = {
                        "state": "DEPLOYED" if is_synced else "DEPLOYING",
                        "deployment_group": deployed_index.deployment_group,
                        "create_time": deployed_index.create_time,
                        "index_sync_time": getattr(deployed_index, 'index_sync_time', None)
                    }
                    logger.info(f"Deployment state retrieved: {state}")
                    return state

            logger.warning(f"Deployed index not found: {deployed_index_id}")
            return {"state": "NOT_FOUND"}

        except GoogleAPIError as e:
            error_msg = f"Failed to get deployment state: {str(e)}"
            logger.error(error_msg)
            raise GoogleAPIError(error_msg) from e
