# app/vector_store/utils/index_manager.py
"""
Module responsible for managing Vector Search indexes.
Provides functions for index creation, deployment, monitoring, etc.
"""
from google.cloud import aiplatform
from google.cloud.aiplatform_v1 import (
    IndexServiceClient,
    IndexEndpointServiceClient,
    Index,
    IndexEndpoint
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
    """Class to manage Vector Search indexes"""

    def __init__(self, project_id: str = PROJECT_ID, region: str = REGION):
        """
        Args:
            project_id: Project ID
            region: Region
        """
        self.project_id = project_id
        self.region = region
        self.parent = f"projects/{project_id}/locations/{region}"

        # Initialize API clients
        client_options = {"api_endpoint": f"{region}-aiplatform.googleapis.com"}
        self.index_client = IndexServiceClient(client_options=client_options)
        self.endpoint_client = IndexEndpointServiceClient(client_options=client_options)

        # Initialize Vertex AI
        aiplatform.init(project=project_id, location=region)

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
            Operation of the creation process

        Raises:
            GoogleAPIError: If index creation fails
        """
        try:
            # Prepare index configuration
            config = INDEX_CONFIG.copy()
            config['dimensions'] = dimension

            index = Index(
                display_name=display_name,
                description=description or f"Vector search index created at {time.strftime('%Y-%m-%d %H:%M:%S')}",
                metadata_schema_uri="gs://google-cloud-aiplatform/schema/matchingengine/metadata/nearest_neighbor_search_1.0.0.yaml",
                metadata={"config": config}
            )

            # Execute index creation operation
            operation = self.index_client.create_index(
                parent=self.parent,
                index=index
            )

            logger.info(f"Index creation started: {display_name}")
            return operation

        except GoogleAPIError as e:
            logger.error(f"Index creation error: {str(e)}")
            raise

    def create_endpoint(self,
                        display_name: str,
                        description: Optional[str] = None) -> Operation:
        """Create a new endpoint

        Args:
            display_name: Display name of the endpoint
            description: Description of the endpoint (optional)

        Returns:
            Operation of the creation process

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
            logger.error(f"Endpoint creation error: {str(e)}")
            raise

    def deploy_index(self,
                    index_name: str,
                    endpoint_name: str,
                    deployed_index_id: str) -> Operation:
        """Deploy an index to an endpoint

        Args:
            index_name: Name of the index to deploy
            endpoint_name: Name of the endpoint to deploy to
            deployed_index_id: ID of the deployed index

        Returns:
            Operation of the deployment process

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
            logger.error(f"Index deployment error: {str(e)}")
            raise

    def wait_for_operation(self,
                            operation: Operation,
                            timeout_minutes: int = DEPLOYMENT_TIMEOUT_MINUTES) -> Index:
        """Wait for the operation to complete

        Args:
            operation: Operation to wait for
            timeout_minutes: Timeout in minutes

        Returns:
            Result of the operation

        Raises:
            TimeoutError: If the operation does not complete within the specified time
            GoogleAPIError: If the operation fails
        """
        try:
            start_time = time.time()
            while True:
                if operation.done():
                    logger.info("Operation completed")
                    return operation.result()

                if time.time() - start_time > timeout_minutes * 60:
                    raise TimeoutError(f"Operation timed out: {timeout_minutes} minutes")

                logger.debug("Waiting for operation to complete...")
                time.sleep(DEPLOYMENT_CHECK_INTERVAL)

        except GoogleAPIError as e:
            logger.error(f"Error while waiting for operation: {str(e)}")
            raise

    def get_deployment_state(self,
                            endpoint_name: str,
                            deployed_index_id: str) -> Dict[str, Any]:
        """Get the deployment state

        Args:
            endpoint_name: Endpoint name
            deployed_index_id: Deployed index ID

        Returns:
            Dictionary containing deployment state information

        Raises:
            GoogleAPIError: If state retrieval fails
        """
        try:
            endpoint = self.endpoint_client.get_index_endpoint(name=endpoint_name)

            for deployed_index in endpoint.deployed_indexes:
                if deployed_index.id == deployed_index_id:
                    # Check for the existence of index_sync_time to determine deployment state
                    is_synced = hasattr(deployed_index, 'index_sync_time')

                    state = {
                        "state": "DEPLOYED" if is_synced else "DEPLOYING",
                        "deployment_group": deployed_index.deployment_group,
                        "create_time": deployed_index.create_time,
                        "index_sync_time": getattr(deployed_index, 'index_sync_time', None)
                    }
                    logger.info(f"Deployment state: {state}")
                    return state

            logger.warning(f"Deployed index not found: {deployed_index_id}")
            return {"state": "NOT_FOUND"}

        except GoogleAPIError as e:
            logger.error(f"Deployment state retrieval error: {str(e)}")
            raise
