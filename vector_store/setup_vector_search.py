# app/vector_store/setup_vector_search.py
"""
Main module responsible for executing Vector Store setup.
Integrates index creation, data storage in Firestore, and deployment execution.
"""
import uuid
import time
from typing import List, Dict, Any
import logging
import os
from google.cloud.aiplatform_v1 import IndexServiceClient
from google.cloud.aiplatform_v1.types import (
    IndexDatapoint,
    UpsertDatapointsRequest
)
from ..common.config import (
    PROJECT_ID,
    REGION,
    INDEX_DISPLAY_NAME,
    ENDPOINT_DISPLAY_NAME,
    DEPLOYED_INDEX_ID,
    FIRESTORE_COLLECTION
)
from ..common.utils.embeddings import embed_texts
from .utils.firestore_ops import FirestoreManager
from .utils.index_manager import IndexManager

class VectorStoreSetup:
    """Class to manage Vector Store setup execution"""

    def __init__(self):
        """Initialize necessary managers and clients"""
        self.firestore_manager = FirestoreManager()
        self.index_manager = IndexManager(PROJECT_ID, REGION)
        self.logger = logging.getLogger(__name__)

    def process_texts(self, texts: List[Dict[str, str]]) -> Dict[str, Any]:
        """Process texts to generate embeddings and create datapoints

        Args:
            texts: List of texts to process. Each element should be a dictionary with
                'filename' and 'content' keys.

        Returns:
            Dictionary containing:
                - data_point_ids: List of generated datapoint IDs
                - dimension: Dimension of the embeddings

        Raises:
            GoogleAPIError: If datapoint creation fails
        """
        try:
            # Generate data point IDs
            data_point_ids = [str(uuid.uuid4()) for _ in texts]

            # Generate embeddings
            text_contents = [text_info['content'] for text_info in texts]
            embeddings = embed_texts(texts)

            # Create IndexDatapoints
            datapoints = [
                IndexDatapoint(
                    datapoint_id=data_point_id,
                    feature_vector=embedding
                )
                for data_point_id, embedding in zip(data_point_ids, embeddings)
            ]

            # Save metadata to Firestore
            metadata_list = [
                {
                    'data_point_id': data_point_id,
                    'text': text,
                    'additional_metadata': {
                        'embedding_dimension': len(embeddings[0])
                    }
                }
                for data_point_id, text in zip(data_point_ids, text_contents)
            ]
            self.firestore_manager.batch_save_text_metadata(
                FIRESTORE_COLLECTION,
                metadata_list
            )

            # Get index name from the current operation's context
            index_name = f"projects/{self.project_id}/locations/{self.region}/indexes/{INDEX_DISPLAY_NAME}"

            # Create and execute upsert request
            request = UpsertDatapointsRequest(
                index=index_name,
                datapoints=datapoints
            )

            # Create index client and upsert
            client = IndexServiceClient()
            client.upsert_datapoints(request=request)

            return {
                'data_point_ids': data_point_ids,
                'dimension': len(embeddings[0])
            }
        except Exception as e:
            self.logger.error(f"Text processing error: {str(e)}")
            raise

    def setup_vector_search(self,
                            texts: List[Dict[str, str]]) -> None:
        """Executes Vector Search environment setup

        Args:
            texts: List of texts to be used as initial data

        Returns:
            None

        Raises:
            Exception: If an error occurs during setup
        """
        start_time = time.time()
        try:
            self.logger.info("Starting Vector Search setup")

            # Process texts
            process_start = time.time()
            process_result = self.process_texts(texts)
            data_point_ids = process_result['data_point_ids']
            embeddings = process_result['embeddings']
            dimension = process_result['dimension']
            self.logger.info(f"Text processing time: {int(time.time() - process_start)} seconds")

            # Create index
            self.logger.info("Start creating index")
            index_start = time.time()
            index_op = self.index_manager.create_index(
                display_name=INDEX_DISPLAY_NAME,
                dimension=dimension,
                description="RAG system vector search index"
            )
            index_result = self.index_manager.wait_for_operation(index_op)
            index_name = index_result.name
            self.logger.info(f"Index created: {index_name}")
            self.logger.info(f"Index creation time: {int(time.time() - index_start)} seconds")

            # Create endpoint
            self.logger.info("Start creating endpoint")
            endpoint_start = time.time()
            endpoint_op = self.index_manager.create_endpoint(
                display_name=ENDPOINT_DISPLAY_NAME,
                description="RAG system vector search endpoint"
            )
            endpoint_result = self.index_manager.wait_for_operation(endpoint_op)
            endpoint_name = endpoint_result.name
            self.logger.info(f"Endpoint created: {endpoint_name}")
            self.logger.info(f"Endpoint creation time: {int(time.time() - endpoint_start)} seconds")

            # Deploy index
            self.logger.info("Start deploying index")
            deploy_start = time.time()
            deploy_op = self.index_manager.deploy_index(
                index_name=index_name,
                endpoint_name=endpoint_name,
                deployed_index_id=DEPLOYED_INDEX_ID
            )
            self.index_manager.wait_for_operation(deploy_op)

            # Get endpoint information after deployment
            endpoint_info = self.index_manager.endpoint_client.get_index_endpoint(
                name=endpoint_name
            )

            deploy_time = int(time.time() - deploy_start)
            self.logger.info("Index deployment completed")
            self.logger.info(f"Deployment time: {deploy_time} seconds")
            self.logger.info(f"Public endpoint: {endpoint_info.public_endpoint_domain_name}")

            # Output deployed index information to logs
            for deployed_index in endpoint_info.deployed_indexes:
                if deployed_index.id == DEPLOYED_INDEX_ID:
                    self.logger.info(f"Deployed index information:")
                    self.logger.info(f"  ID: {deployed_index.id}")
                    self.logger.info(f"  Creation time: {deployed_index.create_time}")
                    self.logger.info(f"  Index path: {deployed_index.index}")
                    break

            # Check deployment status
            state = self.index_manager.get_deployment_state(
                endpoint_name,
                DEPLOYED_INDEX_ID
            )
            if state['state'] == "DEPLOYED":
                total_time = int(time.time() - start_time)
                self.logger.info("Vector Search setup completed successfully")
                self.logger.info(f"Total execution time: {total_time} seconds")
                self.logger.info(f"Deployment information:")
                self.logger.info(f"  Deployment group: {state['deployment_group']}")
                self.logger.info(f"  Creation time: {state['create_time']}")
                self.logger.info(f"  Index sync time: {state['index_sync_time']}")
            else:
                self.logger.error(f"Deployment issue detected: {state}")
                raise RuntimeError(f"Deployment failed with state: {state['state']}")

        except Exception as e:
            total_time = int(time.time() - start_time)
            self.logger.error(f"Vector Search setup error: {str(e)}")
            self.logger.error(f"Execution time until error: {total_time} seconds")
            raise

def load_md_files(md_folder_path: str) -> List[Dict[str, str]]:
    """Loads information from MD files and returns it as a list of dictionaries

    Args:
        md_folder_path: Path to the folder containing the MD files

    Returns:
        List of dictionaries containing MD file information
        Each dictionary has the filename (without extension) as the key and the file content as the value
    """
    md_files_info = []
    for filename in os.listdir(md_folder_path):
        if filename.endswith(".md"):
            file_path = os.path.join(md_folder_path, filename)
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            md_files_info.append({'filename': filename, 'content': content})
    return md_files_info

def main():
    """Main execution function"""
    # Log settings
    log_dir = 'app/log'
    os.makedirs(log_dir, exist_ok=True)
    log_filename = os.path.join(log_dir, 'vector_store_setup.log')
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_filename, mode='w')  # Overwrite each time with 'w' mode
        ]
    )
    logger = logging.getLogger(__name__)

    # Path to the folder containing MD files
    md_folder_path = os.path.join(os.path.dirname(__file__), "md")

    try:
        # Load information from MD files
        md_files_info = load_md_files(md_folder_path)

        setup = VectorStoreSetup()
        setup.setup_vector_search(md_files_info)
    except Exception as e:
        logger.error(f"Execution error: {str(e)}")
        raise

if __name__ == "__main__":
    main()
