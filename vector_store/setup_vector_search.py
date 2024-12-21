from typing import List, Dict, Any
import logging
import os
import uuid
import time
from google.cloud.aiplatform_v1 import IndexServiceClient
from google.cloud.aiplatform_v1.types import IndexDatapoint, UpsertDatapointsRequest

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
        self.project_id = PROJECT_ID
        self.region = REGION
        self.firestore_manager = FirestoreManager()
        self.index_manager = IndexManager()
        self.logger = logging.getLogger(__name__)

    def process_texts(self, texts: List[Dict[str, str]]) -> Dict[str, Any]:
        """Process texts to generate embeddings and create datapoints

        Args:
            texts: List of texts to process. Each element should be a dictionary with
                'filename' and 'content' keys.

        Returns:
            Dictionary containing processed information:
                - datapoints: List of IndexDatapoint objects
                - metadata_list: List of metadata for Firestore
                - dimension: Dimension of the embeddings

        Raises:
            Exception: If text processing fails
        """
        try:
            # Generate embeddings
            self.logger.info("Generating embeddings...")
            embeddings = embed_texts(texts)
            dimension = len(embeddings[0])

            # Generate data point IDs
            data_point_ids = [str(uuid.uuid4()) for _ in texts]

            # Create IndexDatapoints for Vector Search
            datapoints = [
                IndexDatapoint(
                    datapoint_id=data_point_id,
                    feature_vector=embedding
                )
                for data_point_id, embedding in zip(data_point_ids, embeddings)
            ]

            # Prepare metadata for Firestore
            metadata_list = [
                {
                    'data_point_id': data_point_id,
                    'filename': text['filename'],
                    'content': text['content'],
                    'additional_metadata': {
                        'embedding_dimension': dimension
                    }
                }
                for data_point_id, text in zip(data_point_ids, texts)
            ]

            return {
                'datapoints': datapoints,
                'metadata_list': metadata_list,
                'dimension': dimension
            }
        except Exception as e:
            self.logger.error(f"Text processing error: {str(e)}")
            raise

    def setup_vector_search(self, texts: List[Dict[str, str]]) -> Dict[str, Any]:
        """Executes Vector Search environment setup

        Args:
            texts: List of texts to be used as initial data

        Returns:
            Dictionary containing setup information:
                - index_name: Name of the created index
                - endpoint_name: Name of the created endpoint
                - deployment_state: Final deployment state

        Raises:
            Exception: If setup fails
        """
        start_time = time.time()
        try:
            self.logger.info("Starting Vector Search setup")

            # Process texts and generate embeddings
            process_result = self.process_texts(texts)
            datapoints = process_result['datapoints']
            metadata_list = process_result['metadata_list']
            dimension = process_result['dimension']

            # Create index
            self.logger.info("Creating index...")
            index_op = self.index_manager.create_index(
                display_name=INDEX_DISPLAY_NAME,
                dimension=dimension,
                description="RAG system vector search index"
            )
            index_result = self.index_manager.wait_for_operation(index_op)
            index_name = index_result.name

            # Create endpoint
            self.logger.info("Creating endpoint...")
            endpoint_op = self.index_manager.create_endpoint(
                display_name=ENDPOINT_DISPLAY_NAME,
                description="RAG system vector search endpoint"
            )
            endpoint_result = self.index_manager.wait_for_operation(endpoint_op)
            endpoint_name = endpoint_result.name

            # Save metadata to Firestore
            self.logger.info("Saving metadata to Firestore...")
            self.firestore_manager.batch_save_text_metadata(
                FIRESTORE_COLLECTION,
                metadata_list
            )

            # Insert vectors into index
            self.logger.info("Inserting vectors into index...")
            request = UpsertDatapointsRequest(
                index=index_name,
                datapoints=datapoints
            )
            client = IndexServiceClient(client_options={
                "api_endpoint": f"{self.region}-aiplatform.googleapis.com"
            })
            client.upsert_datapoints(request=request)

            # Deploy index
            self.logger.info("Deploying index...")
            deploy_op = self.index_manager.deploy_index(
                index_name=index_name,
                endpoint_name=endpoint_name,
                deployed_index_id=DEPLOYED_INDEX_ID
            )
            self.index_manager.wait_for_operation(deploy_op)

            # Get final deployment state
            deployment_state = self.index_manager.get_deployment_state(
                endpoint_name,
                DEPLOYED_INDEX_ID
            )

            if deployment_state['state'] != "DEPLOYED":
                raise RuntimeError(f"Deployment failed with state: {deployment_state['state']}")

            total_time = int(time.time() - start_time)
            self.logger.info(f"Vector Search setup completed successfully in {total_time} seconds")

            return {
                'index_name': index_name,
                'endpoint_name': endpoint_name,
                'deployment_state': deployment_state
            }

        except Exception as e:
            total_time = int(time.time() - start_time)
            self.logger.error(f"Vector Search setup failed after {total_time} seconds: {str(e)}")
            raise

def load_md_files(md_folder_path: str) -> List[Dict[str, str]]:
    """Loads information from MD files

    Args:
        md_folder_path: Path to the folder containing MD files

    Returns:
        List of dictionaries containing file information
    """
    md_files_info = []
    for filename in os.listdir(md_folder_path):
        if filename.endswith(".md"):
            file_path = os.path.join(md_folder_path, filename)
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            md_files_info.append({
                'filename': filename,
                'content': content
            })
    return md_files_info

def setup_logging():
    """Configure logging settings"""
    log_dir = 'app/log'
    os.makedirs(log_dir, exist_ok=True)
    log_filename = os.path.join(log_dir, 'vector_store_setup.log')

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_filename, mode='w')
        ]
    )

def main():
    """Main execution function"""
    setup_logging()
    logger = logging.getLogger(__name__)

    try:
        md_folder_path = os.path.join(os.path.dirname(__file__), "md")
        md_files_info = load_md_files(md_folder_path)

        setup = VectorStoreSetup()
        result = setup.setup_vector_search(md_files_info)

        logger.info("Setup completed successfully")
        logger.info(f"Index name: {result['index_name']}")
        logger.info(f"Endpoint name: {result['endpoint_name']}")
        logger.info(f"Deployment state: {result['deployment_state']}")

    except Exception as e:
        logger.error(f"Setup failed: {str(e)}")
        raise

if __name__ == "__main__":
    main()
