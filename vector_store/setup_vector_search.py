# app/vector_store/setup_vector_search.py
from typing import List, Dict, Any
import logging
import os
import uuid
from datetime import datetime
import time
from google.cloud.aiplatform_v1 import IndexDatapoint

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
        """Initialize necessary managers"""
        self.project_id = PROJECT_ID
        self.region = REGION
        self.firestore_manager = FirestoreManager()
        self.index_manager = IndexManager()
        self.logger = logging.getLogger(__name__)

    def process_texts(self, texts: List[Dict[str, str]]) -> Dict[str, Any]:
        """Process texts to generate embeddings and create datapoints

        Args:
            texts: List of texts to process. Each text should be a dictionary with
                    'filename' and 'content' keys.

        Returns:
            Dictionary containing:
                - datapoints: List of IndexDatapoint objects
                - metadata_list: List of metadata dictionaries
                - dimension: Dimension of the embeddings

        Raises:
            Exception: If text processing fails
        """
        try:
            # Generate embeddings
            embeddings = embed_texts(texts)
            dimension = len(embeddings[0])
            self.logger.info(f"Generated embeddings with dimension: {dimension}")

            # Generate unique IDs for data points
            data_point_ids = [str(uuid.uuid4()) for _ in texts]

            # Create IndexDatapoints with metadata
            datapoints = []
            for data_point_id, embedding, text in zip(data_point_ids, embeddings, texts):
                # Create restrictions for filtering
                file_restrict = IndexDatapoint.Restriction(
                    namespace="file_type",
                    allow_list=["markdown"]
                )
                content_restrict = IndexDatapoint.Restriction(
                    namespace="content_type",
                    allow_list=["documentation"]
                )

                # Add numeric restrictions
                dimension_restrict = IndexDatapoint.NumericRestriction(
                    namespace="embedding_dimension",
                    value_int=dimension
                )
                content_length_restrict = IndexDatapoint.NumericRestriction(
                    namespace="content_length",
                    value_int=len(text['content'])
                )

                # Add crowding tag based on filename
                crowding = IndexDatapoint.CrowdingTag(
                    crowding_attribute=text['filename']
                )

                # Create datapoint with all metadata
                datapoint = IndexDatapoint(
                    datapoint_id=data_point_id,
                    feature_vector=embedding,
                    restricts=[file_restrict, content_restrict],
                    numeric_restricts=[dimension_restrict, content_length_restrict],
                    crowding_tag=crowding
                )
                datapoints.append(datapoint)

            # Prepare metadata for Firestore
            metadata_list = [
                {
                    'data_point_id': data_point_id,
                    'filename': text['filename'],
                    'content': text['content'],
                    'additional_metadata': {
                        'embedding_dimension': dimension,
                        'content_length': len(text['content']),
                        'file_type': 'markdown',
                        'content_type': 'documentation',
                        'created_at': datetime.now().isoformat()
                    }
                }
                for data_point_id, text in zip(data_point_ids, texts)
            ]

            self.logger.info(f"Processed {len(texts)} texts successfully")
            return {
                'datapoints': datapoints,
                'metadata_list': metadata_list,
                'dimension': dimension
            }

        except Exception as e:
            error_msg = f"Text processing error: {str(e)}"
            self.logger.error(error_msg)
            raise Exception(error_msg) from e

    def setup_vector_search(self, texts: List[Dict[str, str]]) -> Dict[str, Any]:
        """Execute Vector Search environment setup

        Args:
            texts: List of texts to be used as initial data. Each text should be a
                    dictionary with 'filename' and 'content' keys.

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
            self.logger.info(f"Creating index: {INDEX_DISPLAY_NAME}")
            index_op = self.index_manager.create_index(
                display_name=INDEX_DISPLAY_NAME,
                dimension=dimension,
                description="RAG system vector search index"
            )
            index_result = self.index_manager.wait_for_operation(index_op)
            index_name = index_result.name
            self.logger.info(f"Index created: {index_name}")

            # Create endpoint
            self.logger.info(f"Creating endpoint: {ENDPOINT_DISPLAY_NAME}")
            endpoint_op = self.index_manager.create_endpoint(
                display_name=ENDPOINT_DISPLAY_NAME,
                description="RAG system vector search endpoint"
            )
            endpoint_result = self.index_manager.wait_for_operation(endpoint_op)
            endpoint_name = endpoint_result.name
            self.logger.info(f"Endpoint created: {endpoint_name}")

            # Save metadata to Firestore
            self.logger.info("Saving metadata to Firestore...")
            self.firestore_manager.batch_save_text_metadata(
                FIRESTORE_COLLECTION,
                metadata_list
            )

            # Insert vectors into index
            self.logger.info("Inserting vectors into index...")
            request = {
                "index": index_name,
                "datapoints": datapoints
            }
            self.index_manager.index_client.upsert_datapoints(request=request)

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
            error_msg = f"Vector Search setup failed after {total_time} seconds: {str(e)}"
            self.logger.error(error_msg)
            raise Exception(error_msg) from e

def load_md_files(md_folder_path: str) -> List[Dict[str, str]]:
    """Load information from MD files

    Args:
        md_folder_path: Path to the folder containing MD files

    Returns:
        List of dictionaries containing file information

    Raises:
        Exception: If file loading fails
    """
    try:
        if not os.path.exists(md_folder_path):
            raise FileNotFoundError(f"MD folder not found: {md_folder_path}")

        md_files_info = []
        for filename in os.listdir(md_folder_path):
            if filename.endswith(".md"):
                file_path = os.path.join(md_folder_path, filename)
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                    md_files_info.append({
                        'filename': filename,
                        'content': content
                    })
                except Exception as e:
                    logging.error(f"Error reading file {filename}: {str(e)}")
                    raise

        if not md_files_info:
            raise ValueError("No MD files found in the specified directory")

        return md_files_info

    except Exception as e:
        error_msg = f"Error loading MD files: {str(e)}"
        logging.error(error_msg)
        raise Exception(error_msg) from e

def setup_logging():
    """Configure logging settings"""
    log_dir = 'app/log'
    os.makedirs(log_dir, exist_ok=True)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_filename = os.path.join(log_dir, f'vector_store_setup_{timestamp}.log')

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_filename, mode='w', encoding='utf-8')
        ]
    )

def main():
    """Main execution function"""
    setup_logging()
    logger = logging.getLogger(__name__)

    try:
        md_folder_path = os.path.join(os.path.dirname(__file__), "md")
        md_files_info = load_md_files(md_folder_path)

        logger.info(f"Found {len(md_files_info)} MD files to process")

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
