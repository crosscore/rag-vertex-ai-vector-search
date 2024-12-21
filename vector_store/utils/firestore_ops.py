# app/vector_store/utils/firestore_ops.py
from google.cloud import firestore
from datetime import datetime
from typing import Dict, Any, List, Optional
import logging
from ...common.config import PROJECT_ID, REGION, FIRESTORE_DATABASE_ID

logger = logging.getLogger(__name__)

class FirestoreManager:
    """Class to manage Firestore data operations"""

    def __init__(self):
        self.project_id = PROJECT_ID
        self.database_id = FIRESTORE_DATABASE_ID
        self.region = REGION
        self._initialize_client()

    def _initialize_client(self) -> None:
        """Initialize Firestore client"""
        try:
            self.db = firestore.Client(
                project=self.project_id,
                database=self.database_id
            )
            logger.info("Firestore client initialized")
        except Exception as e:
            logger.error(f"Firestore initialization error: {str(e)}")
            raise

    def save_text_metadata(self,
                            collection: str,
                            data_point_id: str,
                            filename: str,
                            content: str,
                            additional_metadata: Optional[Dict[str, Any]] = None) -> None:
        """Save text metadata to Firestore"""
        try:
            doc_ref = self.db.collection(collection).document(data_point_id)
            now = datetime.now()

            metadata = {
                "data_point_id": data_point_id,
                "filename": filename,
                "content": content,
                "created_at": now,
                "updated_at": now
            }

            if additional_metadata:
                metadata.update(additional_metadata)

            doc_ref.set(metadata)
            logger.info(f"Metadata saved for {data_point_id}")

        except Exception as e:
            logger.error(f"Metadata save error: {str(e)}")
            raise

    def batch_save_text_metadata(self,
                                collection: str,
                                metadata_list: List[Dict[str, Any]]) -> None:
        """Batch save multiple text metadata"""
        try:
            batch = self.db.batch()
            now = datetime.now()

            for metadata in metadata_list:
                doc_ref = self.db.collection(collection).document(metadata['data_point_id'])
                doc_data = {
                    "data_point_id": metadata['data_point_id'],
                    "filename": metadata['filename'],
                    "content": metadata['content'],
                    "created_at": now,
                    "updated_at": now
                }

                if 'additional_metadata' in metadata:
                    doc_data.update(metadata['additional_metadata'])

                batch.set(doc_ref, doc_data)

            batch.commit()
            logger.info(f"Batch save completed: {len(metadata_list)} items")

        except Exception as e:
            logger.error(f"Batch save error: {str(e)}")
            raise

    def get_text_metadata(self,
                            collection: str,
                            data_point_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve text metadata from Firestore"""
        try:
            doc_ref = self.db.collection(collection).document(data_point_id)
            doc = doc_ref.get()

            if doc.exists:
                logger.info(f"Metadata retrieved for {data_point_id}")
                return doc.to_dict()
            else:
                logger.info(f"Metadata not found for {data_point_id}")
                return None

        except Exception as e:
            logger.error(f"Metadata retrieval error: {str(e)}")
            raise

    def update_text_metadata(self,
                            collection: str,
                            data_point_id: str,
                            updates: Dict[str, Any]) -> None:
        """Update text metadata"""
        try:
            doc_ref = self.db.collection(collection).document(data_point_id)
            updates['updated_at'] = datetime.now()
            doc_ref.update(updates)
            logger.info(f"Metadata updated for {data_point_id}")

        except Exception as e:
            logger.error(f"Metadata update error: {str(e)}")
            raise
