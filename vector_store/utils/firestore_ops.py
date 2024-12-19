# rag_system/vector_store/utils/firestore_ops.py

"""
Firestoreのデータ操作を管理するモジュール。
メタデータの保存、取得、更新などの操作を提供する。
"""
from google.cloud import firestore
from google.api_core.exceptions import GoogleAPIError
from datetime import datetime
from typing import Dict, Any, List, Optional
import logging
from ...common.config import PROJECT_ID

logger = logging.getLogger(__name__)

class FirestoreManager:
    """Firestoreデータ操作を管理するクラス"""

    def __init__(self, project_id: str = PROJECT_ID):
        """
        Args:
            project_id: Google CloudプロジェクトID
        """
        try:
            self.db = firestore.Client(project=project_id)
            logger.info(f"Firestoreクライアントの初期化完了: {project_id}")
        except Exception as e:
            logger.error(f"Firestoreクライアント初期化エラー: {str(e)}")
            raise

    def save_text_metadata(self,
                            collection: str,
                            data_point_id: str,
                            text: str,
                            additional_metadata: Optional[Dict[str, Any]] = None) -> None:
        """テキストのメタデータをFirestoreに保存

        Args:
            collection: コレクション名
            data_point_id: データポイントID
            text: 保存するテキスト
            additional_metadata: 追加のメタデータ（オプション）

        Raises:
            GoogleAPIError: Firestore操作に失敗した場合
        """
        try:
            doc_ref = self.db.collection(collection).document(data_point_id)

            # 基本メタデータの準備
            metadata = {
                "data_point_id": data_point_id,
                "text": text,
                "created_at": datetime.now(),
                "updated_at": datetime.now()
            }

            # 追加メタデータがある場合は統合
            if additional_metadata:
                metadata.update(additional_metadata)

            # データを保存
            doc_ref.set(metadata)
            logger.info(f"メタデータを保存しました: collection={collection}, id={data_point_id}")

        except GoogleAPIError as e:
            logger.error(f"メタデータ保存エラー: {str(e)}")
            raise

    def get_text_metadata(self,
                            collection: str,
                            data_point_id: str) -> Optional[Dict[str, Any]]:
        """テキストのメタデータをFirestoreから取得

        Args:
            collection: コレクション名
            data_point_id: データポイントID

        Returns:
            メタデータの辞書。存在しない場合はNone

        Raises:
            GoogleAPIError: Firestore操作に失敗した場合
        """
        try:
            doc_ref = self.db.collection(collection).document(data_point_id)
            doc = doc_ref.get()

            if doc.exists:
                logger.info(f"メタデータを取得しました: collection={collection}, id={data_point_id}")
                return doc.to_dict()
            else:
                logger.warning(f"メタデータが見つかりません: collection={collection}, id={data_point_id}")
                return None

        except GoogleAPIError as e:
            logger.error(f"メタデータ取得エラー: {str(e)}")
            raise

    def batch_save_text_metadata(self,
                                collection: str,
                                metadata_list: List[Dict[str, Any]]) -> None:
        """複数のテキストメタデータをバッチ保存

        Args:
            collection: コレクション名
            metadata_list: メタデータのリスト。各要素は以下のキーを含む必要がある:
                            - data_point_id: データポイントID
                            - text: テキスト
                            - additional_metadata: 追加のメタデータ（オプション）

        Raises:
            GoogleAPIError: Firestore操作に失敗した場合
        """
        try:
            batch = self.db.batch()
            now = datetime.now()

            for metadata in metadata_list:
                doc_ref = self.db.collection(collection).document(metadata['data_point_id'])

                # 基本メタデータの準備
                doc_data = {
                    "data_point_id": metadata['data_point_id'],
                    "text": metadata['text'],
                    "created_at": now,
                    "updated_at": now
                }

                # 追加メタデータがある場合は統合
                if 'additional_metadata' in metadata:
                    doc_data.update(metadata['additional_metadata'])

                batch.set(doc_ref, doc_data)

            # バッチ書き込みを実行
            batch.commit()
            logger.info(f"バッチ保存完了: {len(metadata_list)}件")

        except GoogleAPIError as e:
            logger.error(f"バッチ保存エラー: {str(e)}")
            raise

    def update_text_metadata(self,
                            collection: str,
                            data_point_id: str,
                            updates: Dict[str, Any]) -> None:
        """テキストのメタデータを更新

        Args:
            collection: コレクション名
            data_point_id: データポイントID
            updates: 更新するフィールドと値の辞書

        Raises:
            GoogleAPIError: Firestore操作に失敗した場合
        """
        try:
            doc_ref = self.db.collection(collection).document(data_point_id)

            # 更新日時を自動的に追加
            updates['updated_at'] = datetime.now()

            doc_ref.update(updates)
            logger.info(f"メタデータを更新しました: collection={collection}, id={data_point_id}")

        except GoogleAPIError as e:
            logger.error(f"メタデータ更新エラー: {str(e)}")
            raise
