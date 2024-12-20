# app/vector_store/setup_vector_search.py

"""
Vector Store設定の実行を担当するメインモジュール。
インデックスの作成、Firestoreへのデータ保存、デプロイメントの実行を統合する。
"""

import uuid
from typing import List, Dict, Any
import logging
from ..common.config import (
    PROJECT_ID,
    REGION,
    INDEX_NAME,
    INDEX_ENDPOINT_ID,
    DEPLOYED_INDEX_ID,
    FIRESTORE_COLLECTION
)
from ..common.utils.embeddings import embed_texts, get_embedding_dimension
from .utils.firestore_ops import FirestoreManager
from .utils.index_manager import IndexManager

# ロガーの設定
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class VectorStoreSetup:
    """Vector Store設定の実行を管理するクラス"""

    def __init__(self):
        """必要なマネージャーとクライアントを初期化"""
        self.firestore_manager = FirestoreManager(PROJECT_ID)
        self.index_manager = IndexManager(PROJECT_ID, REGION)

    def process_texts(self,
                        texts: List[str]) -> Dict[str, Any]:
        """テキストの処理とメタデータの保存を実行

        Args:
            texts: 処理対象のテキストリスト

        Returns:
            処理結果を含む辞書

        Raises:
            Exception: 処理中にエラーが発生した場合
        """
        try:
            # データポイントIDの生成
            data_point_ids = [str(uuid.uuid4()) for _ in texts]

            # Embeddingの生成
            logger.info("Embeddingの生成を開始")
            embeddings = embed_texts(texts)
            embedding_dimension = len(embeddings[0])
            logger.info(f"Embedding生成完了: {len(embeddings)}件, 次元数: {embedding_dimension}")

            # Firestoreへのメタデータ保存
            logger.info("Firestoreへのメタデータ保存を開始")
            metadata_list = [
                {
                    'data_point_id': data_point_id,
                    'text': text,
                    'additional_metadata': {
                        'embedding_dimension': embedding_dimension
                    }
                }
                for data_point_id, text in zip(data_point_ids, texts)
            ]
            self.firestore_manager.batch_save_text_metadata(
                FIRESTORE_COLLECTION,
                metadata_list
            )
            logger.info("メタデータ保存完了")

            return {
                'data_point_ids': data_point_ids,
                'embeddings': embeddings,
                'dimension': embedding_dimension
            }

        except Exception as e:
            logger.error(f"テキスト処理エラー: {str(e)}")
            raise

    def setup_vector_search(self,
                            texts: List[str]) -> None:
        """Vector Search環境の設定を実行

        Args:
            texts: 初期データとして使用するテキストリスト

        Returns:
            None

        Raises:
            Exception: セットアップ中にエラーが発生した場合
        """
        try:
            logger.info("Vector Search設定を開始")

            # テキストの処理
            process_result = self.process_texts(texts)
            dimension = process_result['dimension']

            # インデックスの作成
            logger.info("インデックスの作成を開始")
            index_op = self.index_manager.create_index(
                display_name=INDEX_NAME,
                dimension=dimension,
                description="RAG system vector search index"
            )
            index_result = self.index_manager.wait_for_operation(index_op)
            logger.info(f"インデックス作成完了: {index_result.name}")

            # エンドポイントの作成
            logger.info("エンドポイントの作成を開始")
            endpoint_op = self.index_manager.create_endpoint(
                display_name=INDEX_ENDPOINT_ID,
                description="RAG system vector search endpoint"
            )
            endpoint_result = self.index_manager.wait_for_operation(endpoint_op)
            logger.info(f"エンドポイント作成完了: {endpoint_result.name}")

            # インデックスのデプロイ
            logger.info("インデックスのデプロイを開始")
            deploy_op = self.index_manager.deploy_index(
                index_name=index_result.name,
                endpoint_name=endpoint_result.name,
                deployed_index_id=DEPLOYED_INDEX_ID
            )
            self.index_manager.wait_for_operation(deploy_op)

            # デプロイ後のエンドポイント情報を取得
            endpoint_info = self.index_manager.endpoint_client.get_index_endpoint(
                name=endpoint_result.name
            )

            logger.info("インデックスのデプロイが完了しました")
            logger.info(f"パブリックエンドポイント: {endpoint_info.public_endpoint_domain_name}")

            # デプロイされたインデックスの情報をログ出力
            for deployed_index in endpoint_info.deployed_indexes:
                if deployed_index.id == DEPLOYED_INDEX_ID:
                    logger.info(f"デプロイ済みインデックス情報:")
                    logger.info(f"  ID: {deployed_index.id}")
                    logger.info(f"  作成時刻: {deployed_index.create_time}")
                    logger.info(f"  インデックスパス: {deployed_index.index}")
                    break

            # デプロイメント状態の確認
            state = self.index_manager.get_deployment_state(
                endpoint_result.name,
                DEPLOYED_INDEX_ID
            )

            if state['state'] == "DEPLOYED":
                logger.info("Vector Search設定が正常に完了しました")
            else:
                logger.error(f"デプロイメントに問題が発生: {state}")
                raise RuntimeError(f"Deployment failed with state: {state['state']}")

        except Exception as e:
            logger.error(f"Vector Search設定エラー: {str(e)}")
            raise

def main():
    """メイン実行関数"""
    # サンプルテキスト
    table_texts = [
        "test_table: テストデータ保存用のテーブルです。",
        "outliers_table: 外れ値データを含むテーブルです。",
    ]

    try:
        setup = VectorStoreSetup()
        setup.setup_vector_search(table_texts)
    except Exception as e:
        logger.error(f"実行エラー: {str(e)}")
        raise

if __name__ == "__main__":
    main()
