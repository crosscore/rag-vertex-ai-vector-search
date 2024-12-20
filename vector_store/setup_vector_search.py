# app/vector_store/setup_vector_search.py
"""
Vector Store設定の実行を担当するメインモジュール。
インデックスの作成、Firestoreへのデータ保存、デプロイメントの実行を統合する。
"""
import uuid
import time
from typing import List, Dict, Any
import logging
import os
from ..common.config import (
    PROJECT_ID,
    REGION,
    INDEX_NAME,
    INDEX_ENDPOINT_ID,
    DEPLOYED_INDEX_ID,
    FIRESTORE_COLLECTION
)
from ..common.utils.embeddings import embed_texts
from .utils.firestore_ops import FirestoreManager
from .utils.index_manager import IndexManager

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
        start_time = time.time()
        try:
            logger.info("Vector Search設定を開始")

            # テキストの処理
            process_start = time.time()
            process_result = self.process_texts(texts)
            dimension = process_result['dimension']
            logger.info(f"テキスト処理時間: {int(time.time() - process_start)}秒")

            # インデックスの作成
            logger.info("インデックスの作成を開始")
            index_start = time.time()
            index_op = self.index_manager.create_index(
                display_name=INDEX_NAME,
                dimension=dimension,
                description="RAG system vector search index"
            )
            index_result = self.index_manager.wait_for_operation(index_op)
            logger.info(f"インデックス作成完了: {index_result.name}")
            logger.info(f"インデックス作成時間: {int(time.time() - index_start)}秒")

            # エンドポイントの作成
            logger.info("エンドポイントの作成を開始")
            endpoint_start = time.time()
            endpoint_op = self.index_manager.create_endpoint(
                display_name=INDEX_ENDPOINT_ID,
                description="RAG system vector search endpoint"
            )
            endpoint_result = self.index_manager.wait_for_operation(endpoint_op)
            logger.info(f"エンドポイント作成完了: {endpoint_result.name}")
            logger.info(f"エンドポイント作成時間: {int(time.time() - endpoint_start)}秒")

            # インデックスのデプロイ
            logger.info("インデックスのデプロイを開始")
            deploy_start = time.time()
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

            deploy_time = int(time.time() - deploy_start)
            logger.info("インデックスのデプロイが完了しました")
            logger.info(f"デプロイ所要時間: {deploy_time}秒")
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
                total_time = int(time.time() - start_time)
                logger.info("Vector Search設定が正常に完了しました")
                logger.info(f"総実行時間: {total_time}秒")
                logger.info(f"デプロイ情報:")
                logger.info(f"  デプロイグループ: {state['deployment_group']}")
                logger.info(f"  作成時刻: {state['create_time']}")
                logger.info(f"  同期時刻: {state['index_sync_time']}")
            else:
                logger.error(f"デプロイメントに問題が発生: {state}")
                raise RuntimeError(f"Deployment failed with state: {state['state']}")

        except Exception as e:
            total_time = int(time.time() - start_time)
            logger.error(f"Vector Search設定エラー: {str(e)}")
            logger.error(f"エラーまでの実行時間: {total_time}秒")
            raise

def load_md_files(md_folder_path: str) -> List[Dict[str, str]]:
    """MDファイルから情報を読み込み、辞書のリストとして返す

    Args:
        md_folder_path: MDファイルが格納されているフォルダのパス

    Returns:
        MDファイルの情報を格納した辞書のリスト
        各辞書はファイル名(拡張子なし)をキー、ファイルの内容を値とする
    """
    md_files_info = []
    for filename in os.listdir(md_folder_path):
        if filename.endswith(".md"):
            file_path = os.path.join(md_folder_path, filename)
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            md_files_info.append({filename[:-3]: content})
    return md_files_info

def main():
    """メイン実行関数"""
    # MDファイルが格納されているフォルダのパス
    md_folder_path = os.path.join(os.path.dirname(__file__), "md")

    try:
        # MDファイルから情報を読み込む
        md_files_info = load_md_files(md_folder_path)

        # 各MDファイルの内容をリストにまとめる
        texts = [list(md_info.values())[0] for md_info in md_files_info]

        setup = VectorStoreSetup()
        setup.setup_vector_search(texts)
    except Exception as e:
        logger.error(f"実行エラー: {str(e)}")
        raise

if __name__ == "__main__":
    main()
