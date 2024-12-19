# app/common/utils/vector_search.py

"""
Vector Search操作の基本機能を提供するモジュール。
検索、データ追加、削除などの基本操作を実装。
"""

from google.cloud import aiplatform
from google.api_core.exceptions import GoogleAPIError
from typing import List, Dict, Any, Optional
import logging
from ...common.utils.embeddings import embed_texts
from ...common.config import EMBEDDING_MODEL

logger = logging.getLogger(__name__)

class VectorSearchClient:
    """Vector Search操作を管理するクライアントクラス"""

    def __init__(self,
                    project_id: str,
                    location: str,
                    index_endpoint: aiplatform.MatchingEngineIndexEndpoint):
        """
        Args:
            project_id: プロジェクトID
            location: リージョン
            index_endpoint: 使用するインデックスエンドポイント
        """
        self.project_id = project_id
        self.location = location
        self.index_endpoint = index_endpoint
        aiplatform.init(project=project_id, location=location)

    def upsert_data_points(self,
                            deployed_index_id: str,
                            data_points: List[Dict[str, Any]]) -> None:
        """データポイントをインデックスに追加または更新する

        Args:
            deployed_index_id: デプロイ済みインデックスのID
            data_points: 追加・更新するデータポイントのリスト
                        各データポイントは {'id': str, 'embedding': List[float]} の形式

        Raises:
            GoogleAPIError: API呼び出しに失敗した場合
        """
        try:
            logger.info(f"{len(data_points)}件のデータポイントの更新を開始")
            self.index_endpoint.upsert_datapoints(
                deployed_index_id=deployed_index_id,
                datapoints=data_points
            )
            logger.info("データポイントの更新が完了しました")
        except GoogleAPIError as e:
            logger.error(f"データポイント更新エラー: {str(e)}")
            raise

    def remove_data_points(self,
                            deployed_index_id: str,
                            ids: List[str]) -> None:
        """データポイントをインデックスから削除する

        Args:
            deployed_index_id: デプロイ済みインデックスのID
            ids: 削除するデータポイントのIDリスト

        Raises:
            GoogleAPIError: API呼び出しに失敗した場合
        """
        try:
            logger.info(f"{len(ids)}件のデータポイントの削除を開始")
            self.index_endpoint.remove_datapoints(
                deployed_index_id=deployed_index_id,
                datapoint_ids=ids
            )
            logger.info("データポイントの削除が完了しました")
        except GoogleAPIError as e:
            logger.error(f"データポイント削除エラー: {str(e)}")
            raise

    def search(self,
                deployed_index_id: str,
                query: str,
                num_neighbors: int = 5,
                filter_expr: Optional[str] = None) -> List[Dict[str, Any]]:
        """クエリに類似したデータポイントを検索する

        Args:
            deployed_index_id: デプロイ済みインデックスのID
            query: 検索クエリ文字列
            num_neighbors: 取得する近傍点の数
            filter_expr: フィルタ式（オプション）

        Returns:
            検索結果のリスト。各要素は以下の形式:
            {
                'id': データポイントID,
                'distance': 類似度スコア,
                'neighbor_count': 近傍点数
            }

        Raises:
            GoogleAPIError: API呼び出しに失敗した場合
        """
        try:
            # クエリのembedding生成
            query_embedding = embed_texts([query], EMBEDDING_MODEL)[0]
            logger.info(f"検索クエリのembedding生成完了: {len(query_embedding)}次元")

            # 近傍検索実行
            response = self.index_endpoint.find_neighbors(
                deployed_index_id=deployed_index_id,
                queries=[query_embedding],
                num_neighbors=num_neighbors,
                filter=filter_expr
            )

            # レスポンスの整形
            results = []
            for neighbor in response[0]:
                results.append({
                    'id': neighbor.id,
                    'distance': neighbor.distance,
                    'neighbor_count': len(response[0])
                })

            logger.info(f"検索完了: {len(results)}件の結果")
            return results

        except GoogleAPIError as e:
            logger.error(f"検索実行エラー: {str(e)}")
            raise
        except Exception as e:
            logger.error(f"予期せぬエラー: {str(e)}")
            raise

    def get_index_stats(self, deployed_index_id: str) -> Dict[str, Any]:
        """インデックスの統計情報を取得する

        Args:
            deployed_index_id: デプロイ済みインデックスのID

        Returns:
            統計情報を含む辞書

        Raises:
            GoogleAPIError: API呼び出しに失敗した場合
        """
        try:
            stats = self.index_endpoint.get_index_stats(deployed_index_id)
            logger.info(f"インデックス統計情報取得完了: {deployed_index_id}")
            return {
                'total_data_points': stats.total_data_points,
                'updated_at': stats.updated_at,
                'deployed_index_id': deployed_index_id
            }
        except GoogleAPIError as e:
            logger.error(f"統計情報取得エラー: {str(e)}")
            raise
