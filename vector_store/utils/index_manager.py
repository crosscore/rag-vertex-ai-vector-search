# rag_system/vector_store/utils/index_manager.py

"""
Vector Search インデックスの管理を担当するモジュール。
インデックスの作成、デプロイ、監視などの機能を提供する。
"""

from google.cloud import aiplatform
from google.cloud.aiplatform_v1 import (
    IndexServiceClient,
    IndexEndpointServiceClient,
    Index,
    IndexEndpoint
)
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

# ロガーの設定
logger = logging.getLogger(__name__)

class IndexManager:
    """Vector Search インデックスの管理を行うクラス"""

    def __init__(self, project_id: str = PROJECT_ID, region: str = REGION):
        """
        Args:
            project_id: プロジェクトID
            region: リージョン
        """
        self.project_id = project_id
        self.region = region
        self.parent = f"projects/{project_id}/locations/{region}"

        # APIクライアントの初期化
        client_options = {"api_endpoint": f"{region}-aiplatform.googleapis.com"}
        self.index_client = IndexServiceClient(client_options=client_options)
        self.endpoint_client = IndexEndpointServiceClient(client_options=client_options)

        # Vertex AI の初期化
        aiplatform.init(project=project_id, location=region)

    def create_index(self,
                    display_name: str,
                    dimension: int,
                    description: Optional[str] = None) -> Operation:
        """新しいインデックスを作成する

        Args:
            display_name: インデックスの表示名
            dimension: ベクトルの次元数
            description: インデックスの説明（オプション）

        Returns:
            作成操作のOperation

        Raises:
            GoogleAPIError: インデックス作成に失敗した場合
        """
        try:
            # インデックス設定の準備
            config = INDEX_CONFIG.copy()
            config['dimensions'] = dimension

            index = Index(
                display_name=display_name,
                description=description or f"Vector search index created at {time.strftime('%Y-%m-%d %H:%M:%S')}",
                metadata_schema_uri="gs://google-cloud-aiplatform/schema/matchingengine/metadata/nearest_neighbor_search_1.0.0.yaml",
                metadata={"config": config}
            )

            # インデックス作成操作の実行
            operation = self.index_client.create_index(
                parent=self.parent,
                index=index
            )

            logger.info(f"インデックス作成を開始しました: {display_name}")
            return operation

        except GoogleAPIError as e:
            logger.error(f"インデックス作成エラー: {str(e)}")
            raise

    def create_endpoint(self,
                        display_name: str,
                        description: Optional[str] = None) -> Operation:
        """新しいエンドポイントを作成する

        Args:
            display_name: エンドポイントの表示名
            description: エンドポイントの説明（オプション）

        Returns:
            作成操作のOperation

        Raises:
            GoogleAPIError: エンドポイント作成に失敗した場合
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

            logger.info(f"エンドポイント作成を開始しました: {display_name}")
            return operation

        except GoogleAPIError as e:
            logger.error(f"エンドポイント作成エラー: {str(e)}")
            raise

    def deploy_index(self,
                    index_name: str,
                    endpoint_name: str,
                    deployed_index_id: str) -> Operation:
        """インデックスをエンドポイントにデプロイする

        Args:
            index_name: デプロイするインデックスの名前
            endpoint_name: デプロイ先のエンドポイント名
            deployed_index_id: デプロイ済みインデックスのID

        Returns:
            デプロイ操作のOperation

        Raises:
            GoogleAPIError: デプロイに失敗した場合
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
            logger.info(f"インデックスのデプロイを開始しました: {deployed_index_id}")
            return operation

        except GoogleAPIError as e:
            logger.error(f"インデックスデプロイエラー: {str(e)}")
            raise

    def wait_for_operation(self,
                            operation: Operation,
                            timeout_minutes: int = DEPLOYMENT_TIMEOUT_MINUTES) -> Any:
        """操作の完了を待機する

        Args:
            operation: 待機する操作
            timeout_minutes: タイムアウトまでの分数

        Returns:
            操作の結果

        Raises:
            TimeoutError: 指定時間内に操作が完了しなかった場合
            GoogleAPIError: 操作が失敗した場合
        """
        try:
            start_time = time.time()
            while True:
                if operation.done():
                    logger.info("操作が完了しました")
                    return operation.result()

                if time.time() - start_time > timeout_minutes * 60:
                    raise TimeoutError(f"操作がタイムアウトしました: {timeout_minutes}分")

                logger.debug("操作の完了を待機中...")
                time.sleep(DEPLOYMENT_CHECK_INTERVAL)

        except GoogleAPIError as e:
            logger.error(f"操作待機中にエラーが発生: {str(e)}")
            raise

    def get_deployment_state(self,
                            endpoint_name: str,
                            deployed_index_id: str) -> Dict[str, Any]:
        """デプロイの状態を取得する

        Args:
            endpoint_name: エンドポイント名
            deployed_index_id: デプロイ済みインデックスのID

        Returns:
            デプロイメントの状態情報を含む辞書

        Raises:
            GoogleAPIError: 状態取得に失敗した場合
        """
        try:
            endpoint = self.endpoint_client.get_index_endpoint(name=endpoint_name)

            for deployed_index in endpoint.deployed_indexes:
                if deployed_index.id == deployed_index_id:
                    state = {
                        "state": deployed_index.deployment_state.state,
                        "error_msg": deployed_index.deployment_state.error_message,
                        "create_time": deployed_index.create_time,
                    }
                    logger.info(f"デプロイメント状態: {state}")
                    return state

            logger.warning(f"デプロイされたインデックスが見つかりません: {deployed_index_id}")
            return {"state": "NOT_FOUND"}

        except GoogleAPIError as e:
            logger.error(f"デプロイメント状態取得エラー: {str(e)}")
            raise
