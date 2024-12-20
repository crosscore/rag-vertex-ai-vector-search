# app/common/config.py
"""
RAGシステムの共通設定値を管理するモジュール。
プロジェクトID、リージョン、モデル名などの基本設定。
"""
# Google Cloud Project settings
PROJECT_ID = "business-test-001"
REGION = "asia-northeast1"

# Vector Search settings
INDEX_NAME = "table_metadata_index"
INDEX_ENDPOINT_ID = "table_metadata_index_endpoint"
DEPLOYED_INDEX_ID = "table_metadata_index_deployed"

# Model settings
EMBEDDING_MODEL = "text-multilingual-embedding-002"

# Firestore settings
FIRESTORE_DATABASE_ID = "database-test-001"
FIRESTORE_COLLECTION = "table_metadata"

# BigQuery settings
DATASET_ID = "test_dataset"

# Index configuration
INDEX_CONFIG = {
    "dimensions": None,  # Will be set dynamically based on embedding size
    "approximate_neighbors_count": 150,
    "distance_measure_type": "DOT_PRODUCT_DISTANCE",
    "algorithm_config": {
        "tree_ah_config": {
            "leaf_node_embedding_count": 500,
            "leaf_nodes_to_search_percent": 10
        }
    },
    "shard_size": "SHARD_SIZE_SMALL"
}

# Deployment configuration
DEPLOYMENT_CONFIG = {
    "machine_spec": {
        "machine_type": "e2-standard-2"
    },
    "min_replica_count": 1,
    "max_replica_count": 1
}

# Timeout settings
DEPLOYMENT_TIMEOUT_MINUTES = 15
DEPLOYMENT_CHECK_INTERVAL = 30  # seconds

# Token limits
MAX_TOKENS_PER_TEXT = 2042
MAX_TOTAL_TOKENS = 3072
