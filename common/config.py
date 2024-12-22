# app/common/config.py
"""
A module that manages common settings for the RAG system.
Basic settings such as project ID, region, and model name.
"""
# Google Cloud Project settings
PROJECT_ID = "990681154812"
REGION = "asia-northeast1"

# Vector Search Resource IDs
INDEX_RESOURCE_ID = "6352696710313017344"
ENDPOINT_RESOURCE_ID = "4314694733625556992"
DEPLOYED_INDEX_ID = "table_metadata_index_deployed"

# Vector Search Display Names (for human readability)
INDEX_DISPLAY_NAME = "table_metadata_index"
ENDPOINT_DISPLAY_NAME = "table_metadata_index_endpoint"

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
DEPLOYMENT_TIMEOUT_MINUTES = 45  # minutes
DEPLOYMENT_CHECK_INTERVAL = 1  # seconds

# Token limits
MAX_TOKENS_PER_TEXT = 2042

# Embedding configuration
MAX_RETRY_ATTEMPTS = 3
RETRY_DELAY_SECONDS = 1
EMBEDDING_BATCH_SIZE = 10
