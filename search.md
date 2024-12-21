
# app/common/utils/vector_search.py で定義された関数の使用例

```
# エンドポイントの初期化
endpoint = get_matching_engine_endpoint(
    project_id="your-project",
    location="us-central1",
    endpoint_name="your-endpoint-name"
)

# 類似検索の実行
results = get_neighbors(
    endpoint=endpoint,
    deployed_index_id="your-deployed-index",
    queries=[[1.0, 2.0, 3.0]],
    num_neighbors=5
)

# 特定のデータポイントの取得
datapoints = get_datapoints(
    endpoint=endpoint,
    deployed_index_id="your-deployed-index",
    datapoint_ids=["id1", "id2"]
)
```
