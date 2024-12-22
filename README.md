# RAG System - Vector Store Setup

## システム概要

Vertex AI Vector Search とFirestoreを利用した検索システムで、テキストデータに対する効率的なセマンティック検索を実現します。

## フォルダ構成

```
app/
├── common/                     # 共通コンポーネント
│   ├── __init__.py
│   ├── config.py               # 設定値の管理
│   └── utils/
│       ├── __init__.py
│       └── embeddings.py       # Embedding生成機能
│
├── rag/                        # RAG検索機能
│   ├── __init__.py
│   └── search.py               # セマンティック検索メイン機能
│
├── vector_store/               # Vector Store設定
│   ├── __init__.py
│   ├── setup_vector_search.py  # Vector Store設定実行
│   └── utils/
│       ├── __init__.py
│       ├── firestore_ops.py    # Firestore操作
│       └── index_manager.py    # インデックス管理
│
└── README.md
```

## 依存関係

```
google-cloud-aiplatform==1.75.0
google-cloud-firestore==2.19.0
pyarrow==18.1.0
tiktoken==0.8.0
```

## 構築済みindexEndpoint情報
```
(gcp-venv) PS C:\Users\645777\repos> gcloud ai indexes list --region=asia-northeast1
Using endpoint [https://asia-northeast1-aiplatform.googleapis.com/]
---
createTime: '2024-12-22T18:37:03.055946Z'
deployedIndexes:
- deployedIndexId: table_metadata_index_deployed
  displayName: Deployed index table_metadata_index_deployed
  indexEndpoint: projects/990681154812/locations/asia-northeast1/indexEndpoints/3778766377968467968
description: RAG system vector search index
displayName: table_metadata_index
encryptionSpec: {}
etag: AMEw9yOtMVtltDIoR_cB05NNvrIlK0JQErZA3xn3He9l7PX_45YECNhBO0GnfOXYZJA=
indexStats:
  shardsCount: 1
  vectorsCount: '7'
indexUpdateMethod: STREAM_UPDATE
metadata:
  config:
    algorithmConfig:
      treeAhConfig:
        leafNodeEmbeddingCount: '500'
        leafNodesToSearchPercent: 10
    approximateNeighborsCount: 150
    dimensions: 768
    distanceMeasureType: DOT_PRODUCT_DISTANCE
    shardSize: SHARD_SIZE_SMALL
metadataSchemaUri: gs://google-cloud-aiplatform/schema/matchingengine/metadata/nearest_neighbor_search_1.0.0.yaml
name: projects/990681154812/locations/asia-northeast1/indexes/3231702168545263616
updateTime: '2024-12-22T19:32:27.098320Z'
```

## gcloud config list
```
(gcp-venv) PS C:\Users\645777\repos> gcloud config list
[accessibility]
screen_reader = False
[compute]
region = asia-northeast1
zone = asia-northeast1-a
[core]
account = g-ysakahara@ga.taknet.co.jp
custom_ca_certs_file = C:\Zscaler_root.pem
disable_usage_reporting = False
project = business-test-001

Your active configuration is: [default]
```

## pip list
```
(gcp-venv) PS C:\Users\645777\repos> pip list
Package                       Version
----------------------------- -----------
annotated-types               0.7.0
cachetools                    5.5.0
certifi                       2024.12.14
charset-normalizer            3.4.0
contourpy                     1.3.1
cycler                        0.12.1
docstring_parser              0.16
fonttools                     4.55.3
google-api-core               2.24.0
google-auth                   2.37.0
google-cloud-aiplatform       1.75.0
google-cloud-bigquery         3.27.0
google-cloud-core             2.4.1
google-cloud-firestore        2.19.0
google-cloud-resource-manager 1.14.0
google-cloud-storage          2.19.0
google-crc32c                 1.6.0
google-resumable-media        2.7.2
googleapis-common-protos      1.66.0
grpc-google-iam-v1            0.13.1
grpcio                        1.68.1
grpcio-status                 1.68.1
idna                          3.10
kiwisolver                    1.4.7
matplotlib                    3.10.0
numpy                         2.2.1
packaging                     24.2
pandas                        2.2.3
pillow                        11.0.0
pip                           24.3.1
proto-plus                    1.25.0
protobuf                      5.29.2
pyarrow                       18.1.0
pyasn1                        0.6.1
pyasn1_modules                0.4.1
pydantic                      2.10.4
pydantic_core                 2.27.2
pyparsing                     3.2.0
python-dateutil               2.9.0.post0
pytz                          2024.2
regex                         2024.11.6
requests                      2.32.3
rsa                           4.9
shapely                       2.0.6
six                           1.17.0
tiktoken                      0.8.0
typing_extensions             4.12.2
tzdata                        2024.2
urllib3                       2.3.0
```
