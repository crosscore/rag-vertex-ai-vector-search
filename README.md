# RAG System - Vector Store Setup

## システム概要

このシステムは、Vertex AI Vector Search とFirestoreを利用した検索システムで、テキストデータに対する効率的なセマンティック検索を実現します。

## フォルダ構成

```
app/
├── common/                     # 共通コンポーネント
│   ├── __init__.py
│   ├── config.py               # 設定値の管理
│   └── utils/
│       ├── __init__.py
│       ├── embeddings.py       # Embedding生成機能
│       ├── vector_search.py    # Vector Search操作
│       └── result_formatter.py # 結果整形・分析機能
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

## コンポーネントの説明

### Common Directory

#### config.py
- プロジェクトの設定値を一元管理
- 環境変数とデフォルト値の設定
- 各種設定のバリデーション機能

#### utils/embeddings.py
- テキストのembedding生成
- トークン検証と管理
- 並列処理によるバッチ処理
- カスタム例外処理
- 詳細なログ出力

主要クラス:
- `TextTokenizer`: トークン化と検証
- `EmbeddingGenerator`: embedding生成の管理
- `EmbeddingConfig`: 設定管理

#### utils/vector_search.py
- Vector Search基本操作の実装
- クライアント管理と初期化
- 検索操作の実行
- データポイント管理

主要クラス:
- `VectorSearchClient`: 検索操作の管理
- `SearchConfiguration`: 検索設定の管理

#### utils/result_formatter.py
- 検索結果の分析と整形
- 統計情報の計算
- 結果のフィルタリングと整形

主要クラス:
- `ResultFormatter`: 結果整形の管理
- `ResultAnalyzer`: 統計分析の実行
- `StatisticalSummary`: 統計情報の構造化

### RAG Directory

#### search.py
- セマンティック検索のメイン実装
- 検索パラメータの管理
- 結果処理のワークフロー管理

主要クラス:
- `SemanticSearcher`: 検索処理の統合管理
- `SearchParameters`: 検索パラメータの設定

### Vector Store Directory

#### setup_vector_search.py
- Vector Store環境の構築
- インデックス作成とデプロイ
- データの初期登録
- メタデータ管理

#### utils/firestore_ops.py
- Firestoreデータ操作
- メタデータの管理
- バッチ処理の実装

#### utils/index_manager.py
- Vector Searchインデックス管理
- デプロイメント制御
- 状態監視

## 使用方法

### Vector Store設定

```bash
# Vector Store環境の構築
python -m app.vector_store.setup_vector_search

# 設定の確認
python -m app.vector_store.setup_vector_search --check
```

### 検索実行

```python
from app.rag.search import SemanticSearcher, SearchParameters

# 検索パラメータの設定
params = SearchParameters(
    num_results=10,
    min_similarity_score=0.5,
    include_metadata=True,
    compute_statistics=True
)

# 検索の実行
searcher = SemanticSearcher()
results = searcher.search(
    questions=["検索したいクエリ"],
    params=params
)

# 結果の確認
print(results['summary'])
print(results['statistics'])
```

## エラーハンドリング

システムは以下の例外クラスを提供します：

- `EmbeddingError`: Embedding生成関連のエラー
  - `TokenizationError`: トークン化エラー
  - `EmbeddingGenerationError`: 生成エラー
  - `BatchProcessingError`: バッチ処理エラー

- `VectorSearchError`: Vector Search関連のエラー
  - `EndpointInitializationError`: 初期化エラー
  - `SearchOperationError`: 検索操作エラー
  - `DatapointOperationError`: データポイント操作エラー

- `SearchError`: 検索全般のエラー

## ログ出力

ログレベルは環境変数で制御可能:
```bash
export LOG_LEVEL="DEBUG"  # INFO, WARNING, ERROR なども指定可能
```

## 依存関係

```
google-cloud-aiplatform
google-cloud-firestore
vertexai
tiktoken
python-dotenv
```
