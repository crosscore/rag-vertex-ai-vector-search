# RAG System - Vector Store Setup

## フォルダ構成

```
app/
├── common/                     # 共通コンポーネント
│   ├── init.py
│   ├── config.py               # 設定値の管理
│   └── utils/
│       ├── init.py
│       ├── embeddings.py       # Embedding生成機能
│       └── vector_search.py    # Vector Search基本操作
│
├── rag/                        # RAG機能
│   ├── init.py
│   ├── search.py              # セマンティック検索メイン機能
│   └── utils/
│       ├── init.py
│       └── result_formatter.py # 検索結果フォーマット機能
│
├── vector_store/              # Vector Store設定
│   ├── init.py
│   ├── setup_vector_search.py  # Vector Store設定実行
│   └── utils/
│       ├── init.py
│       ├── firestore_ops.py    # Firestore操作
│       └── index_manager.py    # インデックス管理
│
└── README.md
```

## ファイル概要

### Common Directory

#### config.py

-   プロジェクトの設定値を管理
-   Google Cloud 設定、モデル設定、インデックス設定など

#### utils/embeddings.py

-   テキストの embedding 生成機能
-   トークン検証

#### utils/vector_search.py

-   Vector Search 基本操作の実装
-   検索、データ追加、削除機能
-   クライアント管理

### RAG Directory

#### search.py

-   セマンティック検索のメイン実装
-   複数質問に対する一括検索機能
-   Vector SearchとFirestoreの統合

#### utils/result_formatter.py

-   検索結果の分析と整形
-   統計情報の計算
-   結果のフィルタリングと整形機能

### Vector Store Directory

#### setup_vector_search.py

-   Vector Store 設定のメインスクリプト
-   全体のセットアッププロセスを統合（インデックス作成、デプロイ、データ登録）
-   実行エントリーポイント
-   メタデータをFirestoreに保存し、ベクトルデータをVector Searchに格納

#### utils/firestore_ops.py

-   Firestore データ操作
-   メタデータの保存と取得
-   バッチ処理サポート

#### utils/index_manager.py

-   Vector Search インデックス管理
-   インデックス作成とデプロイ
-   デプロイメント状態監視

## 実行方法

```bash
# Vector Store設定の実行
python -m rag_system.vector_store.setup_vector_search

# セマンティック検索の実行
python -m rag_system.rag.search
```

## 依存関係

```
google-cloud-aiplatform
google-cloud-firestore
vertexai
tiktoken
```
