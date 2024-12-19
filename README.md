# RAG System - Vector Store Setup

## フォルダ構成
```
rag_system/
├── common/                     # 共通コンポーネント
│   ├── init.py
│   ├── config.py               # 設定値の管理
│   └── utils/
│       ├── init.py
│       ├── embeddings.py       # Embedding生成機能
│       └── vector_search.py    # Vector Search基本操作
│
├── vector_store/               # Vector Store設定
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
- プロジェクトの設定値を管理
- Google Cloud設定、モデル設定、インデックス設定など

#### utils/embeddings.py
- テキストのembedding生成機能
- トークン検証
- 次元数の取得

#### utils/vector_search.py
- Vector Search基本操作の実装
- 検索、データ追加、削除機能
- クライアント管理

### Vector Store Directory

#### setup_vector_search.py
- Vector Store設定のメインスクリプト
- 全体のセットアッププロセスを統合
- 実行エントリーポイント

#### utils/firestore_ops.py
- Firestoreデータ操作
- メタデータの保存と取得
- バッチ処理サポート

#### utils/index_manager.py
- Vector Searchインデックス管理
- インデックス作成とデプロイ
- デプロイメント状態監視

## 実行方法

```bash
# Vector Store設定の実行
python -m rag_system.vector_store.setup_vector_search
```

## 依存関係
```
google-cloud-aiplatform
google-cloud-firestore
vertexai
tiktoken
```
