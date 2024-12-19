# rag_system/common/utils/embeddings.py

"""
テキストのembedding生成を担当するモジュール。
トークン検証とembedding生成の機能を提供する。
"""

from vertexai.language_models import TextEmbeddingModel
import tiktoken
from typing import List
import logging
from ...common.config import (
    EMBEDDING_MODEL,
    MAX_TOKENS_PER_TEXT,
    MAX_TOTAL_TOKENS
)

# ロガーの設定
logger = logging.getLogger(__name__)

def validate_token_count(texts: List[str]) -> None:
    """入力テキストのトークン数を検証する

    Args:
        texts: 検証対象のテキストリスト

    Raises:
        ValueError: トークン数が制限を超えている場合
    """
    encoding = tiktoken.get_encoding("cl100k_base")

    total_tokens = 0
    for text in texts:
        num_tokens = len(encoding.encode(text))
        if num_tokens > MAX_TOKENS_PER_TEXT:
            raise ValueError(
                f"テキストのトークン数が制限を超えています。制限: {MAX_TOKENS_PER_TEXT}, "
                f"実際: {num_tokens}, テキスト先頭: '{text[:50]}...'"
            )
        total_tokens += num_tokens

    if total_tokens > MAX_TOTAL_TOKENS:
        raise ValueError(
            f"全テキストの合計トークン数が制限を超えています。制限: {MAX_TOTAL_TOKENS}, "
            f"実際: {total_tokens}"
        )

def embed_single_text(text: str, model: TextEmbeddingModel) -> List[float]:
    """単一のテキストをembeddingに変換する

    Args:
        text: 変換対象のテキスト
        model: 使用するembeddingモデル

    Returns:
        embedding値のリスト

    Raises:
        Exception: embedding生成に失敗した場合
    """
    try:
        embedding = model.get_embeddings([text])[0]
        return embedding.values
    except Exception as e:
        logger.error(f"Embedding生成エラー - テキスト: '{text[:50]}...': {str(e)}")
        raise

def embed_texts(texts: List[str], model_name: str = EMBEDDING_MODEL) -> List[List[float]]:
    """複数のテキストをembeddingに変換する

    Args:
        texts: 変換対象のテキストリスト
        model_name: 使用するembeddingモデルの名前

    Returns:
        embedding値のリストのリスト

    Raises:
        ValueError: トークン数の検証に失敗した場合
        Exception: embedding生成に失敗した場合
    """
    try:
        # トークン数の検証
        validate_token_count(texts)

        # モデルの初期化
        model = TextEmbeddingModel.from_pretrained(model_name)

        # バッチ処理でembedding生成
        logger.info(f"{len(texts)}件のテキストのembedding生成を開始")
        embeddings = model.get_embeddings(texts)

        # 結果を変換
        result = [embedding.values for embedding in embeddings]
        logger.info(f"embedding生成完了: {len(result)}件")

        return result

    except ValueError as ve:
        logger.error(f"トークン数検証エラー: {str(ve)}")
        raise
    except Exception as e:
        logger.error(f"Embedding生成プロセスエラー: {str(e)}")
        raise

def get_embedding_dimension(model_name: str = EMBEDDING_MODEL) -> int:
    """指定されたモデルのembeddingの次元数を取得する

    Args:
        model_name: embeddingモデルの名前

    Returns:
        embedding次元数

    Note:
        この関数は、モデルのサンプル出力から次元数を動的に取得します
    """
    model = TextEmbeddingModel.from_pretrained(model_name)
    sample_embedding = model.get_embeddings(["test"])[0]
    return len(sample_embedding.values)
