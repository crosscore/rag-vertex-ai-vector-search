# app/common/utils/embeddings.py
"""
テキストのembedding生成を担当するモジュール。
トークン検証とembedding生成の機能を提供する。
"""
from vertexai.language_models import TextEmbeddingModel
import tiktoken
from typing import List, Dict
import logging
from ...common.config import (
    EMBEDDING_MODEL,
    MAX_TOKENS_PER_TEXT,
)

logger = logging.getLogger(__name__)

def validate_token_count_per_text(text_info_list: List[Dict[str, str]]) -> None:
    """入力テキストのトークン数を検証する (各テキスト単位)

    Args:
        text_info_list: 検証対象のテキスト情報リスト。各要素は {'filename': 'ファイル名', 'content': 'テキスト内容'} の辞書。

    Raises:
        ValueError: トークン数が制限を超えている場合
    """
    encoding = tiktoken.get_encoding("cl100k_base")

    total_tokens = 0
    for text_info in text_info_list:
        filename = text_info['filename']
        text = text_info['content']
        num_tokens = len(encoding.encode(text))
        logger.info(f"{filename} のトークン数: {num_tokens}")
        if num_tokens > MAX_TOKENS_PER_TEXT:
            raise ValueError(
                f"テキストのトークン数が制限を超えています。制限: {MAX_TOKENS_PER_TEXT}, "
                f"実際: {num_tokens}, ファイル: {filename}, テキスト先頭: '{text[:50]}...'"
            )
        total_tokens += num_tokens

    logger.info(f"全テキストの合計トークン数: {total_tokens}")

def embed_single_text(text: str, model: TextEmbeddingModel) -> List[float]:
    """単一のテキストをembeddingに変換する

    Args:
        text: 変換対象のテキスト
        model: 使用するembeddingモデル

    Returns:
        embedding値のリスト
    """
    try:
        embedding = model.get_embeddings([text])[0]
        return embedding.values
    except Exception as e:
        logger.error(f"Embedding生成エラー - テキスト: '{text[:50]}...': {str(e)}")
        raise

def embed_texts(text_info_list: List[Dict[str, str]], model_name: str = EMBEDDING_MODEL) -> List[List[float]]:
    """複数のテキストをembeddingに変換する

    Args:
        text_info_list: 変換対象のテキスト情報リスト。各要素は {'filename': 'ファイル名', 'content': 'テキスト内容'} の辞書。
        model_name: 使用するembeddingモデルの名前

    Returns:
        embedding値のリストのリスト
    """
    try:
        # トークン数の検証 (各テキスト単位)
        validate_token_count_per_text(text_info_list)

        # モデルの初期化
        model = TextEmbeddingModel.from_pretrained(model_name)

        # 結果を格納するリスト
        result = []

        # 各テキストを個別に処理
        for text_info in text_info_list:
            filename = text_info['filename']
            text = text_info['content']
            logger.info(f"{filename} のembedding生成を開始: '{text[:50]}...'")
            embedding = model.get_embeddings([text])[0]  # 各テキストを個別にベクトル化
            result.append(embedding.values)
            logger.info(f"{filename} のembedding生成完了: {len(embedding.values)}次元")

        logger.info(f"合計 {len(result)}件のembedding生成完了")
        return result

    except ValueError as ve:
        logger.error(f"トークン数検証エラー: {str(ve)}")
        raise
    except Exception as e:
        logger.error(f"Embedding生成プロセスエラー: {str(e)}")
        raise

def get_embedding_dimension(model_name: str = EMBEDDING_MODEL) -> int:
    model = TextEmbeddingModel.from_pretrained(model_name)
    sample_embedding = model.get_embeddings(["test"])[0]
    return len(sample_embedding.values)
