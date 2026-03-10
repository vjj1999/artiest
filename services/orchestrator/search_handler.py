"""
Search Handler — 向量相似度搜索

基于 VectorStoreInterface 执行 top_k 搜索，支持 metadata 过滤。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from brain.logging import logger
from libs.clients.vector_store_interface import VectorStoreInterface, VectorSearchResult


@dataclass
class SearchConfig:
    """搜索配置"""
    default_collection: str = "perception"
    default_namespace: str = "v1"
    default_top_k: int = 10
    max_top_k: int = 100


class SearchHandler:
    """搜索处理器"""

    def __init__(
        self,
        vector_store: VectorStoreInterface,
        cfg: Optional[SearchConfig] = None,
    ):
        self._store = vector_store
        self._cfg = cfg or SearchConfig()

    def search(
        self,
        query_vector: List[float],
        top_k: Optional[int] = None,
        collection: Optional[str] = None,
        namespace: Optional[str] = None,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """
        执行向量相似度搜索

        Args:
            query_vector: 查询向量
            top_k: 返回数量（默认 10，最大 100）
            collection: 集合名（默认 perception）
            namespace: 命名空间（默认 v1）
            filters: 元数据过滤

        Returns:
            [{"id": "...", "score": 0.95, "metadata": {...}}, ...]
        """
        cfg = self._cfg
        k = min(top_k or cfg.default_top_k, cfg.max_top_k)
        col = collection or cfg.default_collection
        ns = namespace or cfg.default_namespace

        results = self._store.search(
            collection=col,
            query_vector=query_vector,
            top_k=k,
            namespace=ns,
            filters=filters,
        )

        logger.info("[Search] 搜索完成: collection=%s, top_k=%d, 命中=%d", col, k, len(results))

        return [
            {"id": r.id, "score": r.score, "metadata": r.metadata}
            for r in results
        ]
