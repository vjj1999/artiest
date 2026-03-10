"""
Vector Store Interface — 向量存储统一抽象

定义 upsert / search / delete 三大操作，
Milvus 和 Qdrant 各自实现。namespace 用于隔离不同 embedding_version。
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class VectorSearchResult:
    """向量搜索结果"""
    id: str
    score: float
    metadata: Dict[str, Any] = field(default_factory=dict)


class VectorStoreInterface(ABC):
    """向量存储统一接口"""

    @abstractmethod
    def upsert(
        self,
        collection: str,
        ids: List[str],
        vectors: List[List[float]],
        metadatas: Optional[List[Dict[str, Any]]] = None,
        namespace: str = "v1",
    ) -> int:
        """
        写入/更新向量

        Args:
            collection: 集合名
            ids: 向量 ID 列表
            vectors: 向量数据列表
            metadatas: 可选元数据列表（与 ids 对齐）
            namespace: 命名空间（用于 embedding_version 隔离）

        Returns:
            成功写入的数量
        """

    @abstractmethod
    def search(
        self,
        collection: str,
        query_vector: List[float],
        top_k: int = 10,
        namespace: str = "v1",
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[VectorSearchResult]:
        """
        相似度搜索

        Args:
            collection: 集合名
            query_vector: 查询向量
            top_k: 返回前 K 个结果
            namespace: 命名空间
            filters: 元数据过滤条件

        Returns:
            搜索结果列表，按 score 降序
        """

    @abstractmethod
    def delete(
        self,
        collection: str,
        ids: List[str],
        namespace: str = "v1",
    ) -> int:
        """
        删除向量

        Args:
            collection: 集合名
            ids: 要删除的向量 ID 列表
            namespace: 命名空间

        Returns:
            成功删除的数量
        """

    @abstractmethod
    def ensure_collection(
        self,
        collection: str,
        dim: int,
        namespace: str = "v1",
    ) -> None:
        """确保集合存在（不存在则创建）"""

    @abstractmethod
    def drop_collection(self, collection: str, namespace: str = "v1") -> None:
        """删除集合"""
