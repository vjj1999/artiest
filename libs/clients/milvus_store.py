"""
Milvus Vector Store — 基于 pymilvus 的向量存储实现
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from brain.logging import logger
from libs.clients.vector_store_interface import VectorStoreInterface, VectorSearchResult


@dataclass
class MilvusConfig:
    """Milvus 连接配置"""
    host: str = "localhost"
    port: int = 19530
    user: str = ""
    password: str = ""
    db_name: str = "default"
    # 索引类型：IVF_FLAT / HNSW / IVF_SQ8
    index_type: str = "IVF_FLAT"
    metric_type: str = "COSINE"
    nlist: int = 128
    nprobe: int = 16


def _ns_collection(collection: str, namespace: str) -> str:
    """拼接 namespace 到集合名，隔离不同 embedding_version"""
    return f"{collection}_{namespace}" if namespace else collection


class MilvusStore(VectorStoreInterface):
    """Milvus 向量存储"""

    def __init__(self, cfg: Optional[MilvusConfig] = None):
        self._cfg = cfg or MilvusConfig()
        self._connections = None
        self._connected = False

    def _ensure_connection(self):
        if self._connected:
            return
        try:
            from pymilvus import connections
            self._connections = connections
            connections.connect(
                alias="default",
                host=self._cfg.host,
                port=self._cfg.port,
                user=self._cfg.user or "",
                password=self._cfg.password or "",
                db_name=self._cfg.db_name,
            )
            self._connected = True
            logger.info("[Milvus] 已连接 %s:%d", self._cfg.host, self._cfg.port)
        except Exception as e:
            raise RuntimeError(f"Milvus 连接失败: {e}") from e

    def ensure_collection(self, collection: str, dim: int, namespace: str = "v1") -> None:
        self._ensure_connection()
        from pymilvus import CollectionSchema, FieldSchema, DataType, Collection, utility

        name = _ns_collection(collection, namespace)
        if utility.has_collection(name):
            logger.debug("[Milvus] 集合已存在: %s", name)
            return

        fields = [
            FieldSchema(name="id", dtype=DataType.VARCHAR, is_primary=True, max_length=256),
            FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=dim),
            FieldSchema(name="metadata", dtype=DataType.JSON),
        ]
        schema = CollectionSchema(fields, description=f"{collection} (ns={namespace})")
        col = Collection(name=name, schema=schema)

        # 创建索引
        index_params = {
            "index_type": self._cfg.index_type,
            "metric_type": self._cfg.metric_type,
            "params": {"nlist": self._cfg.nlist},
        }
        col.create_index("vector", index_params)
        col.load()
        logger.info("[Milvus] 创建集合: %s (dim=%d)", name, dim)

    def drop_collection(self, collection: str, namespace: str = "v1") -> None:
        self._ensure_connection()
        from pymilvus import utility
        name = _ns_collection(collection, namespace)
        if utility.has_collection(name):
            utility.drop_collection(name)
            logger.info("[Milvus] 删除集合: %s", name)

    def upsert(
        self,
        collection: str,
        ids: List[str],
        vectors: List[List[float]],
        metadatas: Optional[List[Dict[str, Any]]] = None,
        namespace: str = "v1",
    ) -> int:
        self._ensure_connection()
        from pymilvus import Collection

        name = _ns_collection(collection, namespace)
        col = Collection(name)
        metas = metadatas or [{}] * len(ids)

        data = [ids, vectors, metas]
        col.upsert(data)
        col.flush()
        logger.debug("[Milvus] upsert %d 条到 %s", len(ids), name)
        return len(ids)

    def search(
        self,
        collection: str,
        query_vector: List[float],
        top_k: int = 10,
        namespace: str = "v1",
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[VectorSearchResult]:
        self._ensure_connection()
        from pymilvus import Collection

        name = _ns_collection(collection, namespace)
        col = Collection(name)
        col.load()

        # 构建过滤表达式
        expr = ""
        if filters:
            parts = []
            for k, v in filters.items():
                if isinstance(v, str):
                    parts.append(f'metadata["{k}"] == "{v}"')
                else:
                    parts.append(f'metadata["{k}"] == {v}')
            expr = " and ".join(parts)

        search_params = {"metric_type": self._cfg.metric_type, "params": {"nprobe": self._cfg.nprobe}}
        results = col.search(
            data=[query_vector],
            anns_field="vector",
            param=search_params,
            limit=top_k,
            expr=expr or None,
            output_fields=["metadata"],
        )

        out: List[VectorSearchResult] = []
        for hits in results:
            for hit in hits:
                out.append(VectorSearchResult(
                    id=str(hit.id),
                    score=float(hit.score),
                    metadata=hit.entity.get("metadata", {}),
                ))
        return out

    def delete(
        self,
        collection: str,
        ids: List[str],
        namespace: str = "v1",
    ) -> int:
        self._ensure_connection()
        from pymilvus import Collection

        name = _ns_collection(collection, namespace)
        col = Collection(name)
        expr = f'id in {ids}'
        col.delete(expr)
        col.flush()
        logger.debug("[Milvus] 删除 %d 条从 %s", len(ids), name)
        return len(ids)
