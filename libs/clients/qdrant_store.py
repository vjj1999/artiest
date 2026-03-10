"""
Qdrant Vector Store — 基于 qdrant-client 的向量存储实现

兼容 qdrant-client >= 1.16（使用 query_points 替代已废弃的 search）。
字符串 ID 通过 uuid5 确定性映射为 UUID，原始 ID 存入 payload._id。
"""
from __future__ import annotations

import uuid as _uuid
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from brain.logging import logger
from libs.clients.vector_store_interface import VectorStoreInterface, VectorSearchResult

# 用于 uuid5 的固定命名空间
_NS = _uuid.UUID("6ba7b810-9dad-11d1-80b4-00c04fd430c8")  # NAMESPACE_DNS


@dataclass
class QdrantConfig:
    """Qdrant 连接配置"""
    host: str = "localhost"
    port: int = 6333
    grpc_port: int = 6334
    api_key: Optional[str] = None
    prefer_grpc: bool = True
    # 距离度量: Cosine / Euclid / Dot
    distance: str = "Cosine"


def _ns_collection(collection: str, namespace: str) -> str:
    """拼接 namespace 到集合名"""
    return f"{collection}_{namespace}" if namespace else collection


def _to_uuid(string_id: str) -> str:
    """字符串 ID → 确定性 UUID 字符串"""
    return str(_uuid.uuid5(_NS, string_id))


class QdrantStore(VectorStoreInterface):
    """Qdrant 向量存储"""

    def __init__(self, cfg: Optional[QdrantConfig] = None, client: Any = None):
        self._cfg = cfg or QdrantConfig()
        self._client = client

    def _ensure_client(self):
        if self._client is not None:
            return
        try:
            from qdrant_client import QdrantClient
            self._client = QdrantClient(
                host=self._cfg.host,
                port=self._cfg.port,
                grpc_port=self._cfg.grpc_port,
                api_key=self._cfg.api_key,
                prefer_grpc=self._cfg.prefer_grpc,
            )
            logger.info("[Qdrant] 已连接 %s:%d", self._cfg.host, self._cfg.port)
        except Exception as e:
            raise RuntimeError(f"Qdrant 连接失败: {e}") from e

    def ensure_collection(self, collection: str, dim: int, namespace: str = "v1") -> None:
        self._ensure_client()
        from qdrant_client.models import Distance, VectorParams

        name = _ns_collection(collection, namespace)
        collections = [c.name for c in self._client.get_collections().collections]
        if name in collections:
            logger.debug("[Qdrant] 集合已存在: %s", name)
            return

        distance_map = {
            "Cosine": Distance.COSINE,
            "Euclid": Distance.EUCLID,
            "Dot": Distance.DOT,
        }
        dist = distance_map.get(self._cfg.distance, Distance.COSINE)

        self._client.create_collection(
            collection_name=name,
            vectors_config=VectorParams(size=dim, distance=dist),
        )
        logger.info("[Qdrant] 创建集合: %s (dim=%d, distance=%s)", name, dim, self._cfg.distance)

    def drop_collection(self, collection: str, namespace: str = "v1") -> None:
        self._ensure_client()
        name = _ns_collection(collection, namespace)
        try:
            self._client.delete_collection(name)
            logger.info("[Qdrant] 删除集合: %s", name)
        except Exception:
            pass

    def upsert(
        self,
        collection: str,
        ids: List[str],
        vectors: List[List[float]],
        metadatas: Optional[List[Dict[str, Any]]] = None,
        namespace: str = "v1",
    ) -> int:
        self._ensure_client()
        from qdrant_client.models import PointStruct

        name = _ns_collection(collection, namespace)
        metas = metadatas or [{}] * len(ids)

        points = []
        for uid, vec, meta in zip(ids, vectors, metas):
            payload = {**meta, "_id": uid}
            points.append(PointStruct(id=_to_uuid(uid), vector=vec, payload=payload))

        self._client.upsert(collection_name=name, points=points)
        logger.debug("[Qdrant] upsert %d 条到 %s", len(ids), name)
        return len(ids)

    def search(
        self,
        collection: str,
        query_vector: List[float],
        top_k: int = 10,
        namespace: str = "v1",
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[VectorSearchResult]:
        self._ensure_client()

        name = _ns_collection(collection, namespace)
        query_filter = None
        if filters:
            from qdrant_client.models import Filter, FieldCondition, MatchValue
            conditions = [
                FieldCondition(key=k, match=MatchValue(value=v))
                for k, v in filters.items()
            ]
            query_filter = Filter(must=conditions)

        resp = self._client.query_points(
            collection_name=name,
            query=query_vector,
            limit=top_k,
            query_filter=query_filter,
            with_payload=True,
        )

        results: List[VectorSearchResult] = []
        for point in resp.points:
            payload = point.payload or {}
            original_id = payload.pop("_id", str(point.id))
            results.append(VectorSearchResult(
                id=original_id,
                score=float(point.score),
                metadata=payload,
            ))
        return results

    def delete(
        self,
        collection: str,
        ids: List[str],
        namespace: str = "v1",
    ) -> int:
        self._ensure_client()
        from qdrant_client.models import PointIdsList

        name = _ns_collection(collection, namespace)
        uuid_ids = [_to_uuid(uid) for uid in ids]
        self._client.delete(
            collection_name=name,
            points_selector=PointIdsList(points=uuid_ids),
        )
        logger.debug("[Qdrant] 删除 %d 条从 %s", len(ids), name)
        return len(ids)
