"""T3: VectorStoreInterface 接口验证（不需要实际连接）"""
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from libs.clients.vector_store_interface import VectorStoreInterface, VectorSearchResult


class InMemoryVectorStore(VectorStoreInterface):
    """内存向量存储，用于测试接口契约"""

    def __init__(self):
        # {ns_collection: {id: (vector, metadata)}}
        self._data: Dict[str, Dict[str, tuple]] = {}

    def _key(self, collection: str, namespace: str) -> str:
        return f"{collection}_{namespace}" if namespace else collection

    def ensure_collection(self, collection: str, dim: int, namespace: str = "v1") -> None:
        key = self._key(collection, namespace)
        if key not in self._data:
            self._data[key] = {}

    def drop_collection(self, collection: str, namespace: str = "v1") -> None:
        key = self._key(collection, namespace)
        self._data.pop(key, None)

    def upsert(
        self,
        collection: str,
        ids: List[str],
        vectors: List[List[float]],
        metadatas: Optional[List[Dict[str, Any]]] = None,
        namespace: str = "v1",
    ) -> int:
        key = self._key(collection, namespace)
        if key not in self._data:
            self._data[key] = {}
        metas = metadatas or [{}] * len(ids)
        for uid, vec, meta in zip(ids, vectors, metas):
            self._data[key][uid] = (vec, meta)
        return len(ids)

    def search(
        self,
        collection: str,
        query_vector: List[float],
        top_k: int = 10,
        namespace: str = "v1",
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[VectorSearchResult]:
        key = self._key(collection, namespace)
        store = self._data.get(key, {})

        results = []
        for uid, (vec, meta) in store.items():
            # 过滤
            if filters:
                skip = False
                for fk, fv in filters.items():
                    if meta.get(fk) != fv:
                        skip = True
                        break
                if skip:
                    continue
            # 余弦相似度（简化：点积）
            score = sum(a * b for a, b in zip(query_vector, vec))
            results.append(VectorSearchResult(id=uid, score=score, metadata=meta))

        results.sort(key=lambda r: r.score, reverse=True)
        return results[:top_k]

    def delete(
        self,
        collection: str,
        ids: List[str],
        namespace: str = "v1",
    ) -> int:
        key = self._key(collection, namespace)
        store = self._data.get(key, {})
        count = 0
        for uid in ids:
            if uid in store:
                del store[uid]
                count += 1
        return count


@pytest.fixture
def store():
    return InMemoryVectorStore()


class TestVectorStoreInterface:
    """验证 VectorStoreInterface 接口契约"""

    def test_ensure_collection(self, store: InMemoryVectorStore):
        store.ensure_collection("test", dim=4, namespace="v1")
        assert "test_v1" in store._data

    def test_upsert_and_search(self, store: InMemoryVectorStore):
        store.ensure_collection("test", dim=3, namespace="v1")
        store.upsert(
            collection="test",
            ids=["a", "b"],
            vectors=[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            metadatas=[{"tag": "x"}, {"tag": "y"}],
            namespace="v1",
        )
        results = store.search("test", query_vector=[1.0, 0.0, 0.0], top_k=2, namespace="v1")
        assert len(results) == 2
        assert results[0].id == "a"
        assert results[0].score > results[1].score

    def test_search_with_filter(self, store: InMemoryVectorStore):
        store.ensure_collection("test", dim=2, namespace="v1")
        store.upsert("test", ["a", "b"], [[1.0, 0.0], [0.5, 0.5]], [{"stage": "audio"}, {"stage": "visual"}])
        results = store.search("test", [1.0, 0.0], top_k=10, filters={"stage": "audio"})
        assert len(results) == 1
        assert results[0].id == "a"

    def test_delete(self, store: InMemoryVectorStore):
        store.ensure_collection("test", dim=2)
        store.upsert("test", ["a", "b"], [[1.0, 0.0], [0.0, 1.0]])
        deleted = store.delete("test", ["a"])
        assert deleted == 1
        results = store.search("test", [1.0, 0.0])
        assert len(results) == 1
        assert results[0].id == "b"

    def test_namespace隔离(self, store: InMemoryVectorStore):
        store.ensure_collection("test", dim=2, namespace="v1")
        store.ensure_collection("test", dim=2, namespace="v2")
        store.upsert("test", ["a"], [[1.0, 0.0]], namespace="v1")
        store.upsert("test", ["b"], [[0.0, 1.0]], namespace="v2")
        r1 = store.search("test", [1.0, 0.0], namespace="v1")
        r2 = store.search("test", [1.0, 0.0], namespace="v2")
        assert len(r1) == 1 and r1[0].id == "a"
        assert len(r2) == 1 and r2[0].id == "b"

    def test_drop_collection(self, store: InMemoryVectorStore):
        store.ensure_collection("test", dim=2)
        store.upsert("test", ["a"], [[1.0, 0.0]])
        store.drop_collection("test")
        assert "test_v1" not in store._data

    def test_upsert覆盖(self, store: InMemoryVectorStore):
        store.ensure_collection("test", dim=2)
        store.upsert("test", ["a"], [[1.0, 0.0]], [{"v": 1}])
        store.upsert("test", ["a"], [[0.0, 1.0]], [{"v": 2}])
        results = store.search("test", [0.0, 1.0])
        assert results[0].id == "a"
        assert results[0].metadata["v"] == 2
