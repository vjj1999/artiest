"""T8: Search Endpoint 单元测试"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from services.orchestrator.search_handler import SearchHandler, SearchConfig
from tests.unit.test_vector_store import InMemoryVectorStore


@pytest.fixture
def search_env():
    """准备带数据的搜索环境"""
    store = InMemoryVectorStore()
    store.ensure_collection("perception", dim=4, namespace="v1")
    store.upsert(
        "perception",
        ids=["ast_001_audio", "ast_002_audio", "ast_003_visual"],
        vectors=[
            [0.9, 0.1, 0.0, 0.5],
            [0.1, 0.8, 0.0, 0.3],
            [0.0, 0.0, 0.9, 0.2],
        ],
        metadatas=[
            {"asset_id": "ast_001", "stage": "audio"},
            {"asset_id": "ast_002", "stage": "audio"},
            {"asset_id": "ast_003", "stage": "visual"},
        ],
        namespace="v1",
    )
    handler = SearchHandler(store)
    return handler, store


class TestSearchHandler:
    """验证搜索处理器"""

    def test_基础搜索(self, search_env):
        handler, _ = search_env
        results = handler.search(query_vector=[0.9, 0.1, 0.0, 0.5], top_k=3)
        assert len(results) == 3
        assert results[0]["id"] == "ast_001_audio"
        assert results[0]["score"] > results[1]["score"]

    def test_top_k限制(self, search_env):
        handler, _ = search_env
        results = handler.search(query_vector=[0.5, 0.5, 0.0, 0.0], top_k=1)
        assert len(results) == 1

    def test_metadata_filter(self, search_env):
        handler, _ = search_env
        results = handler.search(
            query_vector=[0.5, 0.5, 0.5, 0.5],
            filters={"stage": "audio"},
        )
        assert all(r["metadata"]["stage"] == "audio" for r in results)
        assert len(results) == 2

    def test_max_top_k上限(self, search_env):
        handler, _ = search_env
        handler._cfg.max_top_k = 2
        results = handler.search(query_vector=[0.5, 0.5, 0.5, 0.5], top_k=100)
        assert len(results) <= 2

    def test_返回格式(self, search_env):
        handler, _ = search_env
        results = handler.search(query_vector=[0.9, 0.1, 0.0, 0.5])
        for r in results:
            assert "id" in r
            assert "score" in r
            assert "metadata" in r

    def test_空结果(self):
        store = InMemoryVectorStore()
        store.ensure_collection("perception", dim=4)
        handler = SearchHandler(store)
        results = handler.search(query_vector=[1.0, 0.0, 0.0, 0.0])
        assert results == []

    def test_自定义namespace(self):
        store = InMemoryVectorStore()
        store.ensure_collection("perception", dim=2, namespace="v2")
        store.upsert("perception", ["x"], [[1.0, 0.0]], namespace="v2")
        handler = SearchHandler(store)
        # v1 空，v2 有数据
        r1 = handler.search([1.0, 0.0], namespace="v1")
        r2 = handler.search([1.0, 0.0], namespace="v2")
        assert len(r1) == 0
        assert len(r2) == 1
