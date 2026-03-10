"""T4: API Spec 验证（OpenAPI + gRPC proto 结构检查）"""
import sys
from pathlib import Path

import pytest

OPENAPI_PATH = Path(__file__).resolve().parent.parent.parent / "docs" / "api" / "openapi.yaml"
PROTO_PATH = Path(__file__).resolve().parent.parent.parent / "docs" / "api" / "perception.proto"


class TestOpenAPISpec:
    """验证 OpenAPI spec 包含所有必要端点"""

    @pytest.fixture(autouse=True)
    def _load(self):
        import yaml
        self.spec = yaml.safe_load(OPENAPI_PATH.read_text(encoding="utf-8"))

    def test_版本信息(self):
        assert self.spec["info"]["version"] == "1.0.0"

    def test_ingest端点(self):
        assert "/api/v1/ingest" in self.spec["paths"]
        assert "post" in self.spec["paths"]["/api/v1/ingest"]

    def test_analyze端点(self):
        assert "/api/v1/analyze" in self.spec["paths"]
        assert "post" in self.spec["paths"]["/api/v1/analyze"]

    def test_search端点(self):
        assert "/api/v1/search" in self.spec["paths"]
        assert "post" in self.spec["paths"]["/api/v1/search"]

    def test_get_result端点(self):
        path = "/api/v1/result/{pipeline_id}"
        assert path in self.spec["paths"]
        assert "get" in self.spec["paths"][path]

    def test_schema定义一致(self):
        schemas = self.spec["components"]["schemas"]
        for name in ["IngestRequest", "AnalyzeRequest", "SearchRequest", "ResultEnvelope"]:
            assert name in schemas, f"缺少 schema: {name}"


class TestProtoSpec:
    """验证 gRPC proto 包含所有必要 RPC"""

    @pytest.fixture(autouse=True)
    def _load(self):
        self.content = PROTO_PATH.read_text(encoding="utf-8")

    def test_文件存在(self):
        assert PROTO_PATH.exists()

    def test_ingest_rpc(self):
        assert "rpc Ingest" in self.content

    def test_analyze_rpc(self):
        assert "rpc Analyze" in self.content

    def test_search_rpc(self):
        assert "rpc Search" in self.content

    def test_get_result_rpc(self):
        assert "rpc GetResult" in self.content

    def test_字段一致性(self):
        """HTTP 和 gRPC 共有的关键字段"""
        for field in ["asset_id", "oss_path", "pipeline_id", "query_vector", "top_k"]:
            assert field in self.content, f"proto 缺少字段: {field}"
