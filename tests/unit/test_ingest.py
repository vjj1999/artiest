"""T5: Ingest Service 单元测试（使用 mock MongoDB）"""
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from services.ingest.handler import IngestHandler, IngestConfig, _gen_id


class TestIngestHandler:
    """验证 Ingest 处理器逻辑"""

    def test_gen_id格式(self):
        aid = _gen_id("ast")
        assert aid.startswith("ast_")
        assert len(aid) == 16  # ast_ + 12 hex

    def test_gen_id唯一(self):
        ids = {_gen_id("x") for _ in range(100)}
        assert len(ids) == 100

    @patch("services.ingest.handler.IngestHandler._ensure_db")
    def test_ingest注册资产(self, mock_ensure):
        handler = IngestHandler()
        # mock MongoDB 集合
        mock_assets = MagicMock()
        mock_jobs = MagicMock()
        handler._db = {"assets": mock_assets, "jobs": mock_jobs}

        result = handler.ingest(
            oss_path="oss://bucket/video/test.mp4",
            asset_type="video",
            filename="test.mp4",
        )

        assert "asset_id" in result
        assert "job_id" in result
        assert result["status"] == "queued"
        mock_assets.insert_one.assert_called_once()
        mock_jobs.insert_one.assert_called_once()

    @patch("services.ingest.handler.IngestHandler._ensure_db")
    def test_ingest带stages(self, mock_ensure):
        handler = IngestHandler()
        handler._db = {"assets": MagicMock(), "jobs": MagicMock()}

        result = handler.ingest(
            oss_path="oss://bucket/video/test.mp4",
            stages=["audio", "visual"],
        )
        assert result["status"] == "queued"
        # 验证 job 文档包含 stages
        job_doc = handler._db["jobs"].insert_one.call_args[0][0]
        assert job_doc["stages"] == ["audio", "visual"]

    @patch("services.ingest.handler.IngestHandler._ensure_db")
    def test_ingest默认stages(self, mock_ensure):
        handler = IngestHandler()
        handler._db = {"assets": MagicMock(), "jobs": MagicMock()}

        handler.ingest(oss_path="oss://x/y.mp4")
        job_doc = handler._db["jobs"].insert_one.call_args[0][0]
        assert job_doc["stages"] == ["audio"]

    @patch("services.ingest.handler.IngestHandler._ensure_db")
    def test_get_asset存在(self, mock_ensure):
        handler = IngestHandler()
        mock_col = MagicMock()
        mock_col.find_one.return_value = {"asset_id": "ast_001", "oss_path": "oss://x"}
        handler._db = {"assets": mock_col}

        result = handler.get_asset("ast_001")
        assert result is not None
        assert result["asset_id"] == "ast_001"

    @patch("services.ingest.handler.IngestHandler._ensure_db")
    def test_get_asset不存在(self, mock_ensure):
        handler = IngestHandler()
        mock_col = MagicMock()
        mock_col.find_one.return_value = None
        handler._db = {"assets": mock_col}

        result = handler.get_asset("nonexistent")
        assert result is None
