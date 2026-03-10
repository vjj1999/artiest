"""T2: mongo_init.js 结构验证（不需要实际 MongoDB）"""
import sys
from pathlib import Path

import pytest

INIT_JS = Path(__file__).resolve().parent.parent.parent / "deploy" / "mongo_init.js"


class TestMongoInit:
    """验证 mongo_init.js 包含必要的集合和索引定义"""

    @pytest.fixture(autouse=True)
    def _load(self):
        self.content = INIT_JS.read_text(encoding="utf-8")

    def test_文件存在(self):
        assert INIT_JS.exists()

    def test_创建assets集合(self):
        assert 'createCollection("assets"' in self.content

    def test_创建perception_results集合(self):
        assert 'createCollection("perception_results"' in self.content

    def test_创建jobs集合(self):
        assert 'createCollection("jobs"' in self.content

    def test_assets唯一索引(self):
        assert "db.assets.createIndex" in self.content
        assert "unique: true" in self.content

    def test_pipeline_id唯一索引(self):
        assert "pipeline.pipeline_id" in self.content

    def test_job_id唯一索引(self):
        assert "db.jobs.createIndex" in self.content
