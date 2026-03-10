"""T1: ResultEnvelope schema 验证"""
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from libs.schema.result_envelope import (
    AssetMeta, AssetType, PipelineInfo, StageResult, StageStatus,
    EmbeddingRef, ResultEnvelope,
)


def _sample_envelope() -> ResultEnvelope:
    return ResultEnvelope(
        asset_meta=AssetMeta(
            asset_id="ast_001",
            asset_type=AssetType.VIDEO,
            oss_path="oss://bucket/videos/sample.mp4",
            filename="sample.mp4",
            duration_s=30.5,
            size_bytes=15728640,
        ),
        pipeline=PipelineInfo(
            pipeline_id="pipe_001",
            stages=["audio"],
            status=StageStatus.SUCCESS,
        ),
        outputs=[
            StageResult(
                stage="audio",
                status=StageStatus.SUCCESS,
                duration_ms=1250.0,
                data={"silence_ratio": 0.15, "rms_mean": 0.042, "zcr": 0.08},
            ),
        ],
        embedding_ids=[
            EmbeddingRef(
                store="milvus",
                collection="perception",
                embedding_id="ast_001_audio_v1",
                embedding_version="v1",
                dim=192,
            ),
        ],
    )


class TestResultEnvelope:
    """验证 ResultEnvelope schema"""

    def test_构建完整envelope(self):
        env = _sample_envelope()
        assert env.asset_meta.asset_id == "ast_001"
        assert env.pipeline.status == "success"
        assert len(env.outputs) == 1
        assert env.outputs[0].stage == "audio"
        assert len(env.embedding_ids) == 1

    def test_json序列化往返(self):
        env = _sample_envelope()
        json_str = env.model_dump_json()
        restored = ResultEnvelope.model_validate_json(json_str)
        assert restored.asset_meta.asset_id == env.asset_meta.asset_id
        assert restored.pipeline.pipeline_id == env.pipeline.pipeline_id
        assert restored.outputs[0].data["silence_ratio"] == 0.15

    def test_json_schema生成(self):
        schema = ResultEnvelope.model_json_schema()
        assert "properties" in schema
        assert "asset_meta" in schema["properties"]
        assert "pipeline" in schema["properties"]
        assert "outputs" in schema["properties"]
        assert "embedding_ids" in schema["properties"]

    def test_包含必要字段(self):
        schema = ResultEnvelope.model_json_schema()
        json_str = json.dumps(schema)
        for key in ["asset_meta", "pipeline", "outputs", "embedding_ids"]:
            assert key in json_str

    def test_空outputs合法(self):
        env = ResultEnvelope(
            asset_meta=AssetMeta(asset_id="ast_002", oss_path="oss://x/y.mp4"),
            pipeline=PipelineInfo(pipeline_id="pipe_002"),
        )
        assert env.outputs == []
        assert env.embedding_ids == []

    def test_stage_status枚举值(self):
        for status in ["pending", "running", "success", "failed", "skipped"]:
            sr = StageResult(stage="audio", status=status)
            assert sr.status == status

    def test_asset_type枚举值(self):
        for t in ["video", "audio", "image"]:
            meta = AssetMeta(asset_id="x", asset_type=t, oss_path="oss://a/b")
            assert meta.asset_type == t
