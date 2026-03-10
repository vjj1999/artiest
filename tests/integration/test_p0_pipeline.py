"""
P0 端到端集成测试: mongomock + Qdrant 内存

跑通: 视频 → ingest → analyze(audio only) → embedding → search
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from tests.integration.conftest import make_wav, make_silence_wav


class TestP0Pipeline:
    """P0: ingest → analyze(audio) → embedding(Qdrant) → search"""

    def test_完整P0流水线(self, mongo_db, ingest, orchestrator, search):
        """端到端: ingest → analyze → embedding → search"""
        wav_path = make_wav(duration_s=3.0, freq=440)

        # ── Step 1: Ingest ──
        result = ingest.ingest(
            oss_path="oss://test-bucket/videos/sample.mp4",
            asset_type="video",
            filename="sample.mp4",
            duration_s=3.0,
            stages=["audio"],
        )
        asset_id = result["asset_id"]
        job_id = result["job_id"]
        pipeline_id = result["pipeline_id"]

        assert asset_id.startswith("ast_")
        assert job_id.startswith("job_")
        assert result["status"] == "queued"

        # 验证 MongoDB 写入
        asset_doc = mongo_db["assets"].find_one({"asset_id": asset_id})
        assert asset_doc is not None
        assert asset_doc["oss_path"] == "oss://test-bucket/videos/sample.mp4"

        job_doc = mongo_db["jobs"].find_one({"job_id": job_id})
        assert job_doc is not None
        assert job_doc["status"] == "queued"

        # ── Step 2: Analyze (audio) ──
        envelope = orchestrator.analyze(
            asset_id=asset_id,
            file_path=wav_path,
            stages=["audio"],
            job_id=job_id,
            pipeline_id=pipeline_id,
        )

        assert envelope.pipeline.status == "success"
        assert len(envelope.outputs) == 1

        audio_output = envelope.outputs[0]
        assert audio_output.stage == "audio"
        assert audio_output.status == "success"
        assert "silence_ratio" in audio_output.data
        assert "rms_mean" in audio_output.data
        assert "zcr" in audio_output.data
        assert audio_output.data["rms_mean"] > 0

        # 验证 ResultEnvelope 写入 MongoDB
        result_doc = mongo_db["perception_results"].find_one(
            {"pipeline.pipeline_id": pipeline_id}
        )
        assert result_doc is not None

        # 验证 job 状态更新
        job_doc = mongo_db["jobs"].find_one({"job_id": job_id})
        assert job_doc["status"] == "success"

        # ── Step 3: Search (通过 embedding) ──
        # 构建和 audio embedding 结构相同的查询向量
        query_vec = [0.0, audio_output.data["rms_mean"], 0.0, 0.0] + [0.0] * 28
        results = search.search(query_vector=query_vec, top_k=5)
        assert len(results) >= 1
        assert results[0]["metadata"]["asset_id"] == asset_id
        assert results[0]["metadata"]["stage"] == "audio"

    def test_多资产入库并搜索(self, mongo_db, ingest, orchestrator, search):
        """多个资产入库后搜索最相似的"""
        # 资产 A: 有声音频
        wav_a = make_wav(duration_s=2.0, freq=440, amplitude=0.8)
        r_a = ingest.ingest(oss_path="oss://b/a.mp4", stages=["audio"])
        env_a = orchestrator.analyze(r_a["asset_id"], wav_a, ["audio"])

        # 资产 B: 静音音频
        wav_b = make_silence_wav(duration_s=2.0)
        r_b = ingest.ingest(oss_path="oss://b/b.mp4", stages=["audio"])
        env_b = orchestrator.analyze(r_b["asset_id"], wav_b, ["audio"])

        # 资产 C: 有声音频（不同频率）
        wav_c = make_wav(duration_s=2.0, freq=1000, amplitude=0.3)
        r_c = ingest.ingest(oss_path="oss://b/c.mp4", stages=["audio"])
        env_c = orchestrator.analyze(r_c["asset_id"], wav_c, ["audio"])

        # 查询：用 A 的特征向量搜索
        a_data = env_a.outputs[0].data
        query = [a_data["silence_ratio"], a_data["rms_mean"], a_data["zcr"], 0.0] + [0.0] * 28
        results = search.search(query_vector=query, top_k=10)

        assert len(results) == 3
        # 第一个应该是 A 自己（最相似）
        assert results[0]["metadata"]["asset_id"] == r_a["asset_id"]

    def test_静音检测写入正确(self, mongo_db, ingest, orchestrator, search):
        """验证静音音频的 silence_ratio 接近 1.0"""
        wav = make_silence_wav(3.0)
        r = ingest.ingest(oss_path="oss://b/silent.mp4")
        env = orchestrator.analyze(r["asset_id"], wav, ["audio"])

        data = env.outputs[0].data
        assert data["silence_ratio"] >= 0.95
        assert data["rms_mean"] < 0.001

    def test_Qdrant_metadata_filter(self, mongo_db, ingest, orchestrator, search):
        """验证 Qdrant metadata 过滤"""
        wav = make_wav(2.0)
        r = ingest.ingest(oss_path="oss://b/x.mp4", stages=["audio"])
        orchestrator.analyze(r["asset_id"], wav, ["audio"])

        # 过滤 stage=audio
        q = [0.0] * 32
        results = search.search(query_vector=q, filters={"stage": "audio"})
        assert all(r["metadata"]["stage"] == "audio" for r in results)

        # 过滤不存在的 stage
        results = search.search(query_vector=q, filters={"stage": "visual"})
        assert len(results) == 0
