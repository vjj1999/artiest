"""T7: Orchestrator 单元测试"""
import sys
import tempfile
import wave
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from services.orchestrator.pipeline import Orchestrator, OrchestratorConfig
from tests.unit.test_vector_store import InMemoryVectorStore


def _make_wav(duration_s: float = 2.0, freq: float = 440, sr: int = 16000) -> str:
    """生成临时 WAV 文件"""
    n = int(duration_s * sr)
    t = np.arange(n) / sr
    pcm = (0.5 * np.sin(2 * np.pi * freq * t) * 32767).astype(np.int16)
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    with wave.open(tmp.name, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(pcm.tobytes())
    return tmp.name


def _mock_db(asset_doc: dict = None):
    """创建 mock MongoDB"""
    db = MagicMock()
    default_asset = {
        "asset_id": "ast_001",
        "asset_type": "video",
        "oss_path": "oss://bucket/test.mp4",
        "filename": "test.mp4",
        "duration_s": 2.0,
    }
    db["assets"].find_one.return_value = asset_doc or default_asset
    db["perception_results"].insert_one = MagicMock()
    db["jobs"].update_one = MagicMock()
    return db


class TestOrchestrator:
    """验证编排器核心逻辑"""

    def test_audio流水线(self):
        wav_path = _make_wav(2.0)
        store = InMemoryVectorStore()
        db = _mock_db()
        orch = Orchestrator(vector_store=store, mongo_db=db)

        envelope = orch.analyze(
            asset_id="ast_001",
            file_path=wav_path,
            stages=["audio"],
            job_id="job_001",
        )

        assert envelope.pipeline.status == "success"
        assert len(envelope.outputs) == 1
        assert envelope.outputs[0].stage == "audio"
        assert envelope.outputs[0].status == "success"
        assert "silence_ratio" in envelope.outputs[0].data
        assert "rms_mean" in envelope.outputs[0].data
        assert "zcr" in envelope.outputs[0].data

    def test_结果写入mongo(self):
        wav_path = _make_wav()
        db = _mock_db()
        orch = Orchestrator(mongo_db=db)

        orch.analyze(asset_id="ast_001", file_path=wav_path)
        db["perception_results"].insert_one.assert_called_once()

    def test_embedding写入向量存储(self):
        wav_path = _make_wav()
        store = InMemoryVectorStore()
        db = _mock_db()
        orch = Orchestrator(vector_store=store, mongo_db=db)

        orch.analyze(asset_id="ast_001", file_path=wav_path, stages=["audio"])
        # 验证向量已写入
        results = store.search("perception", [1.0] + [0.0] * 31, top_k=10)
        assert len(results) >= 1
        assert results[0].metadata["asset_id"] == "ast_001"

    def test_job状态更新(self):
        wav_path = _make_wav()
        db = _mock_db()
        orch = Orchestrator(mongo_db=db)

        orch.analyze(asset_id="ast_001", file_path=wav_path, job_id="job_001")
        # 至少调用两次：running + success
        assert db["jobs"].update_one.call_count >= 2

    def test_未实现阶段标记skipped(self):
        wav_path = _make_wav()
        db = _mock_db()
        orch = Orchestrator(mongo_db=db)

        envelope = orch.analyze(asset_id="ast_001", file_path=wav_path, stages=["visual"])
        assert envelope.outputs[0].status == "skipped"

    def test_资产不存在抛异常(self):
        db = _mock_db()
        db["assets"].find_one.return_value = None
        orch = Orchestrator(mongo_db=db)

        with pytest.raises(ValueError, match="资产不存在"):
            orch.analyze(asset_id="nonexist", file_path="x.wav")

    def test_pipeline_id和时间(self):
        wav_path = _make_wav()
        db = _mock_db()
        orch = Orchestrator(mongo_db=db)

        envelope = orch.analyze(asset_id="ast_001", file_path=wav_path)
        assert envelope.pipeline.pipeline_id.startswith("pipe_")
        assert envelope.pipeline.started_at is not None
        assert envelope.pipeline.finished_at is not None
        assert envelope.pipeline.total_duration_ms > 0
