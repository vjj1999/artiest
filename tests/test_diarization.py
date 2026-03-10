"""
pyannote.audio Speaker Diarization 测试

快速测试（不加载模型）：
    python -m pytest tests/test_diarization.py -v -k "not slow"

集成测试（需要模型 + HF token）：
    $env:HF_TOKEN="hf_xxx"; python -m pytest tests/test_diarization.py -v -m slow
"""
import numpy as np
import pytest

from brain.perception.audio.types import SpeakerTurn, TranscriptSegment
from brain.perception.audio.diarization import PyannoteDiarizationConfig, diarize_pyannote


def _to_pcm16_bytes(x: np.ndarray) -> bytes:
    x = np.clip(x, -1.0, 1.0)
    return (x * 32767.0).astype(np.int16).tobytes()


# ============================================================
# 1. 不加载模型的快速测试
# ============================================================

class TestPyannoteDiarizationConfig:
    def test_defaults(self):
        cfg = PyannoteDiarizationConfig()
        assert cfg.sample_rate == 16000
        assert cfg.model_id == "pyannote/speaker-diarization-3.1"
        assert cfg.device == "cpu"
        assert cfg.speaker_prefix == "spk"

    def test_custom_config(self):
        cfg = PyannoteDiarizationConfig(
            model_id="pyannote/speaker-diarization-community-1",
            device="cuda",
            hf_token="test_token",
        )
        assert cfg.model_id == "pyannote/speaker-diarization-community-1"
        assert cfg.device == "cuda"
        assert cfg.hf_token == "test_token"


class TestEdgeCases:
    def test_empty_bytes_returns_empty_list(self):
        cfg = PyannoteDiarizationConfig()
        result = diarize_pyannote(b"", cfg)
        assert result == []

    def test_speaker_turn_fields(self):
        turn = SpeakerTurn(start_ms=100, end_ms=2000, speaker_id="spk0", backend="pyannote")
        assert turn.start_ms == 100
        assert turn.end_ms == 2000
        assert turn.speaker_id == "spk0"
        assert turn.backend == "pyannote"

    def test_transcript_segment_fields(self):
        seg = TranscriptSegment(start_ms=0, end_ms=1000, speaker_id="spk0", text="hello")
        assert seg.text == "hello"
        assert seg.rms_dbfs is None


# ============================================================
# 2. 集成测试（需要 pyannote 模型）
# ============================================================

@pytest.mark.slow
class TestDiarizePyannoteIntegration:
    @pytest.fixture
    def cfg(self):
        import os
        token = os.getenv("HF_TOKEN", "")
        model_id = os.getenv("PYANNOTE_MODEL", "pyannote/speaker-diarization-3.1")
        return PyannoteDiarizationConfig(
            hf_token=token,
            model_id=model_id,
            device="cuda",
        )

    def test_silent_audio(self, cfg):
        silence = np.zeros(16000 * 2, dtype=np.float32)
        pcm = _to_pcm16_bytes(silence)
        turns = diarize_pyannote(pcm, cfg)
        assert len(turns) >= 1
        assert turns[0].backend == "pyannote"

    def test_single_tone(self, cfg):
        t = np.arange(16000 * 3, dtype=np.float32) / 16000
        tone = 0.3 * np.sin(2 * np.pi * 440 * t)
        pcm = _to_pcm16_bytes(tone)
        turns = diarize_pyannote(pcm, cfg)
        assert len(turns) >= 1
        speakers = {t.speaker_id for t in turns}
        assert len(speakers) <= 2

    def test_returns_valid_speaker_turns(self, cfg):
        t = np.arange(16000 * 2, dtype=np.float32) / 16000
        tone = 0.3 * np.sin(2 * np.pi * 300 * t)
        pcm = _to_pcm16_bytes(tone)
        turns = diarize_pyannote(pcm, cfg)
        for turn in turns:
            assert isinstance(turn, SpeakerTurn)
            assert turn.end_ms > turn.start_ms
            assert turn.speaker_id
            assert turn.backend == "pyannote"

    def test_speaker_prefix_mapping(self, cfg):
        t = np.arange(16000 * 2, dtype=np.float32) / 16000
        tone = 0.3 * np.sin(2 * np.pi * 440 * t)
        pcm = _to_pcm16_bytes(tone)
        turns = diarize_pyannote(pcm, cfg)
        for turn in turns:
            assert turn.speaker_id.startswith("spk")
