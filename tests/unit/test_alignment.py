"""T11: Time Alignment Service 单元测试"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from services.align.aligner import TimeAligner, AlignConfig


class TestTimeAligner:
    """验证时间对齐逻辑"""

    def test_纯音频对齐(self):
        aligner = TimeAligner()
        audio_data = {
            "speech_segments": [
                {"start_s": 1.0, "end_s": 3.0},
                {"start_s": 5.0, "end_s": 8.0},
            ]
        }
        result = aligner.align(audio_data=audio_data, duration_s=10.0)
        assert len(result["timeline"]) == 2
        assert result["timeline"][0]["modalities"] == ["audio"]

    def test_视觉音频对齐(self):
        aligner = TimeAligner()
        visual_data = {
            "frame_timestamps_s": [0.0, 1.0, 2.0, 3.0],
        }
        audio_data = {
            "speech_segments": [{"start_s": 0.5, "end_s": 2.5}]
        }
        result = aligner.align(visual_data=visual_data, audio_data=audio_data, duration_s=4.0)
        # 帧和语音段应该合并（因为重叠）
        assert len(result["timeline"]) >= 1
        # 至少有一个包含两种模态的 segment
        multi = [s for s in result["timeline"] if len(s["modalities"]) > 1]
        assert len(multi) >= 1

    def test_全模态对齐(self):
        aligner = TimeAligner()
        result = aligner.align(
            visual_data={"frame_timestamps_s": [0.0, 1.0, 2.0]},
            audio_data={"speech_segments": [{"start_s": 0.0, "end_s": 3.0}]},
            text_data={"transcript": "测试文本", "duration_s": 3.0},
            duration_s=3.0,
        )
        assert "timeline" in result
        assert "modality_coverage" in result
        # 所有模态都应有覆盖
        cov = result["modality_coverage"]
        assert cov["visual"] > 0
        assert cov["audio"] > 0
        assert cov["text"] > 0

    def test_空输入(self):
        aligner = TimeAligner()
        result = aligner.align(duration_s=5.0)
        assert result["timeline"] == []
        assert result["duration_s"] == 5.0

    def test_coverage计算(self):
        aligner = TimeAligner()
        audio_data = {"speech_segments": [{"start_s": 0.0, "end_s": 5.0}]}
        result = aligner.align(audio_data=audio_data, duration_s=10.0)
        assert result["modality_coverage"]["audio"] == 0.5

    def test_segment时间戳有序(self):
        aligner = TimeAligner()
        audio_data = {
            "speech_segments": [
                {"start_s": 5.0, "end_s": 8.0},
                {"start_s": 1.0, "end_s": 3.0},
            ]
        }
        result = aligner.align(audio_data=audio_data, duration_s=10.0)
        times = [s["start_s"] for s in result["timeline"]]
        assert times == sorted(times)

    def test_输出字段完整(self):
        aligner = TimeAligner()
        audio_data = {"speech_segments": [{"start_s": 0.0, "end_s": 1.0}]}
        result = aligner.align(audio_data=audio_data, duration_s=2.0)
        assert "timeline" in result
        assert "duration_s" in result
        assert "modality_coverage" in result
        seg = result["timeline"][0]
        for key in ["start_s", "end_s", "duration_s", "modalities", "data"]:
            assert key in seg
