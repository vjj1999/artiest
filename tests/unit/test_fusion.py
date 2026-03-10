"""T12: Fusion Engine 单元测试"""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from services.fusion.fuser import FusionEngine, FusionConfig


def _sample_visual():
    return {
        "frame_count": 10,
        "motion_score": 0.05,
        "visual_fingerprint": [0.1] * 16,
        "duration_s": 10.0,
        "frame_timestamps_s": [float(i) for i in range(10)],
    }


def _sample_audio():
    return {
        "silence_ratio": 0.2,
        "rms_mean": 0.04,
        "zcr": 0.08,
        "duration_s": 10.0,
        "speech_segments": [{"start_s": 1.0, "end_s": 4.0}, {"start_s": 6.0, "end_s": 9.0}],
    }


def _sample_text():
    return {
        "transcript": "这是一段测试文本，请点击关注",
        "token_count": 14,
        "token_per_sec": 1.4,
        "duration_s": 10.0,
        "cta_matches": [{"keyword": "点击", "count": 1}, {"keyword": "关注", "count": 1}],
        "has_cta": True,
    }


def _sample_timeline():
    return [
        {"start_s": 0.0, "end_s": 4.0, "modalities": ["visual", "audio", "text"], "data": {}},
        {"start_s": 4.0, "end_s": 6.0, "modalities": ["visual"], "data": {}},
        {"start_s": 6.0, "end_s": 10.0, "modalities": ["visual", "audio"], "data": {}},
    ]


class TestFusionEngine:
    """验证融合引擎"""

    def test_融合embedding维度(self):
        engine = FusionEngine(FusionConfig(fusion_dim=64))
        result = engine.fuse(
            visual_data=_sample_visual(),
            audio_data=_sample_audio(),
            text_data=_sample_text(),
            timeline=_sample_timeline(),
            duration_s=10.0,
        )
        assert len(result["fusion_embedding"]) == 64
        assert result["fusion_dim"] == 64

    def test_融合embedding归一化(self):
        engine = FusionEngine()
        result = engine.fuse(
            visual_data=_sample_visual(),
            audio_data=_sample_audio(),
        )
        vec = np.array(result["fusion_embedding"])
        norm = float(np.linalg.norm(vec))
        assert abs(norm - 1.0) < 0.01

    def test_script_graph结构(self):
        engine = FusionEngine()
        result = engine.fuse(
            visual_data=_sample_visual(),
            audio_data=_sample_audio(),
            text_data=_sample_text(),
            timeline=_sample_timeline(),
            duration_s=10.0,
        )
        graph = result["script_graph"]
        assert "nodes" in graph
        assert "edges" in graph
        assert len(graph["nodes"]) == 3
        assert len(graph["edges"]) == 2

    def test_script_graph_node字段(self):
        engine = FusionEngine()
        result = engine.fuse(
            audio_data=_sample_audio(),
            timeline=[{"start_s": 0.0, "end_s": 5.0, "modalities": ["audio"], "data": {}}],
            duration_s=5.0,
        )
        node = result["script_graph"]["nodes"][0]
        for key in ["segment_id", "start_s", "end_s", "modalities", "label"]:
            assert key in node

    def test_segment分类(self):
        engine = FusionEngine()
        result = engine.fuse(
            visual_data=_sample_visual(),
            audio_data=_sample_audio(),
            text_data=_sample_text(),
            timeline=_sample_timeline(),
            duration_s=10.0,
        )
        nodes = result["script_graph"]["nodes"]
        # 第一个: visual+audio+text → narration
        assert nodes[0]["label"] == "narration"
        # 第二个: visual only
        assert nodes[1]["label"] == "visual_only"
        # 第三个: visual+audio → action
        assert nodes[2]["label"] == "action"

    def test_空输入(self):
        engine = FusionEngine()
        result = engine.fuse(duration_s=5.0)
        assert len(result["fusion_embedding"]) == 64
        assert result["script_graph"]["nodes"] == []

    def test_不同输入产生不同embedding(self):
        engine = FusionEngine()
        r1 = engine.fuse(audio_data=_sample_audio())
        r2 = engine.fuse(visual_data=_sample_visual())
        assert r1["fusion_embedding"] != r2["fusion_embedding"]

    def test_边的连续性(self):
        engine = FusionEngine()
        result = engine.fuse(
            timeline=_sample_timeline(),
            duration_s=10.0,
        )
        edges = result["script_graph"]["edges"]
        for edge in edges:
            assert edge["type"] == "sequential"
            assert edge["from"].startswith("seg_")
            assert edge["to"].startswith("seg_")
