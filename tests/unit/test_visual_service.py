"""T9: Visual Service MVP 单元测试"""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from services.visual.analyzer import (
    VisualAnalyzer,
    _compute_motion_score,
    _compute_fingerprint,
)


def _gen_frame(h: int = 64, w: int = 80, value: int = 128) -> np.ndarray:
    """生成灰度帧"""
    return np.full((h, w), value, dtype=np.uint8)


def _gen_gradient_frame(h: int = 64, w: int = 80, offset: int = 0) -> np.ndarray:
    """生成渐变帧（模拟运动）"""
    row = np.arange(w, dtype=np.uint8) + offset
    return np.tile(row, (h, 1))


class TestMotionScore:
    """验证运动评分"""

    def test_静止画面为零(self):
        frames = [_gen_frame(value=100)] * 5
        assert _compute_motion_score(frames) == 0.0

    def test_运动画面大于零(self):
        frames = [_gen_gradient_frame(offset=i * 30) for i in range(5)]
        score = _compute_motion_score(frames)
        assert score > 0

    def test_单帧返回零(self):
        assert _compute_motion_score([_gen_frame()]) == 0.0

    def test_空帧返回零(self):
        assert _compute_motion_score([]) == 0.0


class TestFingerprint:
    """验证视觉指纹"""

    def test_维度正确(self):
        frames = [_gen_frame()] * 3
        fp = _compute_fingerprint(frames, n_bins=16)
        assert len(fp) == 16

    def test_归一化(self):
        frames = [_gen_frame()] * 3
        fp = _compute_fingerprint(frames)
        assert abs(sum(fp) - 1.0) < 0.01

    def test_空帧返回零向量(self):
        fp = _compute_fingerprint([], n_bins=8)
        assert len(fp) == 8
        assert all(v == 0.0 for v in fp)

    def test_不同帧产生不同指纹(self):
        dark = [np.zeros((64, 80), dtype=np.uint8)] * 3
        bright = [np.full((64, 80), 255, dtype=np.uint8)] * 3
        fp_dark = _compute_fingerprint(dark)
        fp_bright = _compute_fingerprint(bright)
        assert fp_dark != fp_bright


class TestVisualAnalyzer:
    """验证 VisualAnalyzer 直接帧分析"""

    def test_analyze_frames(self):
        analyzer = VisualAnalyzer()
        frames = [_gen_gradient_frame(offset=i * 20) for i in range(10)]
        result = analyzer.analyze_frames(frames)
        assert result["frame_count"] == 10
        assert result["motion_score"] > 0
        assert len(result["visual_fingerprint"]) == 16

    def test_输出字段完整(self):
        analyzer = VisualAnalyzer()
        frames = [_gen_frame()]
        result = analyzer.analyze_frames(frames)
        for key in ["frame_count", "motion_score", "visual_fingerprint"]:
            assert key in result
