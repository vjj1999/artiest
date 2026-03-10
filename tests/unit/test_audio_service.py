"""T6: Audio Service MVP 单元测试"""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from services.audio.analyzer import AudioAnalyzer, AudioAnalyzerConfig


def _gen_pcm_silence(duration_s: float, sample_rate: int = 16000) -> bytes:
    """生成静音 PCM"""
    n = int(duration_s * sample_rate)
    return np.zeros(n, dtype=np.int16).tobytes()


def _gen_pcm_tone(duration_s: float, freq: float = 440.0, amplitude: float = 0.5, sample_rate: int = 16000) -> bytes:
    """生成正弦波 PCM"""
    n = int(duration_s * sample_rate)
    t = np.arange(n) / sample_rate
    wave = (amplitude * np.sin(2 * np.pi * freq * t) * 32767).astype(np.int16)
    return wave.tobytes()


class TestAudioAnalyzer:
    """验证 AudioAnalyzer 核心逻辑"""

    def test_静音检测(self):
        analyzer = AudioAnalyzer()
        pcm = _gen_pcm_silence(2.0)
        result = analyzer.analyze_pcm(pcm)
        assert result["silence_ratio"] >= 0.95
        assert result["rms_mean"] < 0.001

    def test_有声音频(self):
        analyzer = AudioAnalyzer()
        pcm = _gen_pcm_tone(2.0, freq=440, amplitude=0.5)
        result = analyzer.analyze_pcm(pcm)
        assert result["silence_ratio"] < 0.2
        assert result["rms_mean"] > 0.1
        assert result["zcr"] > 0

    def test_duration正确(self):
        analyzer = AudioAnalyzer()
        pcm = _gen_pcm_tone(3.0)
        result = analyzer.analyze_pcm(pcm)
        assert abs(result["duration_s"] - 3.0) < 0.01

    def test_speech_segments提取(self):
        analyzer = AudioAnalyzer()
        # 1s 静音 + 2s 声音 + 1s 静音
        silence = _gen_pcm_silence(1.0)
        tone = _gen_pcm_tone(2.0)
        pcm = silence + tone + silence
        result = analyzer.analyze_pcm(pcm)
        assert len(result["speech_segments"]) >= 1
        seg = result["speech_segments"][0]
        assert seg["start_s"] >= 0.5
        assert seg["end_s"] <= 3.5

    def test_空数据(self):
        analyzer = AudioAnalyzer()
        result = analyzer.analyze_pcm(b"")
        assert result["silence_ratio"] == 1.0
        assert result["rms_mean"] == 0.0

    def test_输出字段完整(self):
        analyzer = AudioAnalyzer()
        pcm = _gen_pcm_tone(1.0)
        result = analyzer.analyze_pcm(pcm)
        for key in ["silence_ratio", "rms_mean", "zcr", "duration_s", "speech_segments"]:
            assert key in result, f"缺少输出字段: {key}"

    def test_zcr对不同信号的区分(self):
        analyzer = AudioAnalyzer()
        # 低频信号 ZCR 低，高频信号 ZCR 高
        low_freq = _gen_pcm_tone(1.0, freq=100, amplitude=0.5)
        high_freq = _gen_pcm_tone(1.0, freq=4000, amplitude=0.5)
        r_low = analyzer.analyze_pcm(low_freq)
        r_high = analyzer.analyze_pcm(high_freq)
        assert r_high["zcr"] > r_low["zcr"]
