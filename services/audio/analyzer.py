"""
Audio Service MVP — 音频分析器

对视频/音频文件执行基础音频感知：
- 静音检测 (silence_ratio)
- RMS 均值 (rms_mean)
- 过零率 (zcr)
- 语音段检测
"""
from __future__ import annotations

import os
import subprocess
import tempfile
import wave
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np

from brain.logging import logger


@dataclass
class AudioAnalyzerConfig:
    """音频分析配置"""
    sample_rate: int = 16000
    # 静音 RMS 阈值（归一化 0~1）
    silence_rms_threshold: float = 0.01
    # 帧大小（样本数）
    frame_size: int = 512
    # ffmpeg 路径（用于从视频提取音频）
    ffmpeg_path: str = "ffmpeg"


def _extract_audio_from_video(video_path: str, sample_rate: int, ffmpeg_path: str = "ffmpeg") -> Optional[bytes]:
    """用 ffmpeg 从视频提取 PCM 音频"""
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp_path = tmp.name

    try:
        cmd = [
            ffmpeg_path, "-y", "-i", video_path,
            "-vn", "-acodec", "pcm_s16le",
            "-ar", str(sample_rate), "-ac", "1",
            tmp_path,
        ]
        result = subprocess.run(cmd, capture_output=True, timeout=120)
        if result.returncode != 0:
            logger.error("[Audio] ffmpeg 失败: %s", result.stderr.decode(errors="replace")[:500])
            return None

        with wave.open(tmp_path, "rb") as wf:
            return wf.readframes(wf.getnframes())
    except Exception as e:
        logger.error("[Audio] 提取音频失败: %s", e)
        return None
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass


def _load_wav(wav_path: str, target_sr: int) -> Optional[bytes]:
    """加载 WAV 文件并重采样"""
    try:
        with wave.open(wav_path, "rb") as wf:
            orig_sr = wf.getframerate()
            n_ch = wf.getnchannels()
            raw = wf.readframes(wf.getnframes())

        pcm = np.frombuffer(raw, dtype=np.int16)
        if n_ch > 1:
            pcm = pcm[::n_ch]

        if orig_sr != target_sr:
            import scipy.signal
            num = int(len(pcm) * target_sr / orig_sr)
            pcm = np.clip(scipy.signal.resample(pcm.astype(np.float64), num), -32768, 32767).astype(np.int16)

        return pcm.tobytes()
    except Exception as e:
        logger.error("[Audio] 加载 WAV 失败: %s", e)
        return None


class AudioAnalyzer:
    """音频分析器"""

    def __init__(self, cfg: Optional[AudioAnalyzerConfig] = None):
        self._cfg = cfg or AudioAnalyzerConfig()

    def analyze_file(self, file_path: str) -> Dict[str, Any]:
        """
        分析音频/视频文件

        Returns:
            {
                "silence_ratio": float,
                "rms_mean": float,
                "zcr": float,
                "duration_s": float,
                "speech_segments": [{"start_s": float, "end_s": float}],
            }
        """
        cfg = self._cfg
        ext = os.path.splitext(file_path)[1].lower()

        # 获取 PCM 数据
        if ext in (".wav",):
            pcm_bytes = _load_wav(file_path, cfg.sample_rate)
        elif ext in (".mp4", ".mkv", ".avi", ".mov", ".flv", ".webm", ".mp3", ".aac", ".flac", ".ogg"):
            pcm_bytes = _extract_audio_from_video(file_path, cfg.sample_rate, cfg.ffmpeg_path)
        else:
            pcm_bytes = _extract_audio_from_video(file_path, cfg.sample_rate, cfg.ffmpeg_path)

        if not pcm_bytes:
            return {"error": "无法读取音频", "silence_ratio": 1.0, "rms_mean": 0.0, "zcr": 0.0, "duration_s": 0.0}

        return self.analyze_pcm(pcm_bytes)

    def analyze_pcm(self, pcm_bytes: bytes) -> Dict[str, Any]:
        """分析 PCM 数据（int16, mono, 16kHz）"""
        cfg = self._cfg
        pcm = np.frombuffer(pcm_bytes, dtype=np.int16)
        x = pcm.astype(np.float32) / 32768.0
        duration_s = len(pcm) / cfg.sample_rate

        if len(x) == 0:
            return {"silence_ratio": 1.0, "rms_mean": 0.0, "zcr": 0.0, "duration_s": 0.0, "speech_segments": []}

        # 全局 RMS
        rms_mean = float(np.sqrt(np.mean(x * x)))

        # 逐帧计算 RMS 和 ZCR
        fs = cfg.frame_size
        n_frames = max(1, len(x) // fs)
        rms_per_frame = np.zeros(n_frames)
        zcr_per_frame = np.zeros(n_frames)

        for i in range(n_frames):
            frame = x[i * fs:(i + 1) * fs]
            if len(frame) == 0:
                continue
            rms_per_frame[i] = float(np.sqrt(np.mean(frame * frame)))
            # 过零率：相邻样本符号变化的比例
            signs = np.sign(frame)
            zcr_per_frame[i] = float(np.mean(np.abs(np.diff(signs)) > 0)) if len(signs) > 1 else 0.0

        zcr = float(np.mean(zcr_per_frame))

        # 静音检测：RMS 低于阈值的帧
        silence_mask = rms_per_frame < cfg.silence_rms_threshold
        silence_ratio = float(np.mean(silence_mask))

        # 语音段提取（连续非静音帧合并）
        speech_segments = self._extract_speech_segments(silence_mask, fs, cfg.sample_rate)

        logger.info(
            "[Audio] 分析完成: duration=%.1fs, silence=%.1f%%, rms=%.4f, zcr=%.4f",
            duration_s, silence_ratio * 100, rms_mean, zcr,
        )

        return {
            "silence_ratio": round(silence_ratio, 4),
            "rms_mean": round(rms_mean, 6),
            "zcr": round(zcr, 6),
            "duration_s": round(duration_s, 3),
            "speech_segments": speech_segments,
        }

    @staticmethod
    def _extract_speech_segments(
        silence_mask: np.ndarray,
        frame_size: int,
        sample_rate: int,
    ) -> List[Dict[str, float]]:
        """从静音掩码提取语音段"""
        segments: List[Dict[str, float]] = []
        in_speech = False
        start_frame = 0

        for i, is_silent in enumerate(silence_mask):
            if not is_silent and not in_speech:
                in_speech = True
                start_frame = i
            elif is_silent and in_speech:
                in_speech = False
                start_s = start_frame * frame_size / sample_rate
                end_s = i * frame_size / sample_rate
                if end_s - start_s > 0.1:
                    segments.append({"start_s": round(start_s, 3), "end_s": round(end_s, 3)})

        # 最后一段
        if in_speech:
            start_s = start_frame * frame_size / sample_rate
            end_s = len(silence_mask) * frame_size / sample_rate
            if end_s - start_s > 0.1:
                segments.append({"start_s": round(start_s, 3), "end_s": round(end_s, 3)})

        return segments
