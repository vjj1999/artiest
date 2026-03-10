"""
Voice Activity Detection — 基于 Silero VAD

实时检测音频流中的语音起止，输出语音片段。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Callable

import numpy as np
import torch

from brain.logging import logger


@dataclass
class VADConfig:
    """Silero VAD 配置"""
    sample_rate: int = 16000
    # VAD 检测窗口大小（样本数）：16kHz 用 512，8kHz 用 256
    window_size_samples: int = 512
    # 静音宽容期（秒）：VAD 报告 end 后，等这么久再真正结束（合并短停顿）
    silence_grace_s: float = 1.0
    # 最大静音时长（秒）：超过此时长强制结束（防止一直等）
    max_silence_s: float = 3.0
    # 最短语音时长（秒），过滤短促噪声
    min_speech_duration_s: float = 0.8
    # RMS 能量阈值（0~1 归一化）：低于此值的片段视为远场噪声丢弃，0 表示不启用
    min_rms: float = 0.08
    # 使用 ONNX 加速
    onnx: bool = True


@dataclass
class SpeechSegment:
    """检测到的语音片段"""
    pcm_bytes: bytes = b""
    start_time_s: float = 0.0
    end_time_s: float = 0.0

    @property
    def duration_s(self) -> float:
        return self.end_time_s - self.start_time_s


class SileroVAD:
    """
    Silero VAD 封装

    使用方式（流式）：
        vad = SileroVAD()
        for chunk in audio_stream:
            segments = vad.feed(chunk_bytes)
            for segment in segments:
                process(segment)
        # 音频结束时 flush 缓冲区
        last = vad.flush()
        if last:
            process(last)
    """

    def __init__(self, cfg: Optional[VADConfig] = None):
        self._cfg = cfg or VADConfig()
        self._model = None
        self._vad_iterator = None
        self._utils = None
        self._ensure_model()

        # 状态
        self._is_speech = False
        self._in_grace_period = False  # 静音宽容期内（等待是否恢复说话）
        self._speech_buffer = bytearray()
        self._grace_silence_samples = 0  # 宽容期内累计的静音样本数
        self._speech_start_sample = 0
        self._total_samples = 0

    def _ensure_model(self):
        """加载 Silero VAD 模型"""
        if self._model is not None:
            return

        logger.info("[VAD] 加载 Silero VAD (onnx=%s)...", self._cfg.onnx)
        self._model, self._utils = torch.hub.load(
            repo_or_dir='snakers4/silero-vad',
            model='silero_vad',
            force_reload=False,
            onnx=self._cfg.onnx,
        )

        (self._get_speech_timestamps,
         self._save_audio,
         self._read_audio,
         self._VADIterator,
         self._collect_chunks) = self._utils

        self._vad_iterator = self._VADIterator(
            self._model,
            sampling_rate=self._cfg.sample_rate,
        )
        logger.info("[VAD] 模型加载完成")

    def flush(self) -> Optional[SpeechSegment]:
        """
        强制结束当前正在收集的语音片段并返回。

        在音频流结束时调用，确保缓冲区中的语音不会丢失。
        """
        if self._is_speech or self._in_grace_period:
            if self._speech_buffer:
                logger.debug("[VAD] flush: 输出缓冲区中的语音片段")
                seg = self._finalize_segment()
                if self._vad_iterator:
                    self._vad_iterator.reset_states()
                return seg
        return None

    def reset(self):
        """重置 VAD 状态（丢弃缓冲区，不输出）"""
        self._is_speech = False
        self._in_grace_period = False
        self._speech_buffer = bytearray()
        self._grace_silence_samples = 0
        self._speech_start_sample = 0
        if self._vad_iterator:
            self._vad_iterator.reset_states()

    def feed(self, pcm_bytes: bytes) -> list[SpeechSegment]:
        """
        喂入 PCM 音频数据，检测语音活动。

        核心逻辑：
        - 检测到语音开始 → 开始收集音频
        - 检测到语音结束 → 进入「宽容期」，继续收集音频
        - 宽容期内又检测到语音 → 取消宽容期，合并为同一段
        - 宽容期超时（silence_grace_s） → 真正结束，输出片段

        Args:
            pcm_bytes: int16 PCM 音频数据（单声道）

        Returns:
            检测到的完整语音片段列表（可能为空）。
        """
        cfg = self._cfg
        ws = cfg.window_size_samples
        grace_samples = int(cfg.silence_grace_s * cfg.sample_rate)
        max_silence_samples = int(cfg.max_silence_s * cfg.sample_rate)

        pcm_np = np.frombuffer(pcm_bytes, dtype=np.int16)
        self._total_samples += len(pcm_np)

        segments: list[SpeechSegment] = []
        for i in range(0, len(pcm_np), ws):
            chunk_np = pcm_np[i:i + ws]
            if len(chunk_np) < ws:
                if self._is_speech or self._in_grace_period:
                    self._speech_buffer.extend(chunk_np.tobytes())
                    if self._in_grace_period:
                        self._grace_silence_samples += len(chunk_np)
                break

            chunk_tensor = torch.from_numpy(chunk_np.astype(np.float32)) / 32767.0
            speech_dict = self._vad_iterator(chunk_tensor, return_seconds=True)

            # 正在说话或宽容期内：都要收集音频
            if self._is_speech or self._in_grace_period:
                self._speech_buffer.extend(chunk_np.tobytes())

            if speech_dict is not None and 'start' in speech_dict:
                if not self._is_speech and not self._in_grace_period:
                    # 全新的语音开始
                    self._is_speech = True
                    self._in_grace_period = False
                    self._grace_silence_samples = 0
                    self._speech_start_sample = self._total_samples - len(pcm_np) + i
                    self._speech_buffer = bytearray(chunk_np.tobytes())
                    logger.debug("[VAD] 语音开始")
                elif self._in_grace_period:
                    # 宽容期内又说话了 → 合并，继续收集
                    self._is_speech = True
                    self._in_grace_period = False
                    self._grace_silence_samples = 0
                    logger.debug("[VAD] 宽容期内恢复说话，合并")

            if speech_dict is not None and 'end' in speech_dict:
                if self._is_speech:
                    # 不立即结束，进入宽容期
                    self._is_speech = False
                    self._in_grace_period = True
                    self._grace_silence_samples = 0
                    logger.debug("[VAD] 语音暂停，进入宽容期 (%.1fs)", cfg.silence_grace_s)
                    self._vad_iterator.reset_states()

            # 宽容期计时
            if self._in_grace_period and speech_dict is None:
                self._grace_silence_samples += len(chunk_np)
                if self._grace_silence_samples >= grace_samples:
                    logger.debug("[VAD] 宽容期超时，语音结束")
                    seg = self._finalize_segment()
                    if seg:
                        segments.append(seg)
                    self._vad_iterator.reset_states()

            # 极长静音保护（说话中间长时间无 end 事件但也无 start）
            if not self._is_speech and not self._in_grace_period:
                pass  # 正常空闲
            elif self._in_grace_period and self._grace_silence_samples >= max_silence_samples:
                logger.debug("[VAD] 最大静音超时，强制结束")
                seg = self._finalize_segment()
                if seg:
                    segments.append(seg)
                self._vad_iterator.reset_states()

        return segments

    def _finalize_segment(self) -> Optional[SpeechSegment]:
        """将缓冲区中的语音打包为 SpeechSegment"""
        cfg = self._cfg
        pcm = bytes(self._speech_buffer)
        duration_s = len(pcm) / (cfg.sample_rate * 2)

        # 重置状态
        self._is_speech = False
        self._in_grace_period = False
        self._speech_buffer = bytearray()
        self._grace_silence_samples = 0

        # 过滤：时长太短
        if duration_s < cfg.min_speech_duration_s:
            logger.debug("[VAD] 丢弃短片段: %.2fs < %.2fs", duration_s, cfg.min_speech_duration_s)
            return None

        # 过滤：能量太低（远场小声/背景噪声）
        if cfg.min_rms > 0 and pcm:
            x = np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0
            rms = float(np.sqrt(np.mean(x * x))) if x.size else 0.0
            if rms < cfg.min_rms:
                logger.debug("[VAD] 丢弃低能量片段: RMS=%.4f < %.4f (%.2fs)", rms, cfg.min_rms, duration_s)
                return None
            logger.debug("[VAD] RMS=%.4f (阈值 %.4f)", rms, cfg.min_rms)

        start_s = self._speech_start_sample / cfg.sample_rate
        end_s = start_s + duration_s

        logger.info("[VAD] 语音片段: %.2fs - %.2fs (%.2fs)", start_s, end_s, duration_s)
        return SpeechSegment(pcm_bytes=pcm, start_time_s=start_s, end_time_s=end_s)
