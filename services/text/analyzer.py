"""
Text Service MVP — ASR 转写 + 文本指标

对音频执行 ASR 转写并计算文本指标：
- ASR 转写文本（基于 funasr AutoModel，内置 VAD + 标点）
- 语速 (token_per_sec)
- CTA 关键词检测
"""
from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

from brain.logging import logger


@dataclass
class TextAnalyzerConfig:
    """文本分析配置"""
    sample_rate: int = 16000
    # funasr AutoModel 参数（内置 VAD + 标点，支持长音频）
    asr_model: str = "paraformer-zh"
    vad_model: str = "fsmn-vad"
    punc_model: str = "ct-punc"
    batch_size_s: int = 300
    # CTA 关键词列表
    cta_keywords: List[str] = field(default_factory=lambda: [
        "点击", "关注", "订阅", "链接", "购买", "下单",
        "扫码", "优惠", "限时", "免费", "领取", "加入",
    ])
    # 禁用 ASR（仅做文本分析时使用）
    disable_asr: bool = False


_asr_model = None


def _ensure_asr(cfg: TextAnalyzerConfig):
    """加载 funasr AutoModel（paraformer-zh + fsmn-vad + ct-punc，单例）"""
    global _asr_model
    if _asr_model is not None:
        return
    try:
        from funasr import AutoModel
        logger.info("[Text] 加载离线 ASR: %s + %s + %s",
                    cfg.asr_model, cfg.vad_model, cfg.punc_model)
        _asr_model = AutoModel(
            model=cfg.asr_model,
            vad_model=cfg.vad_model,
            punc_model=cfg.punc_model,
            disable_update=True,
        )
        logger.info("[Text] ASR 模型加载完成")
    except Exception as e:
        logger.error("[Text] ASR 模型加载失败: %s", e)
        raise


def _asr_recognize(pcm_float: np.ndarray, cfg: TextAnalyzerConfig) -> str:
    """执行 ASR 识别（funasr AutoModel，内置 VAD 分段 + 标点）"""
    if _asr_model is None:
        return ""
    try:
        result = _asr_model.generate(
            input=pcm_float,
            batch_size_s=cfg.batch_size_s,
        )
        return result[0].get("text", "") if result and len(result) > 0 else ""
    except Exception as e:
        logger.error("[Text] ASR 识别失败: %s", e)
        return ""


def _count_tokens(text: str) -> int:
    """统计中文字符 + 英文单词数"""
    chinese = len(re.findall(r'[\u4e00-\u9fff]', text))
    english = len(re.findall(r'[a-zA-Z]+', text))
    return chinese + english


def _detect_cta(text: str, keywords: List[str]) -> List[Dict[str, Any]]:
    """检测 CTA 关键词"""
    matches = []
    for kw in keywords:
        positions = [m.start() for m in re.finditer(re.escape(kw), text)]
        if positions:
            matches.append({"keyword": kw, "count": len(positions), "positions": positions})
    return matches


class TextAnalyzer:
    """文本分析器"""

    def __init__(self, cfg: Optional[TextAnalyzerConfig] = None):
        self._cfg = cfg or TextAnalyzerConfig()

    def analyze_pcm(self, pcm_bytes: bytes, duration_s: Optional[float] = None) -> Dict[str, Any]:
        """
        对 PCM 音频执行 ASR + 文本分析

        Args:
            pcm_bytes: int16 mono 16kHz PCM
            duration_s: 音频时长（可选，自动计算）

        Returns:
            {
                "transcript": str,
                "token_count": int,
                "token_per_sec": float,
                "duration_s": float,
                "cta_matches": [...],
                "has_cta": bool,
            }
        """
        cfg = self._cfg
        pcm = np.frombuffer(pcm_bytes, dtype=np.int16)
        dur = duration_s or (len(pcm) / cfg.sample_rate)

        if dur <= 0 or len(pcm) == 0:
            return self._empty_result(0.0)

        # ASR（funasr AutoModel 接受 float32）
        transcript = ""
        if not cfg.disable_asr:
            try:
                _ensure_asr(cfg)
                pcm_float = pcm.astype(np.float32) / 32768.0
                transcript = _asr_recognize(pcm_float, cfg)
            except Exception as e:
                logger.warning("[Text] ASR 不可用: %s", e)

        return self.analyze_text(transcript, dur)

    def analyze_text(self, transcript: str, duration_s: float) -> Dict[str, Any]:
        """
        对已有文本执行指标分析（不走 ASR）

        可用于：1) ASR 在外部完成后传入 2) 纯文本分析测试
        """
        cfg = self._cfg

        token_count = _count_tokens(transcript)
        token_per_sec = token_count / duration_s if duration_s > 0 else 0.0
        cta_matches = _detect_cta(transcript, cfg.cta_keywords)

        logger.info(
            "[Text] 分析完成: tokens=%d, tps=%.1f, cta=%d",
            token_count, token_per_sec, len(cta_matches),
        )

        return {
            "transcript": transcript,
            "token_count": token_count,
            "token_per_sec": round(token_per_sec, 2),
            "duration_s": round(duration_s, 3),
            "cta_matches": cta_matches,
            "has_cta": len(cta_matches) > 0,
        }

    @staticmethod
    def _empty_result(duration_s: float) -> Dict[str, Any]:
        return {
            "transcript": "",
            "token_count": 0,
            "token_per_sec": 0.0,
            "duration_s": round(duration_s, 3),
            "cta_matches": [],
            "has_cta": False,
        }
