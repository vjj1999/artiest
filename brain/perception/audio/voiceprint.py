"""
Voiceprint — 声纹提取与会话级说话人记忆

基于 SpeechBrain ECAPA-TDNN 提取声纹向量，在单次会话中自动建立说话人记忆库，
跨片段持续识别"这个人是谁"。

使用方式：
    memory = SpeakerMemory()

    # 对每段语音：
    speaker_id, confidence = memory.identify_or_register(pcm_bytes)
    # 首次出现 → 自动注册为 "speaker_1"
    # 再次出现 → 匹配到 "speaker_1", confidence=0.92
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from brain.logging import logger


@dataclass
class VoiceprintConfig:
    """声纹配置"""
    sample_rate: int = 16000
    device: str = "cuda"  # cpu / cuda
    # SpeechBrain 预训练模型（无需 HF token，首次会自动下载）
    model_id: str = "speechbrain/spkrec-ecapa-voxceleb"
    cache_dir: Optional[str] = None
    # 匹配阈值：余弦相似度 >= threshold 视为同一人
    # 远场麦克风建议 0.30~0.40；近场/高质量麦克风可用 0.55~0.70
    threshold: float = 0.35
    # 最少音频时长（秒）：太短的片段不适合提取声纹
    min_duration_s: float = 0.5
    # 注册时的最少音频时长（秒）：需要更长才能建立可靠声纹
    min_enroll_duration_s: float = 0.8


# ── SpeechBrain 模型单例 ────────────────────────────────────

_SB_MODEL = None
_SB_MODEL_ID = None


def _ensure_model(cfg: VoiceprintConfig):
    global _SB_MODEL, _SB_MODEL_ID
    if _SB_MODEL is not None and _SB_MODEL_ID == cfg.model_id:
        return

    try:
        import warnings
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            from speechbrain.inference.speaker import EncoderClassifier
    except Exception as e:
        raise RuntimeError("speechbrain 未安装或导入失败: pip install speechbrain") from e

    device = cfg.device
    if device != "cpu" and not torch.cuda.is_available():
        device = "cpu"

    savedir = cfg.cache_dir
    if savedir:
        try:
            os.makedirs(savedir, exist_ok=True)
        except Exception:
            savedir = None

    logger.info("[Voiceprint] 加载模型: %s (device=%s)", cfg.model_id, device)
    _SB_MODEL = EncoderClassifier.from_hparams(
        source=cfg.model_id,
        savedir=savedir,
        run_opts={"device": device},
    )
    _SB_MODEL_ID = cfg.model_id
    logger.info("[Voiceprint] 模型加载完成")


def _trim_silence(pcm_np: np.ndarray, sample_rate: int, threshold: float = 0.01, frame_ms: int = 30) -> np.ndarray:
    """裁掉首尾静音，保留有效语音部分（用于提高声纹质量）"""
    frame_size = int(sample_rate * frame_ms / 1000)
    if pcm_np.size < frame_size:
        return pcm_np

    # 找到第一个有声帧
    start = 0
    for i in range(0, pcm_np.size - frame_size, frame_size):
        rms = float(np.sqrt(np.mean(pcm_np[i:i + frame_size] ** 2)))
        if rms >= threshold:
            start = i
            break

    # 找到最后一个有声帧
    end = pcm_np.size
    for i in range(pcm_np.size - frame_size, start, -frame_size):
        rms = float(np.sqrt(np.mean(pcm_np[i:i + frame_size] ** 2)))
        if rms >= threshold:
            end = min(i + frame_size, pcm_np.size)
            break

    trimmed = pcm_np[start:end]
    return trimmed if trimmed.size > 0 else pcm_np


def extract_embedding(pcm_bytes: bytes, cfg: VoiceprintConfig) -> Optional[np.ndarray]:
    """
    从 PCM 音频提取声纹向量（L2 归一化）

    会先裁掉首尾静音，再提取嵌入，避免宽容期静音稀释声纹。

    Returns:
        np.ndarray shape (192,) float32，或 None（音频太短）
    """
    if not pcm_bytes:
        return None

    duration_s = len(pcm_bytes) / (cfg.sample_rate * 2)
    if duration_s < cfg.min_duration_s:
        return None

    _ensure_model(cfg)
    assert _SB_MODEL is not None

    x = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32) / 32768.0

    # 裁掉首尾静音（宽容期带来的尾部静音会严重稀释声纹）
    x = _trim_silence(x, cfg.sample_rate)

    if x.size < int(cfg.sample_rate * cfg.min_duration_s):
        return None

    wav = torch.from_numpy(x).unsqueeze(0)  # [1, T]

    with torch.inference_mode():
        emb = _SB_MODEL.encode_batch(wav, wav_lens=None)

    emb = emb.squeeze().detach().cpu().float().numpy().astype(np.float32)
    norm = float(np.linalg.norm(emb)) + 1e-9
    return emb / norm


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """余弦相似度"""
    a = a.astype(np.float32)
    b = b.astype(np.float32)
    denom = (float(np.linalg.norm(a)) * float(np.linalg.norm(b))) + 1e-9
    return float(np.dot(a, b) / denom)


# ── 会话级说话人记忆 ────────────────────────────────────────

@dataclass
class _SpeakerProfile:
    """内部：单个说话人的记忆"""
    name: str
    embedding: np.ndarray       # 当前声纹（L2 归一化）
    sample_count: int = 1       # 累计更新次数
    total_duration_s: float = 0.0


class SpeakerMemory:
    """
    会话级说话人记忆库

    在单次监听会话中，自动建立和维护说话人声纹库：
    - 首次出现的声音 → 注册为新说话人（speaker_1, speaker_2, ...）
    - 再次出现 → 根据余弦相似度匹配到已知说话人
    - 声纹向量随着新样本增量更新（移动平均），越说越准
    """

    def __init__(self, cfg: Optional[VoiceprintConfig] = None):
        self._cfg = cfg or VoiceprintConfig()
        self._speakers: Dict[str, _SpeakerProfile] = {}
        self._next_id: int = 1

    @property
    def known_speakers(self) -> List[str]:
        """已知说话人列表"""
        return list(self._speakers.keys())

    @property
    def speaker_count(self) -> int:
        return len(self._speakers)

    def identify_or_register(
        self,
        pcm_bytes: bytes,
        duration_s: float = 0.0,
    ) -> Tuple[Optional[str], float]:
        """
        识别说话人，如果是新人则自动注册。

        Args:
            pcm_bytes: 单段语音 PCM (int16, 16kHz, mono)
            duration_s: 该段时长（用于统计）

        Returns:
            (speaker_name, confidence)
            - 已知说话人: ("speaker_1", 0.92)
            - 新注册:     ("speaker_2", 1.0)
            - 太短无法识别: (None, 0.0)
        """
        cfg = self._cfg
        emb = extract_embedding(pcm_bytes, cfg)
        if emb is None:
            return None, 0.0

        if not duration_s:
            duration_s = len(pcm_bytes) / (cfg.sample_rate * 2)

        # 匹配已知说话人
        best_name, best_score = self._find_best_match(emb)

        if best_name is not None and best_score >= cfg.threshold:
            # 匹配成功：更新声纹（增量移动平均）
            self._update_speaker(best_name, emb, duration_s)
            logger.info(
                "[Voiceprint] 匹配: %s (score=%.3f, 累计 %d 次)",
                best_name, best_score, self._speakers[best_name].sample_count,
            )
            return best_name, best_score
        else:
            # 新说话人：检查是否够长来注册
            if duration_s < cfg.min_enroll_duration_s:
                logger.debug(
                    "[Voiceprint] 未匹配且音频太短(%.1fs)，不注册",
                    duration_s,
                )
                return None, best_score if best_name else 0.0

            name = self._register_new(emb, duration_s)
            logger.info(
                "[Voiceprint] 新说话人: %s (最佳匹配 score=%.3f < threshold=%.2f)",
                name, best_score if best_name else 0.0, cfg.threshold,
            )
            return name, 1.0

    def identify(self, pcm_bytes: bytes) -> Tuple[Optional[str], float]:
        """只识别不注册"""
        cfg = self._cfg
        emb = extract_embedding(pcm_bytes, cfg)
        if emb is None:
            return None, 0.0

        best_name, best_score = self._find_best_match(emb)
        if best_name is not None and best_score >= cfg.threshold:
            return best_name, best_score
        return None, best_score if best_name else 0.0

    def register(self, name: str, pcm_bytes: bytes) -> bool:
        """手动注册指定名字的说话人"""
        cfg = self._cfg
        emb = extract_embedding(pcm_bytes, cfg)
        if emb is None:
            return False

        duration_s = len(pcm_bytes) / (cfg.sample_rate * 2)
        self._speakers[name] = _SpeakerProfile(
            name=name, embedding=emb, sample_count=1, total_duration_s=duration_s,
        )
        logger.info("[Voiceprint] 手动注册: %s", name)
        return True

    def reset(self):
        """清空记忆"""
        self._speakers.clear()
        self._next_id = 1
        logger.info("[Voiceprint] 记忆已清空")

    def _find_best_match(self, emb: np.ndarray) -> Tuple[Optional[str], float]:
        best_name: Optional[str] = None
        best_score = -1.0
        scores = {}
        for name, profile in self._speakers.items():
            score = cosine_similarity(emb, profile.embedding)
            scores[name] = score
            if score > best_score:
                best_score = score
                best_name = name
        if scores:
            score_str = ", ".join(f"{k}={v:.3f}" for k, v in sorted(scores.items()))
            logger.debug("[Voiceprint] 匹配分数: %s", score_str)
        return best_name, best_score

    def _update_speaker(self, name: str, new_emb: np.ndarray, duration_s: float):
        """增量更新声纹（指数移动平均，新样本权重 = 1/(n+1)）"""
        profile = self._speakers[name]
        n = profile.sample_count
        # 加权平均：老声纹 * n/(n+1) + 新声纹 * 1/(n+1)
        alpha = 1.0 / (n + 1)
        updated = profile.embedding * (1 - alpha) + new_emb * alpha
        # 重新归一化
        norm = float(np.linalg.norm(updated)) + 1e-9
        profile.embedding = updated / norm
        profile.sample_count = n + 1
        profile.total_duration_s += duration_s

    def _register_new(self, emb: np.ndarray, duration_s: float) -> str:
        name = f"speaker_{self._next_id}"
        self._next_id += 1
        self._speakers[name] = _SpeakerProfile(
            name=name, embedding=emb, sample_count=1, total_duration_s=duration_s,
        )
        return name
