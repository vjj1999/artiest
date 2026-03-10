from __future__ import annotations

import os
from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import torch
import torchaudio

from brain.perception.audio.types import SpeakerTurn
from brain.logging import logger


@dataclass(frozen=True)
class PyannoteDiarizationConfig:
    """pyannote diarization 配置（生产可用：质量更好，代价更高）。"""

    sample_rate: int = 16000

    # 模型来源（二选一）
    model_id: str = "pyannote/speaker-diarization-3.1"
    local_path: Optional[str] = None  # 指向本地 pipeline/config 目录或文件（离线部署）

    # HuggingFace token（可选：也可用环境变量 HUGGINGFACE_TOKEN/HF_TOKEN）
    hf_token: Optional[str] = None
    hf_home: Optional[str] = None  # 例如 D:/models/hf_cache

    # 推理设备
    device: str = "cpu"  # cpu/cuda

    # 输出映射
    speaker_prefix: str = "spk"


_PIPELINE = None
_PIPELINE_ID = None


def _get_token(cfg: PyannoteDiarizationConfig) -> Optional[str]:
    t = (cfg.hf_token or "").strip()
    if t:
        return t
    return (os.getenv("HUGGINGFACE_TOKEN") or os.getenv("HF_TOKEN") or "").strip() or None


def _ensure_pipeline(cfg: PyannoteDiarizationConfig):
    global _PIPELINE, _PIPELINE_ID
    key = cfg.local_path or cfg.model_id
    if _PIPELINE is not None and _PIPELINE_ID == key:
        return

    # 延迟导入：避免未安装 pyannote 时导入失败影响默认链路
    try:
        import warnings
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            from pyannote.audio import Pipeline  # type: ignore
    except Exception as e:
        raise RuntimeError("pyannote.audio 未安装或导入失败") from e

    if cfg.hf_home:
        try:
            os.makedirs(cfg.hf_home, exist_ok=True)
            os.environ.setdefault("HF_HOME", cfg.hf_home)
            os.environ.setdefault("HUGGINGFACE_HUB_CACHE", os.path.join(cfg.hf_home, "hub"))
        except Exception as e:
            logger.warning("[pyannote] 设置 HF_HOME 失败（忽略）: %s", e)

    token = _get_token(cfg)
    src = cfg.local_path or cfg.model_id
    # pyannote.audio 4.x uses `token=...` (huggingface_hub 新接口)
    # 兼容旧版：若不支持 token 参数，再回退 use_auth_token
    try:
        pipe = Pipeline.from_pretrained(src, token=token)
    except TypeError:
        pipe = Pipeline.from_pretrained(src, use_auth_token=token)
    _PIPELINE = pipe
    _PIPELINE_ID = key


def diarize_pyannote(pcm_bytes: bytes, cfg: PyannoteDiarizationConfig) -> List[SpeakerTurn]:
    """返回 utterance 内的说话人 turn（ms）。"""
    if not pcm_bytes:
        return []

    _ensure_pipeline(cfg)
    assert _PIPELINE is not None

    device = cfg.device
    if device != "cpu" and not torch.cuda.is_available():
        device = "cpu"

    # pyannote expects float waveform tensor
    x = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32) / 32768.0
    wav = torch.from_numpy(x).unsqueeze(0)  # [1, T]

    # pyannote models generally use 16k; resample if needed
    if cfg.sample_rate != 16000:
        wav = torchaudio.functional.resample(wav, orig_freq=cfg.sample_rate, new_freq=16000)
        sr = 16000
    else:
        sr = cfg.sample_rate

    try:
        _PIPELINE.to(torch.device(device))
    except Exception:
        # 某些版本 pipeline 没有 .to；不强制
        pass

    # pyannote 4.x 返回 DiarizeOutput，需要 .speaker_diarization 获取 Annotation
    import warnings
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        result = _PIPELINE({"waveform": wav, "sample_rate": sr})

    # 兼容 pyannote 3.x（直接返回 Annotation）和 4.x（返回 DiarizeOutput）
    diar = getattr(result, "speaker_diarization", result)

    turns: list[SpeakerTurn] = []
    for seg, _track, label in diar.itertracks(yield_label=True):
        start_ms = int(float(seg.start) * 1000)
        end_ms = int(float(seg.end) * 1000)
        if end_ms <= start_ms:
            continue
        speaker_id = str(label)
        # 统一输出前缀（可选）：把 SPEAKER_00 → spk0/1...
        if cfg.speaker_prefix and speaker_id.upper().startswith("SPEAKER_"):
            try:
                idx = int(speaker_id.split("_")[-1])
                speaker_id = f"{cfg.speaker_prefix}{idx}"
            except Exception:
                pass
        turns.append(SpeakerTurn(start_ms=start_ms, end_ms=end_ms, speaker_id=speaker_id, backend="pyannote"))

    if not turns:
        # 兜底：整段单 speaker
        dur_ms = int(len(pcm_bytes) / (cfg.sample_rate * 2) * 1000)
        turns = [SpeakerTurn(start_ms=0, end_ms=dur_ms, speaker_id=f"{cfg.speaker_prefix}0", backend="pyannote")]
    return turns
