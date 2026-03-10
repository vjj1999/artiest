"""
VLM 场景描述器 — 使用 Qwen2-VL 为分镜关键帧生成场景描述

对每个 shot 的 keyframe 调用视觉语言模型，输出一句中文场景描述。
模型懒加载，首次调用时才初始化。
"""
from __future__ import annotations

import os
import subprocess
import tempfile
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from brain.logging import logger


@dataclass
class SceneDescriberConfig:
    """VLM 场景描述配置"""
    # 模型路径（本地路径或 HuggingFace/ModelScope ID）
    model_id: str = r"C:\Users\Administrator\.cache\modelscope\hub\models\Qwen\Qwen2-VL-2B-Instruct"
    # 推理设备
    device: str = "cuda"
    # 生成参数
    max_new_tokens: int = 200
    # 提取关键帧的宽度
    frame_width: int = 640
    # ffmpeg 路径
    ffmpeg_path: str = "ffmpeg"
    # 提示词
    prompt: str = (
        "请用中文一句话描述这张画面的内容，包含：主体、动作、场景、关键物体。"
        "不要编造品牌名或文字。"
    )


_vlm_model = None
_vlm_processor = None
_vlm_tokenizer = None


def _ensure_vlm(cfg: SceneDescriberConfig):
    """懒加载 Qwen2-VL 模型"""
    global _vlm_model, _vlm_processor, _vlm_tokenizer
    if _vlm_model is not None:
        return

    import torch
    from transformers import (
        Qwen2VLForConditionalGeneration,
        Qwen2VLProcessor,
        Qwen2VLImageProcessor,
        AutoTokenizer,
    )
    from transformers.models.qwen2_vl import Qwen2VLVideoProcessor

    logger.info("[SceneDescriber] 加载 VLM: %s", cfg.model_id)

    _vlm_model = Qwen2VLForConditionalGeneration.from_pretrained(
        cfg.model_id,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    _vlm_tokenizer = AutoTokenizer.from_pretrained(cfg.model_id)
    # 手动组装 processor（规避 transformers 5.x AutoVideoProcessor/chat_template bug）
    _img_proc = Qwen2VLImageProcessor.from_pretrained(cfg.model_id)
    _vid_proc = Qwen2VLVideoProcessor.from_pretrained(cfg.model_id)
    _vlm_processor = Qwen2VLProcessor(
        image_processor=_img_proc,
        video_processor=_vid_proc,
        tokenizer=_vlm_tokenizer,
        chat_template=_vlm_tokenizer.chat_template,
    )

    logger.info("[SceneDescriber] VLM 加载完成")


def _extract_keyframe_jpeg(
    video_path: str,
    timestamp_s: float,
    width: int = 640,
    ffmpeg_path: str = "ffmpeg",
) -> Optional[str]:
    """用 ffmpeg 在指定时间戳提取一帧 JPEG，返回临时文件路径"""
    tmp = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False, prefix="brain_kf_")
    tmp_path = tmp.name
    tmp.close()
    try:
        cmd = [
            ffmpeg_path, "-y",
            "-ss", f"{timestamp_s:.3f}",
            "-i", video_path,
            "-vframes", "1",
            "-vf", f"scale={width}:-1",
            "-q:v", "2",
            tmp_path,
        ]
        result = subprocess.run(cmd, capture_output=True, timeout=30)
        if result.returncode != 0 or not os.path.exists(tmp_path) or os.path.getsize(tmp_path) == 0:
            _safe_remove(tmp_path)
            return None
        return tmp_path
    except Exception as e:
        logger.error("[SceneDescriber] 关键帧提取失败 @%.1fs: %s", timestamp_s, e)
        _safe_remove(tmp_path)
        return None


def _safe_remove(path: str):
    try:
        if path and os.path.exists(path):
            os.remove(path)
    except Exception:
        pass


def _describe_image(image_path: str, cfg: SceneDescriberConfig) -> str:
    """对单张图片调用 VLM 生成描述"""
    import torch
    from qwen_vl_utils import process_vision_info

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": f"file:///{image_path.replace(os.sep, '/')}"},
                {"type": "text", "text": cfg.prompt},
            ],
        }
    ]

    text = _vlm_processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
    )
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = _vlm_processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to(_vlm_model.device)

    with torch.no_grad():
        output_ids = _vlm_model.generate(**inputs, max_new_tokens=cfg.max_new_tokens)
    # 截去 prompt 部分
    generated = output_ids[:, inputs.input_ids.shape[1]:]
    result = _vlm_processor.batch_decode(
        generated, skip_special_tokens=True, clean_up_tokenization_spaces=False,
    )
    return (result[0] if result else "").strip()


class SceneDescriber:
    """VLM 场景描述器"""

    def __init__(self, cfg: Optional[SceneDescriberConfig] = None):
        self._cfg = cfg or SceneDescriberConfig()

    def describe_shots(
        self,
        video_path: str,
        shots: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """
        为每个 shot 的关键帧生成场景描述。

        在 shot dict 中添加 'scene_description' 字段。
        返回更新后的 shots 列表。
        """
        cfg = self._cfg

        if not shots:
            return shots

        _ensure_vlm(cfg)

        temp_files: List[str] = []
        try:
            for i, shot in enumerate(shots):
                keyframe_s = shot.get("keyframe_s", 0)
                kf_path = _extract_keyframe_jpeg(
                    video_path, keyframe_s, cfg.frame_width, cfg.ffmpeg_path,
                )
                if kf_path is None:
                    shot["scene_description"] = ""
                    logger.warning(
                        "[SceneDescriber] Shot #%d 关键帧提取失败 @%.1fs",
                        shot.get("shot_id", i), keyframe_s,
                    )
                    continue

                temp_files.append(kf_path)

                try:
                    desc = _describe_image(kf_path, cfg)
                    shot["scene_description"] = desc
                    logger.info(
                        "[SceneDescriber] Shot #%d @%.1fs: %s",
                        shot.get("shot_id", i), keyframe_s, desc[:60],
                    )
                except Exception as e:
                    logger.error(
                        "[SceneDescriber] Shot #%d VLM 推理失败: %s",
                        shot.get("shot_id", i), e,
                    )
                    shot["scene_description"] = ""
        finally:
            for f in temp_files:
                _safe_remove(f)

        return shots
