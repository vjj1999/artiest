"""
Visual Service MVP — 视觉分析器

对视频文件执行视觉感知：
- 关键帧提取
- 运动评分 (motion_score)
- 视觉指纹 (visual_fingerprint)
- 场景切分检测 (scene_cuts)
"""
from __future__ import annotations

import os
import re as _re
import subprocess
import tempfile
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np

from brain.logging import logger


@dataclass
class VisualAnalyzerConfig:
    """视觉分析配置"""
    # 抽帧间隔（秒）
    frame_interval_s: float = 1.0
    # 最大抽帧数
    max_frames: int = 60
    # 帧尺寸（缩放到固定宽度，保持比例）
    frame_width: int = 320
    # ffmpeg/ffprobe 路径
    ffmpeg_path: str = "ffmpeg"
    ffprobe_path: str = "ffprobe"
    # 场景切分阈值（0~1，越小越敏感）
    scene_threshold: float = 0.35
    # 场景切分最大数量
    scene_max_cuts: int = 500
    # fallback 分镜长度（ms），当 scene cuts 为 0 时按此间隔分割
    fallback_shot_ms: int = 3000


def _get_video_duration(video_path: str, ffprobe_path: str = "ffprobe") -> Optional[float]:
    """用 ffprobe 获取视频时长"""
    try:
        cmd = [
            ffprobe_path, "-v", "error",
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            video_path,
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        return float(result.stdout.strip()) if result.returncode == 0 else None
    except Exception:
        return None


def _extract_frames(
    video_path: str,
    interval_s: float,
    max_frames: int,
    width: int,
    ffmpeg_path: str = "ffmpeg",
) -> List[np.ndarray]:
    """用 ffmpeg 抽取关键帧，返回灰度帧列表"""
    tmp_dir = tempfile.mkdtemp(prefix="brain_frames_")
    try:
        cmd = [
            ffmpeg_path, "-y", "-i", video_path,
            "-vf", f"fps=1/{interval_s},scale={width}:-1",
            "-frames:v", str(max_frames),
            "-f", "image2", os.path.join(tmp_dir, "frame_%04d.pgm"),
        ]
        result = subprocess.run(cmd, capture_output=True, timeout=120)
        if result.returncode != 0:
            logger.error("[Visual] ffmpeg 抽帧失败: %s", result.stderr.decode(errors="replace")[:300])
            return []

        frames = []
        for fname in sorted(os.listdir(tmp_dir)):
            if not fname.endswith(".pgm"):
                continue
            fpath = os.path.join(tmp_dir, fname)
            # PGM P5 (binary grayscale) 简单解析
            frame = _load_pgm(fpath)
            if frame is not None:
                frames.append(frame)
        return frames
    except Exception as e:
        logger.error("[Visual] 抽帧异常: %s", e)
        return []
    finally:
        import shutil
        shutil.rmtree(tmp_dir, ignore_errors=True)


def _load_pgm(path: str) -> Optional[np.ndarray]:
    """加载 PGM (P5) 灰度图"""
    try:
        with open(path, "rb") as f:
            magic = f.readline().strip()
            if magic != b"P5":
                return None
            # 跳过注释
            line = f.readline()
            while line.startswith(b"#"):
                line = f.readline()
            w, h = map(int, line.split())
            max_val = int(f.readline().strip())
            data = f.read()
            img = np.frombuffer(data, dtype=np.uint8).reshape((h, w))
            return img
    except Exception:
        return None


def _compute_motion_score(frames: List[np.ndarray]) -> float:
    """计算帧间运动评分（相邻帧差异的均值）"""
    if len(frames) < 2:
        return 0.0

    diffs = []
    for i in range(1, len(frames)):
        prev = frames[i - 1].astype(np.float32) / 255.0
        curr = frames[i].astype(np.float32) / 255.0
        # 尺寸对齐（可能不同帧尺寸不同）
        min_h = min(prev.shape[0], curr.shape[0])
        min_w = min(prev.shape[1], curr.shape[1])
        diff = np.mean(np.abs(prev[:min_h, :min_w] - curr[:min_h, :min_w]))
        diffs.append(float(diff))

    return float(np.mean(diffs))


def _compute_fingerprint(frames: List[np.ndarray], n_bins: int = 16) -> List[float]:
    """
    计算视觉指纹（简化版：各帧灰度直方图的均值向量）

    返回 n_bins 维特征向量
    """
    if not frames:
        return [0.0] * n_bins

    histograms = []
    for frame in frames:
        hist, _ = np.histogram(frame.flatten(), bins=n_bins, range=(0, 255))
        hist = hist.astype(np.float32)
        total = hist.sum()
        if total > 0:
            hist /= total
        histograms.append(hist)

    avg = np.mean(histograms, axis=0)
    return [round(float(x), 6) for x in avg]


def _detect_scene_cuts(
    video_path: str,
    threshold: float = 0.35,
    max_cuts: int = 500,
    ffmpeg_path: str = "ffmpeg",
) -> List[int]:
    """
    使用 ffmpeg 的 scene 过滤器检测场景切分点。
    返回切分时间点（ms，升序），不含 0。
    """
    vf = f"select='gt(scene,{threshold})',showinfo"
    cmd = [
        ffmpeg_path, "-hide_banner",
        "-i", video_path,
        "-vf", vf,
        "-an", "-f", "null", "-",
    ]
    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True,
            encoding="utf-8", errors="replace", timeout=300,
        )
        text = result.stderr or ""
    except Exception as e:
        logger.error("[Visual] 场景检测命令失败: %s", e)
        return []

    pts_times: List[float] = []
    for line in text.splitlines():
        if "pts_time:" not in line:
            continue
        try:
            part = line.split("pts_time:", 1)[1]
            num = ""
            for ch in part:
                if ch.isdigit() or ch in ".-":
                    num += ch
                else:
                    break
            if num:
                pts_times.append(float(num))
        except Exception:
            continue
        if len(pts_times) >= max_cuts:
            break

    ms = sorted({max(int(t * 1000), 0) for t in pts_times if t >= 0})
    return [x for x in ms if x > 0]


def _build_shots(
    scene_cuts_ms: List[int],
    duration_ms: int,
    fallback_shot_ms: int = 3000,
) -> List[Dict[str, Any]]:
    """
    根据场景切分点构建分镜列表。
    当 scene_cuts 为空时，按 fallback_shot_ms 固定间隔分割。
    每个 shot 含 keyframe_ms（取中点）。
    """
    if duration_ms <= 0:
        return []

    if scene_cuts_ms:
        boundaries = [0] + scene_cuts_ms + [duration_ms]
    else:
        # fallback: 按固定间隔分割
        step = max(fallback_shot_ms, 500)
        boundaries = list(range(0, duration_ms, step))
        if not boundaries or boundaries[-1] != duration_ms:
            boundaries.append(duration_ms)
        # 至少 2 个边界才能形成 shot
        if len(boundaries) < 2:
            boundaries = [0, duration_ms]

    shots: List[Dict[str, Any]] = []
    for i in range(len(boundaries) - 1):
        start = boundaries[i]
        end = boundaries[i + 1]
        if end <= start:
            continue
        keyframe_ms = start + (end - start) // 2
        shots.append({
            "shot_id": i,
            "start_ms": start,
            "end_ms": end,
            "duration_ms": end - start,
            "start_s": round(start / 1000, 3),
            "end_s": round(end / 1000, 3),
            "keyframe_ms": keyframe_ms,
            "keyframe_s": round(keyframe_ms / 1000, 3),
        })
    return shots


def _compute_shot_features(
    shots: List[Dict[str, Any]],
    frames: List[np.ndarray],
    frame_interval_s: float,
) -> List[Dict[str, Any]]:
    """
    为每个 shot 计算视觉特征：亮度、对比度、运动评分。
    通过 frame_interval_s 将帧映射到对应的 shot。
    """
    if not frames or not shots:
        return shots

    for shot in shots:
        start_s = shot["start_s"]
        end_s = shot["end_s"]

        # 找到属于这个 shot 的帧
        shot_frames = []
        for idx, frame in enumerate(frames):
            ts = idx * frame_interval_s
            if start_s <= ts < end_s:
                shot_frames.append(frame)

        if not shot_frames:
            shot["avg_brightness"] = 0.0
            shot["contrast"] = 0.0
            shot["shot_motion"] = 0.0
            shot["frame_count"] = 0
            continue

        # 亮度：归一化像素均值 (0~1)
        brightness_vals = [float(np.mean(f)) / 255.0 for f in shot_frames]
        shot["avg_brightness"] = round(float(np.mean(brightness_vals)), 4)

        # 对比度：像素标准差均值 (0~1)
        contrast_vals = [float(np.std(f)) / 255.0 for f in shot_frames]
        shot["contrast"] = round(float(np.mean(contrast_vals)), 4)

        # 运动：相邻帧差异均值
        if len(shot_frames) >= 2:
            diffs = []
            for j in range(1, len(shot_frames)):
                prev = shot_frames[j - 1].astype(np.float32) / 255.0
                curr = shot_frames[j].astype(np.float32) / 255.0
                min_h = min(prev.shape[0], curr.shape[0])
                min_w = min(prev.shape[1], curr.shape[1])
                diffs.append(float(np.mean(np.abs(prev[:min_h, :min_w] - curr[:min_h, :min_w]))))
            shot["shot_motion"] = round(float(np.mean(diffs)), 6)
        else:
            shot["shot_motion"] = 0.0

        shot["frame_count"] = len(shot_frames)

    return shots


class VisualAnalyzer:
    """视觉分析器"""

    def __init__(self, cfg: Optional[VisualAnalyzerConfig] = None):
        self._cfg = cfg or VisualAnalyzerConfig()

    def analyze_file(self, video_path: str) -> Dict[str, Any]:
        """
        分析视频文件

        Returns:
            {
                "frame_count": int,
                "motion_score": float,
                "visual_fingerprint": List[float],
                "duration_s": float,
                "frame_timestamps_s": List[float],
                "scene_cuts_ms": List[int],
                "shots": List[Dict],
                "shot_count": int,
            }
        """
        cfg = self._cfg

        duration_s = _get_video_duration(video_path, cfg.ffprobe_path) or 0.0
        frames = _extract_frames(video_path, cfg.frame_interval_s, cfg.max_frames, cfg.frame_width, cfg.ffmpeg_path)

        # 场景切分检测
        scene_cuts_ms = _detect_scene_cuts(
            video_path, cfg.scene_threshold, cfg.scene_max_cuts, cfg.ffmpeg_path,
        )
        duration_ms = int(duration_s * 1000)
        shots = _build_shots(scene_cuts_ms, duration_ms, cfg.fallback_shot_ms)

        if not frames:
            logger.warning("[Visual] 未能抽取帧: %s", video_path)
            return {
                "frame_count": 0,
                "motion_score": 0.0,
                "visual_fingerprint": [0.0] * 16,
                "duration_s": duration_s,
                "frame_timestamps_s": [],
                "scene_cuts_ms": scene_cuts_ms,
                "shots": shots,
                "shot_count": len(shots),
            }

        motion_score = _compute_motion_score(frames)
        fingerprint = _compute_fingerprint(frames)
        timestamps = [round(i * cfg.frame_interval_s, 3) for i in range(len(frames))]

        # 为每个 shot 计算视觉特征
        _compute_shot_features(shots, frames, cfg.frame_interval_s)

        logger.info(
            "[Visual] 分析完成: frames=%d, motion=%.4f, scenes=%d, shots=%d, duration=%.1fs",
            len(frames), motion_score, len(scene_cuts_ms), len(shots), duration_s,
        )

        return {
            "frame_count": len(frames),
            "motion_score": round(motion_score, 6),
            "visual_fingerprint": fingerprint,
            "duration_s": round(duration_s, 3),
            "frame_timestamps_s": timestamps,
            "scene_cuts_ms": scene_cuts_ms,
            "shots": shots,
            "shot_count": len(shots),
        }

    def analyze_frames(self, frames: List[np.ndarray]) -> Dict[str, Any]:
        """直接分析帧列表（用于测试）"""
        motion_score = _compute_motion_score(frames)
        fingerprint = _compute_fingerprint(frames)
        return {
            "frame_count": len(frames),
            "motion_score": round(motion_score, 6),
            "visual_fingerprint": fingerprint,
        }
