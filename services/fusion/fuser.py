"""
Fusion Service — 多模态融合 Embedding + Script Graph

将视觉、音频、文本特征融合为统一向量，
并生成 script_graph（描述内容时间结构的 JSON）。
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

from brain.logging import logger


@dataclass
class FusionConfig:
    """融合配置"""
    # 融合向量维度
    fusion_dim: int = 64
    # 各模态权重
    weight_visual: float = 0.3
    weight_audio: float = 0.3
    weight_text: float = 0.4
    # Script graph 最大 segment 数
    max_graph_segments: int = 50


@dataclass
class ScriptGraphNode:
    """脚本图节点"""
    segment_id: str
    start_s: float
    end_s: float
    modalities: List[str]
    visual_score: float = 0.0
    audio_energy: float = 0.0
    text_density: float = 0.0
    label: str = ""


class FusionEngine:
    """多模态融合引擎"""

    def __init__(self, cfg: Optional[FusionConfig] = None):
        self._cfg = cfg or FusionConfig()

    def fuse(
        self,
        visual_data: Optional[Dict[str, Any]] = None,
        audio_data: Optional[Dict[str, Any]] = None,
        text_data: Optional[Dict[str, Any]] = None,
        timeline: Optional[List[Dict[str, Any]]] = None,
        duration_s: float = 0.0,
    ) -> Dict[str, Any]:
        """
        执行多模态融合

        Args:
            visual_data: VisualAnalyzer 输出
            audio_data: AudioAnalyzer 输出
            text_data: TextAnalyzer 输出
            timeline: TimeAligner 输出的 timeline
            duration_s: 总时长

        Returns:
            {
                "fusion_embedding": List[float],
                "fusion_dim": int,
                "script_graph": {...},
            }
        """
        cfg = self._cfg

        # 提取各模态特征向量
        v_vec = self._visual_features(visual_data, cfg.fusion_dim)
        a_vec = self._audio_features(audio_data, cfg.fusion_dim)
        t_vec = self._text_features(text_data, cfg.fusion_dim)

        # 加权融合
        fusion = (
            cfg.weight_visual * v_vec +
            cfg.weight_audio * a_vec +
            cfg.weight_text * t_vec
        )
        # L2 归一化
        norm = float(np.linalg.norm(fusion)) + 1e-9
        fusion = fusion / norm

        # 生成 script graph
        graph = self._build_script_graph(
            visual_data, audio_data, text_data, timeline, duration_s,
        )

        logger.info(
            "[Fusion] 融合完成: dim=%d, graph_nodes=%d",
            cfg.fusion_dim, len(graph.get("nodes", [])),
        )

        return {
            "fusion_embedding": [round(float(x), 6) for x in fusion],
            "fusion_dim": cfg.fusion_dim,
            "script_graph": graph,
        }

    def _visual_features(self, data: Optional[Dict[str, Any]], dim: int) -> np.ndarray:
        """从视觉数据构建特征向量"""
        vec = np.zeros(dim, dtype=np.float32)
        if not data:
            return vec

        fp = data.get("visual_fingerprint", [])
        motion = data.get("motion_score", 0.0)
        frame_count = data.get("frame_count", 0)

        # 填入指纹
        for i, v in enumerate(fp):
            if i < dim:
                vec[i] = float(v)
        # 最后几维放运动信息
        if dim > len(fp):
            vec[min(len(fp), dim - 2)] = float(motion)
            vec[min(len(fp) + 1, dim - 1)] = float(frame_count) / 60.0

        return vec

    def _audio_features(self, data: Optional[Dict[str, Any]], dim: int) -> np.ndarray:
        """从音频数据构建特征向量"""
        vec = np.zeros(dim, dtype=np.float32)
        if not data:
            return vec

        vec[0] = float(data.get("silence_ratio", 0.0))
        vec[1] = float(data.get("rms_mean", 0.0))
        vec[2] = float(data.get("zcr", 0.0))
        vec[3] = float(data.get("duration_s", 0.0)) / 60.0  # 归一化到分钟

        # 语音段数量和总覆盖
        segments = data.get("speech_segments", [])
        vec[4] = float(len(segments)) / 10.0
        total_speech = sum(s.get("end_s", 0) - s.get("start_s", 0) for s in segments)
        dur = data.get("duration_s", 1.0) or 1.0
        vec[5] = total_speech / dur

        return vec

    def _text_features(self, data: Optional[Dict[str, Any]], dim: int) -> np.ndarray:
        """从文本数据构建特征向量"""
        vec = np.zeros(dim, dtype=np.float32)
        if not data:
            return vec

        transcript = data.get("transcript", "")
        vec[0] = float(data.get("token_count", 0)) / 100.0
        vec[1] = float(data.get("token_per_sec", 0.0)) / 10.0
        vec[2] = 1.0 if data.get("has_cta", False) else 0.0
        vec[3] = float(len(data.get("cta_matches", []))) / 5.0

        # 简单文本 hash 作为伪语义特征
        if transcript:
            h = hashlib.md5(transcript.encode()).digest()
            for i in range(min(8, dim - 4)):
                vec[4 + i] = float(h[i]) / 255.0

        return vec

    def _build_script_graph(
        self,
        visual_data: Optional[Dict[str, Any]],
        audio_data: Optional[Dict[str, Any]],
        text_data: Optional[Dict[str, Any]],
        timeline: Optional[List[Dict[str, Any]]],
        duration_s: float,
    ) -> Dict[str, Any]:
        """构建脚本图"""
        cfg = self._cfg
        nodes: List[Dict[str, Any]] = []

        segments = timeline or []
        for i, seg in enumerate(segments[:cfg.max_graph_segments]):
            node = ScriptGraphNode(
                segment_id=f"seg_{i:03d}",
                start_s=seg.get("start_s", 0.0),
                end_s=seg.get("end_s", 0.0),
                modalities=seg.get("modalities", []),
            )

            # 填充指标
            if "audio" in node.modalities and audio_data:
                node.audio_energy = audio_data.get("rms_mean", 0.0)
            if "visual" in node.modalities and visual_data:
                node.visual_score = visual_data.get("motion_score", 0.0)
            if "text" in node.modalities and text_data:
                dur = node.end_s - node.start_s
                if dur > 0:
                    node.text_density = text_data.get("token_per_sec", 0.0)

            # 自动标注
            node.label = self._classify_segment(node)

            nodes.append({
                "segment_id": node.segment_id,
                "start_s": round(node.start_s, 3),
                "end_s": round(node.end_s, 3),
                "modalities": node.modalities,
                "visual_score": round(node.visual_score, 4),
                "audio_energy": round(node.audio_energy, 4),
                "text_density": round(node.text_density, 2),
                "label": node.label,
            })

        # 构建边（相邻 segment 之间的时序关系）
        edges = []
        for i in range(len(nodes) - 1):
            edges.append({
                "from": nodes[i]["segment_id"],
                "to": nodes[i + 1]["segment_id"],
                "type": "sequential",
            })

        return {
            "nodes": nodes,
            "edges": edges,
            "duration_s": round(duration_s, 3),
            "total_segments": len(nodes),
        }

    @staticmethod
    def _classify_segment(node: ScriptGraphNode) -> str:
        """根据模态组合和指标自动分类 segment"""
        mods = set(node.modalities)
        if mods == {"visual", "audio", "text"}:
            return "narration"  # 有画面、有声音、有文字 → 叙述段
        elif mods == {"visual", "audio"}:
            return "action"  # 有画面、有声音 → 动作段
        elif mods == {"audio", "text"}:
            return "voiceover"  # 无画面、有声音和文字 → 旁白
        elif "visual" in mods:
            return "visual_only"
        elif "audio" in mods:
            return "audio_only"
        elif "text" in mods:
            return "text_only"
        else:
            return "silence"
