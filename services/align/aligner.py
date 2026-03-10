"""
Time Alignment Service — 多模态时间对齐

将视觉帧时间戳、音频语音段、ASR 转写片段对齐到统一时间轴，
输出按时间排列的多模态 segment 列表。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from brain.logging import logger


@dataclass
class AlignConfig:
    """对齐配置"""
    # 合并阈值（秒）：相邻 segment 间隔小于此值时合并
    merge_gap_s: float = 0.5
    # 最小 segment 时长（秒）
    min_segment_s: float = 0.1


@dataclass
class TimelineSegment:
    """统一时间轴上的片段"""
    start_s: float
    end_s: float
    modalities: List[str] = field(default_factory=list)  # ["visual", "audio", "text"]
    data: Dict[str, Any] = field(default_factory=dict)

    @property
    def duration_s(self) -> float:
        return self.end_s - self.start_s


class TimeAligner:
    """多模态时间对齐器"""

    def __init__(self, cfg: Optional[AlignConfig] = None):
        self._cfg = cfg or AlignConfig()

    def align(
        self,
        visual_data: Optional[Dict[str, Any]] = None,
        audio_data: Optional[Dict[str, Any]] = None,
        text_data: Optional[Dict[str, Any]] = None,
        duration_s: float = 0.0,
    ) -> Dict[str, Any]:
        """
        将多模态数据对齐到统一时间轴

        Args:
            visual_data: VisualAnalyzer 输出
            audio_data: AudioAnalyzer 输出
            text_data: TextAnalyzer 输出
            duration_s: 总时长

        Returns:
            {
                "timeline": [TimelineSegment, ...],
                "duration_s": float,
                "modality_coverage": {"visual": float, "audio": float, "text": float},
            }
        """
        # 收集所有时间事件
        events: List[Dict[str, Any]] = []

        if visual_data:
            events.extend(self._visual_events(visual_data))
        if audio_data:
            events.extend(self._audio_events(audio_data))
        if text_data:
            events.extend(self._text_events(text_data, duration_s))

        # 按 start_s 排序
        events.sort(key=lambda e: e["start_s"])

        # 合并相近事件为 segment
        segments = self._merge_events(events)

        # 计算模态覆盖率
        coverage = self._compute_coverage(segments, duration_s)

        logger.info(
            "[Align] 对齐完成: %d segments, coverage=%s",
            len(segments), coverage,
        )

        return {
            "timeline": [
                {
                    "start_s": round(s.start_s, 3),
                    "end_s": round(s.end_s, 3),
                    "duration_s": round(s.duration_s, 3),
                    "modalities": s.modalities,
                    "data": s.data,
                }
                for s in segments
            ],
            "duration_s": round(duration_s, 3),
            "modality_coverage": coverage,
        }

    def _visual_events(self, visual_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """从视觉数据提取时间事件"""
        events = []
        timestamps = visual_data.get("frame_timestamps_s", [])
        interval = 1.0  # 默认 1s 间隔
        if len(timestamps) >= 2:
            interval = timestamps[1] - timestamps[0]

        for ts in timestamps:
            events.append({
                "start_s": ts,
                "end_s": ts + interval,
                "modality": "visual",
                "data": {"frame_ts": ts},
            })
        return events

    def _audio_events(self, audio_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """从音频数据提取时间事件"""
        events = []
        for seg in audio_data.get("speech_segments", []):
            events.append({
                "start_s": seg["start_s"],
                "end_s": seg["end_s"],
                "modality": "audio",
                "data": {"type": "speech"},
            })
        return events

    def _text_events(self, text_data: Dict[str, Any], duration_s: float) -> List[Dict[str, Any]]:
        """从文本数据提取时间事件（整段覆盖）"""
        transcript = text_data.get("transcript", "")
        if not transcript:
            return []
        return [{
            "start_s": 0.0,
            "end_s": duration_s if duration_s > 0 else text_data.get("duration_s", 0.0),
            "modality": "text",
            "data": {"transcript": transcript[:100]},
        }]

    def _merge_events(self, events: List[Dict[str, Any]]) -> List[TimelineSegment]:
        """将时间事件合并为 segment（相邻且重叠的合并）"""
        if not events:
            return []

        cfg = self._cfg
        segments: List[TimelineSegment] = []

        for ev in events:
            if ev["end_s"] - ev["start_s"] < cfg.min_segment_s:
                continue

            merged = False
            for seg in segments:
                # 检查是否重叠或间隔小于阈值
                if ev["start_s"] <= seg.end_s + cfg.merge_gap_s and ev["end_s"] >= seg.start_s - cfg.merge_gap_s:
                    # 合并
                    seg.start_s = min(seg.start_s, ev["start_s"])
                    seg.end_s = max(seg.end_s, ev["end_s"])
                    if ev["modality"] not in seg.modalities:
                        seg.modalities.append(ev["modality"])
                    seg.data[ev["modality"]] = ev.get("data", {})
                    merged = True
                    break

            if not merged:
                segments.append(TimelineSegment(
                    start_s=ev["start_s"],
                    end_s=ev["end_s"],
                    modalities=[ev["modality"]],
                    data={ev["modality"]: ev.get("data", {})},
                ))

        segments.sort(key=lambda s: s.start_s)
        return segments

    @staticmethod
    def _compute_coverage(
        segments: List[TimelineSegment],
        duration_s: float,
    ) -> Dict[str, float]:
        """计算各模态的时间覆盖率"""
        if duration_s <= 0:
            return {"visual": 0.0, "audio": 0.0, "text": 0.0}

        coverage: Dict[str, float] = {"visual": 0.0, "audio": 0.0, "text": 0.0}
        for seg in segments:
            for mod in seg.modalities:
                if mod in coverage:
                    coverage[mod] += seg.duration_s

        for mod in coverage:
            coverage[mod] = round(min(coverage[mod] / duration_s, 1.0), 4)

        return coverage
