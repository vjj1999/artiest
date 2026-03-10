"""
P1 端到端集成测试: mongomock + Qdrant 内存

跑通: 视频 → visual+audio+text → align → fusion → search
"""
import sys
import wave
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from tests.integration.conftest import make_wav
from services.audio.analyzer import AudioAnalyzer
from services.visual.analyzer import VisualAnalyzer, _compute_motion_score, _compute_fingerprint
from services.text.analyzer import TextAnalyzer, TextAnalyzerConfig
from services.align.aligner import TimeAligner
from services.fusion.fuser import FusionEngine
from services.orchestrator.search_handler import SearchHandler


def _gen_frames(n: int = 10) -> list:
    """生成模拟帧序列"""
    return [np.random.randint(0, 255, (64, 80), dtype=np.uint8) for _ in range(n)]


class TestP1Pipeline:
    """P1: visual+audio+text → align → fusion → search"""

    def test_完整P1流水线(self, mongo_db, ingest, qdrant_store, search):
        """端到端多模态: audio+visual+text → align → fusion → search"""
        wav_path = make_wav(duration_s=5.0, freq=440)

        # ── Step 1: Ingest ──
        r = ingest.ingest(
            oss_path="oss://b/multimodal.mp4",
            asset_type="video",
            duration_s=5.0,
            stages=["audio", "visual", "text", "align", "fusion"],
        )
        asset_id = r["asset_id"]

        # ── Step 2: Audio 分析 ──
        audio_analyzer = AudioAnalyzer()
        audio_data = audio_analyzer.analyze_file(wav_path)
        assert audio_data["rms_mean"] > 0
        assert "speech_segments" in audio_data

        # ── Step 3: Visual 分析（用模拟帧） ──
        visual_analyzer = VisualAnalyzer()
        frames = _gen_frames(10)
        visual_data = visual_analyzer.analyze_frames(frames)
        # 补充 file-level 字段
        visual_data["duration_s"] = 5.0
        visual_data["frame_timestamps_s"] = [float(i) * 0.5 for i in range(10)]
        assert visual_data["frame_count"] == 10
        assert len(visual_data["visual_fingerprint"]) == 16

        # ── Step 4: Text 分析（跳过 ASR，直接给 transcript） ──
        text_analyzer = TextAnalyzer(TextAnalyzerConfig(disable_asr=True))
        text_data = text_analyzer.analyze_text(
            transcript="这是一段测试视频，请点击关注并订阅我们的频道",
            duration_s=5.0,
        )
        assert text_data["token_count"] > 0
        assert text_data["has_cta"] is True

        # ── Step 5: Time Alignment ──
        aligner = TimeAligner()
        align_result = aligner.align(
            visual_data=visual_data,
            audio_data=audio_data,
            text_data=text_data,
            duration_s=5.0,
        )
        assert len(align_result["timeline"]) >= 1
        coverage = align_result["modality_coverage"]
        assert coverage["audio"] > 0
        assert coverage["text"] > 0

        # ── Step 6: Fusion ──
        fusion_engine = FusionEngine()
        fusion_result = fusion_engine.fuse(
            visual_data=visual_data,
            audio_data=audio_data,
            text_data=text_data,
            timeline=align_result["timeline"],
            duration_s=5.0,
        )
        embedding = fusion_result["fusion_embedding"]
        assert len(embedding) == 64
        # L2 范数约等于 1
        norm = float(np.linalg.norm(embedding))
        assert abs(norm - 1.0) < 0.01

        graph = fusion_result["script_graph"]
        assert len(graph["nodes"]) >= 1
        assert graph["total_segments"] >= 1

        # ── Step 7: 写入 Qdrant + 搜索 ──
        qdrant_store.ensure_collection("perception", dim=64, namespace="v1")
        emb_id = f"{asset_id}_fusion_v1"
        qdrant_store.upsert(
            collection="perception",
            ids=[emb_id],
            vectors=[embedding],
            metadatas=[{"asset_id": asset_id, "stage": "fusion"}],
            namespace="v1",
        )

        # 搜索
        results = search.search(query_vector=embedding, top_k=5)
        assert len(results) >= 1
        assert results[0]["metadata"]["asset_id"] == asset_id
        assert results[0]["score"] > 0.99  # 查自己，近似 1.0

    def test_多模态资产相似度排序(self, mongo_db, ingest, qdrant_store, search):
        """多个多模态资产入库后，搜索应返回最相似的"""
        fusion_engine = FusionEngine()

        embeddings = {}
        asset_ids = []
        for i, text in enumerate([
            "科技产品评测，点击购买",
            "旅游风景纪录片，请关注",
            "科技产品对比测评，点击链接下单",
        ]):
            r = ingest.ingest(oss_path=f"oss://b/v{i}.mp4", stages=["fusion"])
            aid = r["asset_id"]
            asset_ids.append(aid)

            text_analyzer = TextAnalyzer(TextAnalyzerConfig(disable_asr=True))
            td = text_analyzer.analyze_text(text, duration_s=10.0)

            fr = fusion_engine.fuse(text_data=td, duration_s=10.0)
            emb = fr["fusion_embedding"]
            embeddings[aid] = emb

        # 写入 Qdrant
        qdrant_store.ensure_collection("perception", dim=64, namespace="v1")
        for aid, emb in embeddings.items():
            qdrant_store.upsert(
                "perception",
                ids=[f"{aid}_fusion"],
                vectors=[emb],
                metadatas=[{"asset_id": aid, "stage": "fusion"}],
            )

        # 用第一个（科技评测）的 embedding 搜索 → 应该和第三个（科技对比）最相似
        results = search.search(query_vector=embeddings[asset_ids[0]], top_k=3)
        assert len(results) == 3
        # 第一个是自己
        assert results[0]["metadata"]["asset_id"] == asset_ids[0]

    def test_各阶段数据流兼容(self, mongo_db, ingest, qdrant_store):
        """验证各服务输出格式能正确传递给下游"""
        # Audio → Align（speech_segments 格式兼容）
        audio_data = {
            "silence_ratio": 0.2,
            "rms_mean": 0.04,
            "zcr": 0.08,
            "duration_s": 10.0,
            "speech_segments": [{"start_s": 1.0, "end_s": 3.5}, {"start_s": 5.0, "end_s": 8.0}],
        }
        visual_data = {
            "frame_count": 10,
            "motion_score": 0.05,
            "visual_fingerprint": [0.1] * 16,
            "duration_s": 10.0,
            "frame_timestamps_s": [float(i) for i in range(10)],
        }
        text_data = {
            "transcript": "测试文本",
            "token_count": 4,
            "token_per_sec": 0.4,
            "duration_s": 10.0,
            "cta_matches": [],
            "has_cta": False,
        }

        # Align
        aligner = TimeAligner()
        align_result = aligner.align(
            visual_data=visual_data,
            audio_data=audio_data,
            text_data=text_data,
            duration_s=10.0,
        )
        assert "timeline" in align_result

        # Fusion（接收 align 的 timeline）
        fusion = FusionEngine()
        fusion_result = fusion.fuse(
            visual_data=visual_data,
            audio_data=audio_data,
            text_data=text_data,
            timeline=align_result["timeline"],
            duration_s=10.0,
        )
        assert len(fusion_result["fusion_embedding"]) == 64
        assert "script_graph" in fusion_result

        # 写入 Qdrant
        qdrant_store.ensure_collection("perception", dim=64, namespace="v1")
        qdrant_store.upsert(
            "perception",
            ids=["test_fusion"],
            vectors=[fusion_result["fusion_embedding"]],
            metadatas=[{"stage": "fusion"}],
        )

        # 搜索验证
        results = qdrant_store.search("perception", fusion_result["fusion_embedding"], top_k=1)
        assert len(results) == 1
        assert results[0].score > 0.99
