"""
端到端视频感知 Demo — 完整 P1 多模态流水线

用法:
    python tools/demo_video_pipeline.py <视频路径>

无需外部服务（使用 mongomock + Qdrant 内存模式）。
流程: 视频 → ingest → audio + visual + text → align → fusion → embedding → search
"""
from __future__ import annotations

import json
import os
import sys
import time

# 确保项目根目录在 path 中
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from brain.logging import logger


def _setup_stores():
    """初始化 mongomock + Qdrant 内存"""
    import mongomock
    from qdrant_client import QdrantClient
    from libs.clients.qdrant_store import QdrantStore

    mongo_client = mongomock.MongoClient()
    db = mongo_client["brain_demo"]

    qdrant_client = QdrantClient(":memory:")
    vector_store = QdrantStore(client=qdrant_client)

    return db, vector_store


def run(video_path: str):
    if not os.path.isfile(video_path):
        print(f"[错误] 文件不存在: {video_path}")
        sys.exit(1)

    file_size = os.path.getsize(video_path)
    print(f"\n{'='*60}")
    print(f"  Brain 多模态感知 Pipeline — 端到端 Demo")
    print(f"{'='*60}")
    print(f"  输入: {video_path}")
    print(f"  大小: {file_size / 1024 / 1024:.1f} MB")
    print(f"{'='*60}\n")

    t0 = time.time()

    # ── 1. 初始化存储 ──
    print("[1/7] 初始化存储 (mongomock + Qdrant 内存) ...")
    db, vector_store = _setup_stores()

    # ── 2. Ingest 注册资产 ──
    print("[2/7] Ingest — 注册资产 ...")
    from services.ingest.handler import IngestHandler
    ingest = IngestHandler(mongo_db=db)
    result = ingest.ingest(
        oss_path=video_path,
        asset_type="video",
        filename=os.path.basename(video_path),
        size_bytes=file_size,
    )
    asset_id = result["asset_id"]
    print(f"       asset_id = {asset_id}")

    # ── 3. 音频分析 ──
    print("[3/7] Audio — 音频分析 (静音/RMS/ZCR/语音段) ...")
    from services.audio.analyzer import AudioAnalyzer
    audio_analyzer = AudioAnalyzer()
    audio_data = audio_analyzer.analyze_file(video_path)
    _print_stage("Audio", {
        "时长": f"{audio_data.get('duration_s', 0):.1f}s",
        "静音比": f"{audio_data.get('silence_ratio', 0)*100:.1f}%",
        "RMS": f"{audio_data.get('rms_mean', 0):.4f}",
        "ZCR": f"{audio_data.get('zcr', 0):.4f}",
        "语音段数": len(audio_data.get("speech_segments", [])),
    })

    # ── 4. 视觉分析 ──
    print("[4/7] Visual — 视觉分析 (抽帧/运动/指纹/场景检测) ...")
    from services.visual.analyzer import VisualAnalyzer
    visual_analyzer = VisualAnalyzer()
    visual_data = visual_analyzer.analyze_file(video_path)
    _print_stage("Visual", {
        "帧数": visual_data.get("frame_count", 0),
        "运动评分": f"{visual_data.get('motion_score', 0):.4f}",
        "视频时长": f"{visual_data.get('duration_s', 0):.1f}s",
        "指纹维度": len(visual_data.get("visual_fingerprint", [])),
        "场景切分": f"{len(visual_data.get('scene_cuts_ms', []))} 个切点",
        "分镜数": visual_data.get("shot_count", 0),
    })

    # ── 4.5 VLM 场景描述 ──
    print("[4.5/7] VLM — 分镜场景描述 (Qwen2-VL) ...")
    shots = visual_data.get("shots", [])
    if shots:
        try:
            from services.visual.scene_describer import SceneDescriber
            describer = SceneDescriber()
            describer.describe_shots(video_path, shots)
            described = sum(1 for s in shots if s.get("scene_description"))
            print(f"       ┌─ SceneDescriber 结果:")
            print(f"       │  已描述: {described}/{len(shots)} 个分镜")
            for s in shots[:5]:
                desc = s.get("scene_description", "")
                if desc:
                    print(f"       │  Shot #{s['shot_id']}: {desc[:60]}")
            print(f"       └─ OK")
        except Exception as e:
            logger.warning("[Demo] VLM 场景描述不可用，跳过: %s", e)

    # ── 5. 文本分析 (ASR 可选) ──
    print("[5/7] Text — 文本分析 (ASR + 语速 + CTA) ...")
    from services.text.analyzer import TextAnalyzer, TextAnalyzerConfig
    # 尝试用 ASR；若 funasr 不可用则跳过
    text_cfg = TextAnalyzerConfig()
    text_analyzer = TextAnalyzer(text_cfg)

    text_data = {"transcript": "", "token_count": 0, "token_per_sec": 0.0,
                 "duration_s": audio_data.get("duration_s", 0), "cta_matches": [], "has_cta": False}
    if audio_data.get("duration_s", 0) > 0:
        from services.audio.analyzer import _extract_audio_from_video
        pcm_bytes = _extract_audio_from_video(video_path, 16000)
        if pcm_bytes:
            try:
                text_data = text_analyzer.analyze_pcm(pcm_bytes, audio_data["duration_s"])
            except Exception as e:
                logger.warning("[Demo] ASR 不可用，跳过文本分析: %s", e)
                text_data = text_analyzer.analyze_text("", audio_data["duration_s"])

    transcript_preview = text_data.get("transcript", "")[:80]
    _print_stage("Text", {
        "转写": transcript_preview if transcript_preview else "(无/ASR 不可用)",
        "Token数": text_data.get("token_count", 0),
        "语速": f"{text_data.get('token_per_sec', 0):.1f} tok/s",
        "CTA": "有" if text_data.get("has_cta") else "无",
    })

    # ── 6. 时间对齐 ──
    print("[6/7] Align — 多模态时间对齐 ...")
    from services.align.aligner import TimeAligner
    aligner = TimeAligner()
    duration_s = max(
        audio_data.get("duration_s", 0),
        visual_data.get("duration_s", 0),
    )
    align_result = aligner.align(
        visual_data=visual_data,
        audio_data=audio_data,
        text_data=text_data,
        duration_s=duration_s,
    )
    timeline = align_result.get("timeline", [])
    coverage = align_result.get("modality_coverage", {})
    _print_stage("Align", {
        "Segment数": len(timeline),
        "视觉覆盖": f"{coverage.get('visual', 0)*100:.0f}%",
        "音频覆盖": f"{coverage.get('audio', 0)*100:.0f}%",
        "文本覆盖": f"{coverage.get('text', 0)*100:.0f}%",
    })

    # ── 7. 融合 Embedding + Script Graph ──
    print("[7/7] Fusion — 多模态融合 + Script Graph ...")
    from services.fusion.fuser import FusionEngine
    fuser = FusionEngine()
    fusion_result = fuser.fuse(
        visual_data=visual_data,
        audio_data=audio_data,
        text_data=text_data,
        timeline=timeline,
        duration_s=duration_s,
    )
    graph = fusion_result.get("script_graph", {})
    emb = fusion_result.get("fusion_embedding", [])
    _print_stage("Fusion", {
        "Embedding维度": fusion_result.get("fusion_dim", 0),
        "Graph节点数": graph.get("total_segments", 0),
        "Graph边数": len(graph.get("edges", [])),
    })

    # ── 写入向量 + 搜索验证 ──
    print("\n[向量] 写入 Qdrant 并执行相似度搜索 ...")
    collection = "perception"
    dim = fusion_result["fusion_dim"]
    vector_store.ensure_collection(collection, dim=dim, namespace="v1")
    vector_store.upsert(
        collection=collection,
        ids=[f"{asset_id}_fusion_v1"],
        vectors=[emb],
        metadatas=[{"asset_id": asset_id, "stage": "fusion"}],
        namespace="v1",
    )

    from services.orchestrator.search_handler import SearchHandler
    searcher = SearchHandler(vector_store)
    hits = searcher.search(query_vector=emb, top_k=5, collection=collection)
    print(f"  搜索命中: {len(hits)} 条")
    for h in hits:
        print(f"    - id={h['id']}, score={h['score']:.4f}")

    total_time = time.time() - t0

    # ── 总结 ──
    print(f"\n{'='*60}")
    print(f"  Pipeline 完成！总耗时: {total_time:.1f}s")
    print(f"{'='*60}")

    # 分镜概览
    shots = visual_data.get("shots", [])
    if shots:
        print(f"\n  分镜列表 ({len(shots)} shots):")
        for s in shots[:15]:
            line = (f"    Shot #{s['shot_id']:2d}  "
                    f"[{s['start_s']:.1f}s - {s['end_s']:.1f}s]  "
                    f"时长 {s['duration_ms']}ms")
            if "avg_brightness" in s:
                line += (f"  亮度={s['avg_brightness']:.2f}"
                         f"  对比度={s.get('contrast', 0):.2f}"
                         f"  运动={s.get('shot_motion', 0):.4f}")
            if "keyframe_s" in s:
                line += f"  关键帧={s['keyframe_s']:.1f}s"
            print(line)
            desc = s.get("scene_description", "")
            if desc:
                print(f"           → {desc[:100]}")
        if len(shots) > 15:
            print(f"    ... 还有 {len(shots)-15} 个分镜")

    # Script Graph 概览
    nodes = graph.get("nodes", [])
    if nodes:
        print(f"\n  Script Graph ({len(nodes)} segments):")
        for n in nodes[:10]:
            print(f"    [{n['start_s']:.1f}s - {n['end_s']:.1f}s] "
                  f"{'/'.join(n['modalities']):20s} → {n['label']}")
        if len(nodes) > 10:
            print(f"    ... 还有 {len(nodes)-10} 个 segment")

    # ASR 转写
    transcript = text_data.get("transcript", "")
    if transcript:
        print(f"\n  ASR 转写:")
        print(f"    {transcript[:200]}")
        if len(transcript) > 200:
            print(f"    ... (共 {len(transcript)} 字符)")

    # 输出完整 JSON
    output_path = os.path.splitext(video_path)[0] + "_perception.json"
    full_result = {
        "asset_id": asset_id,
        "video_path": video_path,
        "duration_s": duration_s,
        "audio": audio_data,
        "visual": {k: v for k, v in visual_data.items() if k not in ("frame_timestamps_s",)},
        "text": text_data,
        "alignment": align_result,
        "fusion": {
            "fusion_dim": fusion_result["fusion_dim"],
            "embedding_norm": float(sum(x**2 for x in emb)**0.5),
            "script_graph": graph,
        },
        "total_time_s": round(total_time, 2),
    }
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(full_result, f, ensure_ascii=False, indent=2, default=str)
    print(f"\n  详细结果已保存: {output_path}\n")


def _print_stage(name: str, metrics: dict):
    """格式化打印阶段结果"""
    print(f"       ┌─ {name} 结果:")
    for k, v in metrics.items():
        print(f"       │  {k}: {v}")
    print(f"       └─ OK")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("用法: python tools/demo_video_pipeline.py <视频路径>")
        print("示例: python tools/demo_video_pipeline.py D:\\videos\\test.mp4")
        sys.exit(1)
    run(sys.argv[1])
