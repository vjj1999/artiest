"""
Orchestrator — 感知流水线编排器

接收 analyze 请求，按阶段顺序执行感知任务：
1. 从 OSS 获取文件
2. 调用 Audio/Visual/Text 等服务
3. 将结果写入 MongoDB (ResultEnvelope)
4. 将 embedding 写入 Vector Store
"""
from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from brain.logging import logger
from libs.schema.result_envelope import (
    AssetMeta, AssetType, PipelineInfo, StageResult, StageStatus,
    EmbeddingRef, ResultEnvelope,
)
from libs.clients.vector_store_interface import VectorStoreInterface
from services.audio.analyzer import AudioAnalyzer, AudioAnalyzerConfig


@dataclass
class OrchestratorConfig:
    """编排器配置"""
    mongo_uri: str = "mongodb://localhost:27017"
    db_name: str = "brain_db"
    results_collection: str = "perception_results"
    jobs_collection: str = "jobs"
    assets_collection: str = "assets"
    # 向量集合名
    vector_collection: str = "perception"
    vector_namespace: str = "v1"
    # 音频分析配置
    audio_config: AudioAnalyzerConfig = field(default_factory=AudioAnalyzerConfig)


def _gen_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


class Orchestrator:
    """感知流水线编排器"""

    def __init__(
        self,
        cfg: Optional[OrchestratorConfig] = None,
        vector_store: Optional[VectorStoreInterface] = None,
        mongo_db: Any = None,
    ):
        self._cfg = cfg or OrchestratorConfig()
        self._vector_store = vector_store
        self._db = mongo_db
        self._audio = AudioAnalyzer(self._cfg.audio_config)

    def _ensure_db(self):
        if self._db is not None:
            return
        try:
            from pymongo import MongoClient
            client = MongoClient(self._cfg.mongo_uri)
            self._db = client[self._cfg.db_name]
        except Exception as e:
            raise RuntimeError(f"MongoDB 连接失败: {e}") from e

    def analyze(
        self,
        asset_id: str,
        file_path: str,
        stages: Optional[List[str]] = None,
        job_id: Optional[str] = None,
        pipeline_id: Optional[str] = None,
    ) -> ResultEnvelope:
        """
        执行感知分析

        Args:
            asset_id: 资产 ID
            file_path: 本地文件路径（或 OSS 路径，需预先下载）
            stages: 要执行的阶段列表，默认 ["audio"]
            job_id: 关联的任务 ID
            pipeline_id: 流水线 ID

        Returns:
            ResultEnvelope 完整结果
        """
        self._ensure_db()
        cfg = self._cfg
        stages = stages or ["audio"]
        pipeline_id = pipeline_id or _gen_id("pipe")
        now = datetime.now(timezone.utc)

        # 获取资产信息
        asset_doc = self._db[cfg.assets_collection].find_one({"asset_id": asset_id}, {"_id": 0})
        if not asset_doc:
            raise ValueError(f"资产不存在: {asset_id}")

        # 构建 envelope
        envelope = ResultEnvelope(
            asset_meta=AssetMeta(
                asset_id=asset_id,
                asset_type=asset_doc.get("asset_type", "video"),
                oss_path=asset_doc.get("oss_path", ""),
                filename=asset_doc.get("filename", ""),
                duration_s=asset_doc.get("duration_s"),
                size_bytes=asset_doc.get("size_bytes"),
            ),
            pipeline=PipelineInfo(
                pipeline_id=pipeline_id,
                stages=stages,
                status=StageStatus.RUNNING,
                started_at=now,
            ),
        )

        # 更新 job 状态
        if job_id:
            self._update_job_status(job_id, "running", started_at=now)

        # 逐阶段执行
        all_success = True
        for stage_name in stages:
            stage_result = self._run_stage(stage_name, file_path, asset_id)
            envelope.outputs.append(stage_result)
            if stage_result.status != "success":
                all_success = False

        # 更新 pipeline 状态
        finished_at = datetime.now(timezone.utc)
        total_ms = (finished_at - now).total_seconds() * 1000
        envelope.pipeline.status = StageStatus.SUCCESS if all_success else StageStatus.FAILED
        envelope.pipeline.finished_at = finished_at
        envelope.pipeline.total_duration_ms = round(total_ms, 2)

        # 写入 MongoDB
        self._save_envelope(envelope)

        # 更新 job 状态
        if job_id:
            self._update_job_status(
                job_id,
                "success" if all_success else "failed",
                finished_at=finished_at,
            )

        logger.info(
            "[Orchestrator] 分析完成: %s (stages=%s, status=%s, %.0fms)",
            asset_id, stages, envelope.pipeline.status, total_ms,
        )
        return envelope

    def _run_stage(self, stage_name: str, file_path: str, asset_id: str) -> StageResult:
        """执行单个感知阶段"""
        start = time.time()
        started_at = datetime.now(timezone.utc)

        try:
            if stage_name == "audio":
                data = self._run_audio(file_path, asset_id)
            else:
                # P1 阶段占位
                data = {"message": f"阶段 {stage_name} 尚未实现"}
                return StageResult(
                    stage=stage_name,
                    status=StageStatus.SKIPPED,
                    started_at=started_at,
                    data=data,
                )

            elapsed_ms = (time.time() - start) * 1000
            return StageResult(
                stage=stage_name,
                status=StageStatus.SUCCESS,
                started_at=started_at,
                finished_at=datetime.now(timezone.utc),
                duration_ms=round(elapsed_ms, 2),
                data=data,
            )
        except Exception as e:
            elapsed_ms = (time.time() - start) * 1000
            logger.error("[Orchestrator] 阶段 %s 失败: %s", stage_name, e)
            return StageResult(
                stage=stage_name,
                status=StageStatus.FAILED,
                started_at=started_at,
                finished_at=datetime.now(timezone.utc),
                duration_ms=round(elapsed_ms, 2),
                error=str(e),
            )

    def _run_audio(self, file_path: str, asset_id: str) -> Dict[str, Any]:
        """执行音频分析 + 写入向量"""
        result = self._audio.analyze_file(file_path)

        # 如果有向量存储，写入 embedding
        if self._vector_store and "rms_mean" in result:
            self._write_audio_embedding(asset_id, result)

        return result

    def _write_audio_embedding(self, asset_id: str, audio_data: Dict[str, Any]):
        """将音频特征写入向量存储"""
        cfg = self._cfg
        # 构建简单的音频特征向量
        feature_vec = [
            audio_data.get("silence_ratio", 0.0),
            audio_data.get("rms_mean", 0.0),
            audio_data.get("zcr", 0.0),
            audio_data.get("duration_s", 0.0),
        ]
        # 填充到固定维度（后续替换为真正的 embedding 模型）
        dim = 32
        while len(feature_vec) < dim:
            feature_vec.append(0.0)
        feature_vec = feature_vec[:dim]

        emb_id = f"{asset_id}_audio_{cfg.vector_namespace}"
        try:
            self._vector_store.ensure_collection(cfg.vector_collection, dim=dim, namespace=cfg.vector_namespace)
            self._vector_store.upsert(
                collection=cfg.vector_collection,
                ids=[emb_id],
                vectors=[feature_vec],
                metadatas=[{"asset_id": asset_id, "stage": "audio"}],
                namespace=cfg.vector_namespace,
            )
            logger.info("[Orchestrator] 音频 embedding 写入: %s", emb_id)
        except Exception as e:
            logger.error("[Orchestrator] 写入向量失败: %s", e)

    def _save_envelope(self, envelope: ResultEnvelope):
        """将 ResultEnvelope 写入 MongoDB"""
        try:
            doc = envelope.model_dump(mode="json")
            self._db[self._cfg.results_collection].insert_one(doc)
            logger.debug("[Orchestrator] 结果写入 MongoDB: %s", envelope.pipeline.pipeline_id)
        except Exception as e:
            logger.error("[Orchestrator] 写入 MongoDB 失败: %s", e)

    def _update_job_status(self, job_id: str, status: str, **kwargs):
        """更新 job 状态"""
        try:
            update = {"$set": {"status": status, **kwargs}}
            self._db[self._cfg.jobs_collection].update_one({"job_id": job_id}, update)
        except Exception as e:
            logger.error("[Orchestrator] 更新 job 失败: %s", e)
