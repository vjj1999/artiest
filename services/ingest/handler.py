"""
Ingest Service — 资产注册处理器

接收 OSS 路径，注册 asset_id，写入 MongoDB assets 集合。
"""
from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from brain.logging import logger


@dataclass
class IngestConfig:
    """Ingest 配置"""
    mongo_uri: str = "mongodb://localhost:27017"
    db_name: str = "brain_db"
    assets_collection: str = "assets"
    jobs_collection: str = "jobs"


def _gen_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


class IngestHandler:
    """资产注册处理器"""

    def __init__(self, cfg: Optional[IngestConfig] = None, mongo_db: Any = None):
        self._cfg = cfg or IngestConfig()
        self._db = mongo_db

    def _ensure_db(self):
        if self._db is not None:
            return
        try:
            from pymongo import MongoClient
            client = MongoClient(self._cfg.mongo_uri)
            self._db = client[self._cfg.db_name]
            logger.info("[Ingest] MongoDB 已连接: %s", self._cfg.db_name)
        except Exception as e:
            raise RuntimeError(f"MongoDB 连接失败: {e}") from e

    def ingest(
        self,
        oss_path: str,
        asset_type: str = "video",
        filename: str = "",
        duration_s: Optional[float] = None,
        size_bytes: Optional[int] = None,
        stages: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        注册资产到 MongoDB

        Returns:
            {"asset_id": "...", "job_id": "...", "status": "queued"}
        """
        self._ensure_db()

        asset_id = _gen_id("ast")
        job_id = _gen_id("job")
        pipeline_id = _gen_id("pipe")
        now = datetime.now(timezone.utc)

        # 写入 assets 集合
        asset_doc = {
            "asset_id": asset_id,
            "asset_type": asset_type,
            "oss_path": oss_path,
            "filename": filename or oss_path.rsplit("/", 1)[-1],
            "duration_s": duration_s,
            "size_bytes": size_bytes,
            "created_at": now,
            "updated_at": now,
        }
        self._db[self._cfg.assets_collection].insert_one(asset_doc)
        logger.info("[Ingest] 资产注册: %s -> %s", asset_id, oss_path)

        # 写入 jobs 集合
        job_doc = {
            "job_id": job_id,
            "asset_id": asset_id,
            "pipeline_id": pipeline_id,
            "status": "queued",
            "stages": stages or ["audio"],
            "created_at": now,
            "started_at": None,
            "finished_at": None,
            "error": None,
        }
        self._db[self._cfg.jobs_collection].insert_one(job_doc)
        logger.info("[Ingest] 任务创建: %s (stages=%s)", job_id, stages or ["audio"])

        return {
            "asset_id": asset_id,
            "job_id": job_id,
            "pipeline_id": pipeline_id,
            "status": "queued",
        }

    def get_asset(self, asset_id: str) -> Optional[Dict[str, Any]]:
        """查询资产"""
        self._ensure_db()
        doc = self._db[self._cfg.assets_collection].find_one({"asset_id": asset_id}, {"_id": 0})
        return dict(doc) if doc else None
