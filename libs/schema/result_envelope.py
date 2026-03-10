"""
Result Envelope — 感知结果统一信封

所有感知流水线的输出都封装为 ResultEnvelope，写入 MongoDB perception_results 集合。
"""
from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field


class AssetType(str, Enum):
    """资产类型"""
    VIDEO = "video"
    AUDIO = "audio"
    IMAGE = "image"


class AssetMeta(BaseModel):
    """资产元信息"""
    asset_id: str = Field(..., description="全局唯一资产 ID")
    asset_type: AssetType = Field(default=AssetType.VIDEO)
    oss_path: str = Field(..., description="OSS 存储路径")
    filename: str = Field(default="")
    duration_s: Optional[float] = Field(default=None, description="时长（秒）")
    size_bytes: Optional[int] = Field(default=None, description="文件大小")
    created_at: datetime = Field(default_factory=datetime.utcnow)

    model_config = ConfigDict(use_enum_values=True)


class StageStatus(str, Enum):
    """阶段执行状态"""
    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"
    SKIPPED = "skipped"


class StageResult(BaseModel):
    """单个感知阶段的输出"""
    stage: str = Field(..., description="阶段名: audio / visual / text / align / fusion")
    status: StageStatus = Field(default=StageStatus.PENDING)
    started_at: Optional[datetime] = None
    finished_at: Optional[datetime] = None
    duration_ms: Optional[float] = None
    data: Dict[str, Any] = Field(default_factory=dict, description="阶段输出数据")
    error: Optional[str] = None

    model_config = ConfigDict(use_enum_values=True)


class EmbeddingRef(BaseModel):
    """向量存储引用"""
    store: str = Field(..., description="向量存储标识: milvus / qdrant")
    collection: str = Field(..., description="集合名")
    embedding_id: str = Field(..., description="向量 ID")
    embedding_version: str = Field(default="v1", description="嵌入模型版本")
    dim: int = Field(..., description="向量维度")


class PipelineInfo(BaseModel):
    """流水线执行信息"""
    pipeline_id: str = Field(..., description="本次执行 ID")
    stages: List[str] = Field(default_factory=list, description="已执行的阶段列表")
    status: StageStatus = Field(default=StageStatus.PENDING)
    started_at: Optional[datetime] = None
    finished_at: Optional[datetime] = None
    total_duration_ms: Optional[float] = None
    config: Dict[str, Any] = Field(default_factory=dict, description="执行时配置快照")

    model_config = ConfigDict(use_enum_values=True)


class ResultEnvelope(BaseModel):
    """
    感知结果统一信封

    每个资产（视频/音频）经过感知流水线后，输出一个 ResultEnvelope，
    包含：资产元信息、流水线执行状态、各阶段输出、向量引用。
    """
    asset_meta: AssetMeta
    pipeline: PipelineInfo
    outputs: List[StageResult] = Field(default_factory=list)
    embedding_ids: List[EmbeddingRef] = Field(default_factory=list)

    model_config = ConfigDict(
        json_schema_extra={
            "title": "ResultEnvelope",
            "description": "多模态感知结果统一信封",
        }
    )
