# Result Envelope — 感知结果统一信封

## 概述

`ResultEnvelope` 是多模态感知流水线的统一输出格式。每个资产（视频/音频/图片）经过感知流水线后，产出一个 Envelope 存入 MongoDB `perception_results` 集合。

## 顶层结构

| 字段 | 类型 | 说明 |
|------|------|------|
| `asset_meta` | AssetMeta | 资产元信息 |
| `pipeline` | PipelineInfo | 流水线执行信息 |
| `outputs` | List[StageResult] | 各感知阶段输出 |
| `embedding_ids` | List[EmbeddingRef] | 向量存储引用 |

## AssetMeta

| 字段 | 类型 | 说明 |
|------|------|------|
| `asset_id` | str | 全局唯一资产 ID |
| `asset_type` | enum | video / audio / image |
| `oss_path` | str | OSS 存储路径 |
| `filename` | str | 原始文件名 |
| `duration_s` | float? | 时长（秒） |
| `size_bytes` | int? | 文件大小 |
| `created_at` | datetime | 创建时间 |

## PipelineInfo

| 字段 | 类型 | 说明 |
|------|------|------|
| `pipeline_id` | str | 本次执行 ID |
| `stages` | List[str] | 已执行的阶段 |
| `status` | enum | pending / running / success / failed / skipped |
| `started_at` | datetime? | 开始时间 |
| `finished_at` | datetime? | 结束时间 |
| `total_duration_ms` | float? | 总耗时（ms） |
| `config` | dict | 执行时配置快照 |

## StageResult

| 字段 | 类型 | 说明 |
|------|------|------|
| `stage` | str | 阶段名: audio / visual / text / align / fusion |
| `status` | enum | pending / running / success / failed / skipped |
| `started_at` | datetime? | 开始时间 |
| `finished_at` | datetime? | 结束时间 |
| `duration_ms` | float? | 耗时（ms） |
| `data` | dict | 阶段输出数据（结构因阶段而异） |
| `error` | str? | 错误信息 |

### audio 阶段 data 示例

```json
{
  "silence_ratio": 0.15,
  "rms_mean": 0.042,
  "zcr": 0.08,
  "speech_segments": [{"start_s": 0.5, "end_s": 3.2}]
}
```

## EmbeddingRef

| 字段 | 类型 | 说明 |
|------|------|------|
| `store` | str | 向量存储标识: milvus / qdrant |
| `collection` | str | 集合名 |
| `embedding_id` | str | 向量 ID |
| `embedding_version` | str | 嵌入模型版本 |
| `dim` | int | 向量维度 |

## 示例

```json
{
  "asset_meta": {
    "asset_id": "ast_001",
    "asset_type": "video",
    "oss_path": "oss://bucket/videos/sample.mp4",
    "filename": "sample.mp4",
    "duration_s": 30.5,
    "size_bytes": 15728640
  },
  "pipeline": {
    "pipeline_id": "pipe_20260213_001",
    "stages": ["audio"],
    "status": "success",
    "total_duration_ms": 1250.0
  },
  "outputs": [
    {
      "stage": "audio",
      "status": "success",
      "duration_ms": 1250.0,
      "data": {
        "silence_ratio": 0.15,
        "rms_mean": 0.042,
        "zcr": 0.08
      }
    }
  ],
  "embedding_ids": [
    {
      "store": "milvus",
      "collection": "perception",
      "embedding_id": "ast_001_audio_v1",
      "embedding_version": "v1",
      "dim": 192
    }
  ]
}
```
