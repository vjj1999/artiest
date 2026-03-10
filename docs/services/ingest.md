# Ingest Service — 资产注册

## 职责

接收 OSS 路径，注册资产到 MongoDB `assets` 集合，创建感知任务到 `jobs` 集合。

## API

`POST /api/v1/ingest`

```json
{
  "oss_path": "oss://bucket/videos/sample.mp4",
  "asset_type": "video",
  "stages": ["audio"]
}
```

响应:
```json
{
  "asset_id": "ast_a1b2c3d4e5f6",
  "job_id": "job_x1y2z3w4v5u6",
  "pipeline_id": "pipe_m1n2o3p4q5r6",
  "status": "queued"
}
```

## 流程

1. 生成 `asset_id`（`ast_` + 12 位 hex）
2. 写入 `assets` 集合
3. 创建 `job` 文档，状态 `queued`
4. 返回 ID 信息

## 代码

- `services/ingest/handler.py` — `IngestHandler`
