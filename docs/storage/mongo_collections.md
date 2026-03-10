# MongoDB Collections — 集合设计

## 数据库

数据库名: `brain_db`

## 集合列表

| 集合 | 用途 | 主键 |
|------|------|------|
| `assets` | 资产注册表 | `asset_id` (unique) |
| `perception_results` | 感知结果（ResultEnvelope） | `pipeline.pipeline_id` (unique) |
| `jobs` | 异步任务队列 | `job_id` (unique) |

## assets

存储所有注册的资产（视频/音频/图片）元信息。

| 字段 | 类型 | 索引 | 说明 |
|------|------|------|------|
| `asset_id` | string | unique | 全局唯一 ID |
| `asset_type` | enum | index | video / audio / image |
| `oss_path` | string | — | OSS 存储路径 |
| `filename` | string | — | 原始文件名 |
| `duration_s` | double? | — | 时长 |
| `size_bytes` | long? | — | 文件大小 |
| `created_at` | date | index (desc) | 创建时间 |

## perception_results

存储感知流水线的输出结果，结构对应 `ResultEnvelope`。

| 字段 | 类型 | 索引 | 说明 |
|------|------|------|------|
| `asset_meta.asset_id` | string | index | 关联资产 |
| `pipeline.pipeline_id` | string | unique | 执行 ID |
| `pipeline.status` | enum | index | 执行状态 |
| `pipeline.started_at` | date | index (desc) | 开始时间 |
| `outputs` | array | — | 各阶段输出 |
| `embedding_ids` | array | — | 向量引用 |

## jobs

异步任务队列，记录感知任务的生命周期。

| 字段 | 类型 | 索引 | 说明 |
|------|------|------|------|
| `job_id` | string | unique | 任务 ID |
| `asset_id` | string | index | 关联资产 |
| `status` | enum | compound(status+created_at) | queued/running/success/failed/cancelled |
| `stages` | array | — | 要执行的阶段 |
| `created_at` | date | — | 创建时间 |

## 初始化

```bash
mongosh brain_db < deploy/mongo_init.js
```
