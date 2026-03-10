/**
 * MongoDB 初始化脚本
 *
 * 创建三个核心集合：assets, perception_results, jobs
 * 执行方式：mongosh brain_db < deploy/mongo_init.js
 */

// ── assets: 资产注册表 ──
db.createCollection("assets", {
  validator: {
    $jsonSchema: {
      bsonType: "object",
      required: ["asset_id", "oss_path", "created_at"],
      properties: {
        asset_id:    { bsonType: "string", description: "全局唯一资产 ID" },
        asset_type:  { enum: ["video", "audio", "image"] },
        oss_path:    { bsonType: "string", description: "OSS 存储路径" },
        filename:    { bsonType: "string" },
        duration_s:  { bsonType: ["double", "null"] },
        size_bytes:  { bsonType: ["long", "int", "null"] },
        created_at:  { bsonType: "date" },
        updated_at:  { bsonType: "date" }
      }
    }
  }
});

db.assets.createIndex({ asset_id: 1 }, { unique: true });
db.assets.createIndex({ created_at: -1 });
db.assets.createIndex({ asset_type: 1 });

print("✓ assets collection created");

// ── perception_results: 感知结果（ResultEnvelope） ──
db.createCollection("perception_results", {
  validator: {
    $jsonSchema: {
      bsonType: "object",
      required: ["asset_meta", "pipeline"],
      properties: {
        asset_meta: {
          bsonType: "object",
          required: ["asset_id"],
          properties: {
            asset_id: { bsonType: "string" }
          }
        },
        pipeline: {
          bsonType: "object",
          required: ["pipeline_id"],
          properties: {
            pipeline_id: { bsonType: "string" },
            status:      { enum: ["pending", "running", "success", "failed", "skipped"] }
          }
        },
        outputs:       { bsonType: "array" },
        embedding_ids: { bsonType: "array" }
      }
    }
  }
});

db.perception_results.createIndex({ "asset_meta.asset_id": 1 });
db.perception_results.createIndex({ "pipeline.pipeline_id": 1 }, { unique: true });
db.perception_results.createIndex({ "pipeline.status": 1 });
db.perception_results.createIndex({ "pipeline.started_at": -1 });

print("✓ perception_results collection created");

// ── jobs: 异步任务队列 ──
db.createCollection("jobs", {
  validator: {
    $jsonSchema: {
      bsonType: "object",
      required: ["job_id", "asset_id", "status"],
      properties: {
        job_id:     { bsonType: "string", description: "任务 ID" },
        asset_id:   { bsonType: "string", description: "关联资产 ID" },
        pipeline_id:{ bsonType: "string" },
        status:     { enum: ["queued", "running", "success", "failed", "cancelled"] },
        stages:     { bsonType: "array", description: "要执行的阶段列表" },
        created_at: { bsonType: "date" },
        started_at: { bsonType: ["date", "null"] },
        finished_at:{ bsonType: ["date", "null"] },
        error:      { bsonType: ["string", "null"] }
      }
    }
  }
});

db.jobs.createIndex({ job_id: 1 }, { unique: true });
db.jobs.createIndex({ asset_id: 1 });
db.jobs.createIndex({ status: 1, created_at: 1 });

print("✓ jobs collection created");
print("MongoDB 初始化完成：assets, perception_results, jobs");
