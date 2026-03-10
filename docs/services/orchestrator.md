# Orchestrator — 感知流水线编排器

## 职责

接收 analyze 请求，按阶段顺序执行感知任务，汇总结果到 `ResultEnvelope`。

## 流程

```
analyze(asset_id, file_path, stages)
    │
    ├─ 1. 从 MongoDB 查询资产信息
    ├─ 2. 更新 job 状态 → running
    ├─ 3. 逐阶段执行：
    │     ├─ audio  → AudioAnalyzer.analyze_file()
    │     ├─ visual → (P1)
    │     ├─ text   → (P1)
    │     ├─ align  → (P1)
    │     └─ fusion → (P1)
    ├─ 4. 写入 embedding 到 Vector Store
    ├─ 5. 写入 ResultEnvelope 到 MongoDB
    └─ 6. 更新 job 状态 → success/failed
```

## P0 支持的阶段

- `audio` — 音频分析（silence_ratio, rms_mean, zcr）

## 代码

- `services/orchestrator/pipeline.py` — `Orchestrator`
