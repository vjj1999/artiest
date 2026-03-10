# Search Endpoint — 向量搜索

## 职责

基于 VectorStoreInterface 执行 top_k 相似度搜索，支持 metadata 过滤。

## API

`POST /api/v1/search`

```json
{
  "query_vector": [0.1, 0.2, 0.3, ...],
  "top_k": 10,
  "namespace": "v1",
  "filters": {"stage": "audio"}
}
```

响应:
```json
{
  "results": [
    {"id": "ast_001_audio_v1", "score": 0.95, "metadata": {"asset_id": "ast_001", "stage": "audio"}},
    ...
  ]
}
```

## 配置

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `default_collection` | perception | 默认集合 |
| `default_namespace` | v1 | 默认命名空间 |
| `default_top_k` | 10 | 默认返回数 |
| `max_top_k` | 100 | 最大返回数 |

## 代码

- `services/orchestrator/search_handler.py` — `SearchHandler`
