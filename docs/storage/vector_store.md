# Vector Store — 向量存储接口

## 概述

提供 Milvus 和 Qdrant 两种向量存储的统一抽象接口 `VectorStoreInterface`。通过 `namespace` 参数隔离不同 `embedding_version` 的向量空间。

## 接口方法

| 方法 | 说明 |
|------|------|
| `ensure_collection(collection, dim, namespace)` | 确保集合存在 |
| `upsert(collection, ids, vectors, metadatas, namespace)` | 写入/更新向量 |
| `search(collection, query_vector, top_k, namespace, filters)` | 相似度搜索 |
| `delete(collection, ids, namespace)` | 删除向量 |
| `drop_collection(collection, namespace)` | 删除集合 |

## Namespace 策略

集合名实际存储为 `{collection}_{namespace}`，例如：
- `perception_v1` — embedding model v1 的向量
- `perception_v2` — embedding model v2 的向量

这样切换模型版本时不影响旧数据，支持灰度切换。

## 配置

### Milvus

```python
MilvusConfig(
    host="localhost",
    port=19530,
    index_type="IVF_FLAT",  # IVF_FLAT / HNSW / IVF_SQ8
    metric_type="COSINE",
)
```

### Qdrant

```python
QdrantConfig(
    host="localhost",
    port=6333,
    prefer_grpc=True,
    distance="Cosine",  # Cosine / Euclid / Dot
)
```

## 使用示例

```python
from libs.clients.milvus_store import MilvusStore, MilvusConfig
from libs.clients.qdrant_store import QdrantStore, QdrantConfig

# 选择后端
store = MilvusStore(MilvusConfig())
# 或
store = QdrantStore(QdrantConfig())

# 创建集合
store.ensure_collection("perception", dim=192, namespace="v1")

# 写入
store.upsert(
    collection="perception",
    ids=["ast_001_audio"],
    vectors=[[0.1, 0.2, ...]],
    metadatas=[{"asset_id": "ast_001", "stage": "audio"}],
    namespace="v1",
)

# 搜索
results = store.search(
    collection="perception",
    query_vector=[0.1, 0.2, ...],
    top_k=5,
    namespace="v1",
    filters={"stage": "audio"},
)
```
