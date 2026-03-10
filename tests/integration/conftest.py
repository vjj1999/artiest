"""集成测试共享 fixtures: mongomock + Qdrant 内存模式"""
import sys
import tempfile
import wave
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import mongomock
from qdrant_client import QdrantClient

from libs.clients.qdrant_store import QdrantStore
from services.ingest.handler import IngestHandler, IngestConfig
from services.orchestrator.pipeline import Orchestrator, OrchestratorConfig
from services.orchestrator.search_handler import SearchHandler


@pytest.fixture
def mongo_db():
    """mongomock 内存数据库"""
    client = mongomock.MongoClient()
    db = client["brain_test"]
    yield db
    client.close()


@pytest.fixture
def qdrant_store():
    """Qdrant 内存向量存储"""
    client = QdrantClient(":memory:")
    store = QdrantStore(client=client)
    return store


@pytest.fixture
def ingest(mongo_db):
    """Ingest 处理器（接 mongomock）"""
    return IngestHandler(mongo_db=mongo_db)


@pytest.fixture
def orchestrator(mongo_db, qdrant_store):
    """Orchestrator（接 mongomock + Qdrant 内存）"""
    return Orchestrator(mongo_db=mongo_db, vector_store=qdrant_store)


@pytest.fixture
def search(qdrant_store):
    """Search 处理器（接 Qdrant 内存）"""
    return SearchHandler(qdrant_store)


def make_wav(duration_s: float = 2.0, freq: float = 440, amplitude: float = 0.5, sr: int = 16000) -> str:
    """生成临时 WAV 文件"""
    n = int(duration_s * sr)
    t = np.arange(n) / sr
    pcm = (amplitude * np.sin(2 * np.pi * freq * t) * 32767).astype(np.int16)
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    with wave.open(tmp.name, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(pcm.tobytes())
    return tmp.name


def make_silence_wav(duration_s: float = 2.0, sr: int = 16000) -> str:
    """生成静音 WAV"""
    n = int(duration_s * sr)
    pcm = np.zeros(n, dtype=np.int16)
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    with wave.open(tmp.name, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(pcm.tobytes())
    return tmp.name
