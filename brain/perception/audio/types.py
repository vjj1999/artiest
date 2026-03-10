from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, ConfigDict, Field


class SpeakerTurn(BaseModel):
    """说话人区间（相对当前 utterance，毫秒）。"""

    start_ms: int = Field(ge=0)
    end_ms: int = Field(gt=0)
    speaker_id: str = Field(min_length=1)
    score: Optional[float] = None
    backend: str = Field(default="simple")

    model_config = ConfigDict(arbitrary_types_allowed=True)


class TranscriptSegment(BaseModel):
    """带说话人标签的转写片段（相对当前 utterance，毫秒）。"""

    start_ms: int = Field(ge=0)
    end_ms: int = Field(gt=0)
    speaker_id: str = Field(min_length=1)
    text: str = Field(default="")
    score: Optional[float] = None
    # 可选：音量标注（dBFS），用于 LLM 判断远近/噪声段
    rms_dbfs: Optional[float] = None
    peak_dbfs: Optional[float] = None

    model_config = ConfigDict(arbitrary_types_allowed=True)
