# Audio Service MVP — 音频分析

## 职责

对视频/音频文件执行基础音频感知。

## 输出指标

| 指标 | 类型 | 说明 |
|------|------|------|
| `silence_ratio` | float | 静音帧占比 (0~1) |
| `rms_mean` | float | 全局 RMS 均值 |
| `zcr` | float | 平均过零率 |
| `duration_s` | float | 音频时长 |
| `speech_segments` | list | 语音段列表 [{start_s, end_s}] |

## 流程

1. 加载音频（WAV 直接读，视频用 ffmpeg 提取）
2. 重采样到 16kHz mono
3. 逐帧计算 RMS 和 ZCR
4. 静音检测：RMS < 阈值的帧标记为静音
5. 语音段提取：连续非静音帧合并

## 代码

- `services/audio/analyzer.py` — `AudioAnalyzer`
