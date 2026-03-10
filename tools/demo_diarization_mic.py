"""
麦克风实时录音 -> 说话人分离 演示

使用方式：
    python tools/demo_diarization_mic.py                    # 默认录 8 秒
    python tools/demo_diarization_mic.py --seconds 15       # 录 15 秒
    python tools/demo_diarization_mic.py --seconds 0        # 按 Enter 停止录音

测试建议：
    1. 一个人说一段话，停顿，另一个人说一段话（验证多说话人）
    2. 同一个人连续说两句话（验证单说话人）
    3. 两个人交替说话（验证说话人切换）
"""
import argparse
import os
import sys
import time
import threading
import wave
from datetime import datetime
from pathlib import Path

import numpy as np
import sounddevice as sd

# 项目根目录加入 sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from brain.perception.audio import PyannoteDiarizationConfig, diarize_pyannote, SpeakerTurn

SAMPLE_RATE = 16000
CHANNELS = 1


def record_fixed(seconds: int) -> bytes:
    """录制固定时长"""
    print(f"\n  [mic] 开始录音（{seconds} 秒）...")
    print("   请对着麦克风说话，可以多人轮流说\n")
    audio = sd.rec(int(seconds * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=CHANNELS, dtype='int16')
    for i in range(seconds, 0, -1):
        print(f"   剩余 {i} 秒...", end='\r')
        time.sleep(1)
    sd.wait()
    print("   录音完成！                ")
    return audio.tobytes()


def record_until_enter() -> bytes:
    """录制直到按 Enter"""
    print("\n  [mic] 开始录音（按 Enter 停止）...")
    print("   请对着麦克风说话，可以多人轮流说\n")

    chunks = []
    stop_flag = threading.Event()

    def callback(indata, frames, time_info, status):
        if not stop_flag.is_set():
            chunks.append(indata.copy())

    stream = sd.InputStream(samplerate=SAMPLE_RATE, channels=CHANNELS, dtype='int16', callback=callback)
    stream.start()
    input("   按 Enter 停止录音...")
    stop_flag.set()
    stream.stop()
    stream.close()

    if not chunks:
        return b""
    audio = np.concatenate(chunks, axis=0)
    print(f"   录音完成！时长 {len(audio) / SAMPLE_RATE:.1f} 秒")
    return audio.tobytes()


def save_wav(pcm_bytes: bytes, filepath: str):
    """保存为 WAV 文件"""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with wave.open(filepath, 'wb') as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(2)
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(pcm_bytes)


def main():
    parser = argparse.ArgumentParser(description="麦克风录音 -> 说话人分离")
    parser.add_argument("--seconds", type=int, default=8, help="录音时长（秒），0=按Enter停止")
    parser.add_argument("--device", type=str, default="cuda", help="推理设备: cuda / cpu")
    parser.add_argument("--save", action="store_true", help="保存录音为 WAV 文件")
    args = parser.parse_args()

    token = os.getenv("HF_TOKEN", "")
    if not token:
        print("[warn] 未设置 HF_TOKEN 环境变量，尝试不带 token 加载模型...")

    # 1. 录音
    if args.seconds > 0:
        pcm_bytes = record_fixed(args.seconds)
    else:
        pcm_bytes = record_until_enter()

    if not pcm_bytes:
        print("[error] 未录到音频")
        return

    duration_s = len(pcm_bytes) / (SAMPLE_RATE * 2)
    print(f"\n[info] 音频: {duration_s:.1f}s, {len(pcm_bytes)} bytes, {SAMPLE_RATE}Hz mono int16")

    # 可选：保存 WAV
    if args.save:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        wav_path = f"output/diarization_{timestamp}.wav"
        save_wav(pcm_bytes, wav_path)
        print(f"[save] {wav_path}")

    # 2. 说话人分离
    print(f"\n[diarization] 正在分析 (device={args.device})...")
    cfg = PyannoteDiarizationConfig(
        hf_token=token,
        device=args.device,
    )
    start = time.perf_counter()
    turns = diarize_pyannote(pcm_bytes, cfg)
    elapsed = (time.perf_counter() - start) * 1000

    # 3. 结果展示
    speakers = set(t.speaker_id for t in turns)
    print(f"\n{'='*60}")
    print(f" 说话人分离结果")
    print(f"{'='*60}")
    print(f"   检测到 {len(speakers)} 个说话人: {', '.join(sorted(speakers))}")
    print(f"   共 {len(turns)} 个语音片段")
    print(f"   处理耗时: {elapsed:.0f}ms")
    print(f"{'='*60}")

    print(f"\n{'说话人':<10} {'时间段':<20} {'时长':>6}")
    print("-" * 40)

    for turn in turns:
        start_s = turn.start_ms / 1000
        end_s = turn.end_ms / 1000
        dur_s = end_s - start_s
        time_range = f"{start_s:.1f}s - {end_s:.1f}s"
        print(f"{turn.speaker_id:<10} {time_range:<20} {dur_s:>5.1f}s")

    # 4. 统计
    print(f"\n{'='*60}")
    print(" 说话人统计")
    print(f"{'='*60}")
    for spk in sorted(speakers):
        spk_turns = [t for t in turns if t.speaker_id == spk]
        total_dur = sum((t.end_ms - t.start_ms) / 1000 for t in spk_turns)
        pct = total_dur / duration_s * 100 if duration_s > 0 else 0
        bar = "#" * int(pct / 2)
        print(f"   {spk:<8} {total_dur:>5.1f}s ({pct:>4.1f}%) {bar}")

    print(f"\n[done]")


if __name__ == "__main__":
    main()
