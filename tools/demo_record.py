"""
模式1: 纯录音 — 只录音保存，不做任何处理

    python tools/demo_record.py                     # 录 10 秒
    python tools/demo_record.py --seconds 20        # 录 20 秒
    python tools/demo_record.py --seconds 0         # 按 Enter 停
    python tools/demo_record.py -o output/my.wav    # 指定输出路径
"""
import argparse
import os
import sys
import time
import threading
import wave

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

import numpy as np
import sounddevice as sd


def get_native_sr(device=None) -> int:
    info = sd.query_devices(device=device, kind='input')
    return int(info['default_samplerate'])


def main():
    parser = argparse.ArgumentParser(description="纯录音保存")
    parser.add_argument("--seconds", type=int, default=10, help="录音时长，0=按Enter停止")
    parser.add_argument("-o", "--output", type=str, default="", help="输出 WAV 路径")
    parser.add_argument("--device", type=int, default=None, help="音频设备ID（留空用默认）")
    args = parser.parse_args()

    sr = get_native_sr(args.device)
    out_path = args.output or f"output/record_{time.strftime('%Y%m%d_%H%M%S')}.wav"
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    print(f"  采样率: {sr}Hz | 设备: {args.device or '默认'}")

    if args.seconds > 0:
        print(f"  录音 {args.seconds} 秒...")
        audio = sd.rec(int(args.seconds * sr), samplerate=sr, channels=1, dtype='int16', device=args.device)
        for i in range(args.seconds, 0, -1):
            print(f"  剩余 {i}s...", end='\r')
            time.sleep(1)
        sd.wait()
    else:
        print("  录音中... 按 Enter 停止")
        chunks = []
        stop = threading.Event()
        def cb(indata, frames, ti, status):
            if not stop.is_set():
                chunks.append(indata.copy())
        stream = sd.InputStream(samplerate=sr, channels=1, dtype='int16', device=args.device, callback=cb)
        stream.start()
        input()
        stop.set()
        stream.stop()
        stream.close()
        audio = np.concatenate(chunks, axis=0) if chunks else np.zeros((0, 1), dtype=np.int16)

    dur = len(audio) / sr
    with wave.open(out_path, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(audio.tobytes())

    print(f"  已保存: {out_path} ({dur:.1f}s, {sr}Hz, {os.path.getsize(out_path)} bytes)")


if __name__ == "__main__":
    main()
