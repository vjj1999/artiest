"""
实时对话监听 — 录音 + ASR + Speaker Diarization + 声纹记忆

固定时长录音，录完后依次做 ASR、说话人分离、声纹识别。
按 Ctrl+C 提前结束。

使用方式：
    python tools/demo_live_listen.py
    python tools/demo_live_listen.py --seconds 15
    python tools/demo_live_listen.py --seconds 0       # 按 Enter 停止
"""
import argparse
import os
import sys
import time
import threading
import wave
from datetime import datetime
from pathlib import Path

import warnings
warnings.filterwarnings("ignore")

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')

import numpy as np
import sounddevice as sd
import scipy.signal

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from brain.perception.audio import (
    PyannoteDiarizationConfig, diarize_pyannote, SpeakerTurn,
    SpeakerMemory, VoiceprintConfig,
)

# ── ASR ──
try:
    from funasr_onnx.paraformer_bin import Paraformer as ParaformerOffline
    from funasr_onnx.punc_bin import CT_Transformer
    _asr_model = None
    _punc_model = None
    HAS_ASR = True
except Exception:
    HAS_ASR = False


def _ensure_asr():
    global _asr_model, _punc_model
    if _asr_model is not None:
        return
    _asr_model = ParaformerOffline(
        model_dir="iic/speech_paraformer-large-vad-punc_asr_nat-zh-cn-16k-common-vocab8404-onnx",
        quantize=True,
    )
    _punc_model = CT_Transformer(
        model_dir="iic/punc_ct-transformer_zh-cn-common-vocab272727-onnx",
        quantize=True,
    )


def asr_recognize(pcm_16k_bytes: bytes) -> str:
    if not HAS_ASR or not pcm_16k_bytes:
        return ""
    try:
        _ensure_asr()
        pcm_np = np.frombuffer(pcm_16k_bytes, dtype=np.int16).astype(np.float64) / 32768
        result = _asr_model(pcm_np)
        raw = result[0]['preds'][0] if result and result[0].get('preds') else ""
        if raw and _punc_model:
            return _punc_model(raw)[0]
        return raw
    except Exception as e:
        return f"(ASR error: {e})"


TARGET_RATE = 16000
CHANNELS = 1


def get_native_sample_rate() -> int:
    info = sd.query_devices(kind='input')
    return int(info['default_samplerate'])


def resample_to_16k(pcm_int16: np.ndarray, orig_sr: int) -> np.ndarray:
    if orig_sr == TARGET_RATE:
        return pcm_int16
    num_samples = int(len(pcm_int16) * TARGET_RATE / orig_sr)
    resampled = scipy.signal.resample(pcm_int16.astype(np.float64), num_samples)
    return np.clip(resampled, -32768, 32767).astype(np.int16)


def save_wav(pcm_bytes: bytes, filepath: str, sample_rate: int):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with wave.open(filepath, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm_bytes)


def record_fixed(seconds: int, native_sr: int) -> bytes:
    print(f"\n  [mic] 开始录音（{seconds} 秒，{native_sr}Hz）...")
    audio = sd.rec(int(seconds * native_sr), samplerate=native_sr, channels=CHANNELS, dtype='int16')
    for i in range(seconds, 0, -1):
        print(f"   剩余 {i} 秒...", end='\r')
        time.sleep(1)
    sd.wait()
    print("   录音完成！              ")
    return audio.tobytes()


def record_until_enter(native_sr: int) -> bytes:
    print(f"\n  [mic] 开始录音（按 Enter 停止，{native_sr}Hz）...")
    chunks = []
    stop_flag = threading.Event()

    def callback(indata, frames, time_info, status):
        if not stop_flag.is_set():
            chunks.append(indata.copy())

    stream = sd.InputStream(samplerate=native_sr, channels=CHANNELS, dtype='int16', callback=callback)
    stream.start()
    input("   按 Enter 停止录音...")
    stop_flag.set()
    stream.stop()
    stream.close()

    if not chunks:
        return b""
    audio = np.concatenate(chunks, axis=0)
    print(f"   录音完成！时长 {len(audio) / native_sr:.1f} 秒")
    return audio.tobytes()


def slice_pcm(pcm_bytes: bytes, start_ms: int, end_ms: int, sample_rate: int) -> bytes:
    start_b = max(0, int(start_ms * sample_rate / 1000)) * 2
    end_b = max(start_b, int(end_ms * sample_rate / 1000)) * 2
    return pcm_bytes[start_b:end_b]


def main():
    parser = argparse.ArgumentParser(description="录音 -> ASR + Diarization + 声纹识别")
    parser.add_argument("--seconds", type=int, default=10, help="录音时长（秒），0=按Enter停止")
    parser.add_argument("--device", type=str, default="cuda", help="推理设备: cuda/cpu")
    parser.add_argument("--output-dir", type=str, default="output/listen", help="保存目录")
    parser.add_argument("--no-asr", action="store_true", help="跳过 ASR")
    parser.add_argument("--no-diarization", action="store_true", help="跳过说话人分离")
    parser.add_argument("--no-voiceprint", action="store_true", help="跳过声纹识别")
    parser.add_argument("--vp-threshold", type=float, default=0.35, help="声纹匹配阈值")
    args = parser.parse_args()

    token = os.getenv("HF_TOKEN", "")
    native_sr = get_native_sample_rate()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(args.output_dir, timestamp)

    do_asr = HAS_ASR and not args.no_asr
    dia_cfg = None if args.no_diarization else PyannoteDiarizationConfig(hf_token=token, device=args.device)
    speaker_memory = None
    if not args.no_voiceprint:
        speaker_memory = SpeakerMemory(VoiceprintConfig(device=args.device, threshold=args.vp_threshold))

    print("=" * 60)
    print(" Brain Audio Perception")
    print("=" * 60)
    print(f"  Mic:           {native_sr}Hz (原始保存)")
    print(f"  ASR:           {'FunASR Paraformer' if do_asr else '关闭'}")
    print(f"  Diarization:   {'pyannote 3.1' if dia_cfg else '关闭'}")
    print(f"  Voiceprint:    {'ECAPA (threshold={:.2f})'.format(args.vp_threshold) if speaker_memory else '关闭'}")
    print("=" * 60)

    # ── 录音 ──
    if args.seconds > 0:
        raw_pcm = record_fixed(args.seconds, native_sr)
    else:
        raw_pcm = record_until_enter(native_sr)

    if not raw_pcm:
        print("[error] 未录到音频")
        return

    duration_s = len(raw_pcm) / (native_sr * 2)
    print(f"\n  音频: {duration_s:.1f}s, {native_sr}Hz")

    # 保存原始录音
    os.makedirs(output_dir, exist_ok=True)
    raw_path = os.path.join(output_dir, "recording.wav")
    save_wav(raw_pcm, raw_path, native_sr)
    print(f"  已保存: {raw_path}")

    # ── 重采样到 16kHz（用于处理）──
    raw_np = np.frombuffer(raw_pcm, dtype=np.int16)
    pcm_16k = resample_to_16k(raw_np, native_sr)
    pcm_16k_bytes = pcm_16k.tobytes()

    # ── ASR ──
    if do_asr:
        print(f"\n  [ASR] 识别中...")
        text = asr_recognize(pcm_16k_bytes)
        print(f"  [ASR] 结果: {text}")

    # ── Diarization ──
    if dia_cfg is not None:
        print(f"\n  [Diarization] 分析中...")
        turns = diarize_pyannote(pcm_16k_bytes, dia_cfg)
        speakers = sorted(set(t.speaker_id for t in turns))
        print(f"  [Diarization] 检测到 {len(speakers)} 个说话人: {', '.join(speakers)}")

        for t in turns:
            t_start = t.start_ms / 1000
            t_end = t.end_ms / 1000
            t_dur = t_end - t_start
            line = f"    {t.speaker_id}: {t_start:.1f}s - {t_end:.1f}s"

            # 逐段 ASR
            if do_asr and t_dur >= 0.5:
                seg_pcm = slice_pcm(pcm_16k_bytes, t.start_ms, t.end_ms, TARGET_RATE)
                seg_text = asr_recognize(seg_pcm)
                if seg_text:
                    line += f"  \"{seg_text}\""

            # 逐段声纹
            if speaker_memory is not None and t_dur >= 0.8:
                seg_pcm = slice_pcm(pcm_16k_bytes, t.start_ms, t.end_ms, TARGET_RATE)
                vp_name, vp_score = speaker_memory.identify_or_register(seg_pcm, duration_s=t_dur)
                if vp_name:
                    line += f"  [{vp_name} {vp_score:.2f}]"

            print(line)

            # 保存每个 turn 的原始音频
            turn_raw = slice_pcm(raw_pcm, t.start_ms, t.end_ms, native_sr)
            if turn_raw:
                turn_file = f"turn_{t.speaker_id}_{t.start_ms}ms.wav"
                save_wav(turn_raw, os.path.join(output_dir, turn_file), native_sr)

    # ── 结束报告 ──
    print(f"\n{'='*60}")
    print(" 完成")
    print(f"{'='*60}")
    if speaker_memory and speaker_memory.speaker_count > 0:
        print(f"  识别人数: {speaker_memory.speaker_count}")
        for name in speaker_memory.known_speakers:
            print(f"    - {name}")
    print(f"  保存目录: {output_dir}/")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
