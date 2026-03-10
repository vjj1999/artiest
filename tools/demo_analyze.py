"""
模式2: 分析音频文件 — 读取 WAV，流式做 ASR + Diarization + 声纹识别

    python tools/demo_analyze.py output/record_xxx.wav
    python tools/demo_analyze.py output/record_xxx.wav --device cuda
    python tools/demo_analyze.py output/record_xxx.wav --no-voiceprint
"""
import argparse
import os
import sys
import wave
from pathlib import Path

import warnings
warnings.filterwarnings("ignore")

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

import numpy as np
import scipy.signal

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from brain.perception.audio import (
    PyannoteDiarizationConfig, diarize_pyannote,
    SpeakerMemory, VoiceprintConfig,
)

TARGET_RATE = 16000

# ── ASR（funasr AutoModel，支持长音频）──
_asr_model = None
HAS_ASR = True
try:
    from funasr import AutoModel
except Exception:
    HAS_ASR = False


def _ensure_asr():
    global _asr_model
    if _asr_model is not None:
        return
    print("  [ASR] 加载模型（首次较慢）...")
    _asr_model = AutoModel(
        model="paraformer-zh",
        vad_model="fsmn-vad",
        punc_model="ct-punc",
        disable_update=True,
    )


def asr_recognize(pcm_16k: np.ndarray) -> str:
    """对 16kHz float/int16 numpy 数组做 ASR"""
    if not HAS_ASR or pcm_16k is None or len(pcm_16k) == 0:
        return ""
    try:
        _ensure_asr()
        # AutoModel 接受 numpy float32 或 wav 路径
        if pcm_16k.dtype == np.int16:
            pcm_16k = pcm_16k.astype(np.float32) / 32768.0
        result = _asr_model.generate(input=pcm_16k, batch_size_s=300)
        if result and len(result) > 0:
            return result[0].get("text", "")
        return ""
    except Exception as e:
        return f"(ASR error: {e})"


def resample_to_16k(pcm_int16: np.ndarray, orig_sr: int) -> np.ndarray:
    if orig_sr == TARGET_RATE:
        return pcm_int16
    num_samples = int(len(pcm_int16) * TARGET_RATE / orig_sr)
    resampled = scipy.signal.resample(pcm_int16.astype(np.float64), num_samples)
    return np.clip(resampled, -32768, 32767).astype(np.int16)


def slice_pcm(pcm_bytes: bytes, start_ms: int, end_ms: int, sample_rate: int) -> bytes:
    start_b = max(0, int(start_ms * sample_rate / 1000)) * 2
    end_b = max(start_b, int(end_ms * sample_rate / 1000)) * 2
    return pcm_bytes[start_b:end_b]


def save_wav(pcm_bytes: bytes, filepath: str, sample_rate: int):
    os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)
    with wave.open(filepath, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm_bytes)


def main():
    parser = argparse.ArgumentParser(description="分析音频文件：ASR + Diarization + 声纹")
    parser.add_argument("wav_file", help="输入 WAV 文件路径")
    parser.add_argument("--device", type=str, default="cuda", help="推理设备")
    parser.add_argument("--no-asr", action="store_true")
    parser.add_argument("--no-diarization", action="store_true")
    parser.add_argument("--no-voiceprint", action="store_true")
    parser.add_argument("--vp-threshold", type=float, default=0.35)
    parser.add_argument("--save-turns", action="store_true", help="将每个 turn 保存为单独 WAV")
    args = parser.parse_args()

    # ── 读取 WAV ──
    wav_path = args.wav_file
    if not os.path.exists(wav_path):
        print(f"[error] 文件不存在: {wav_path}")
        return

    with wave.open(wav_path, 'rb') as wf:
        orig_sr = wf.getframerate()
        n_channels = wf.getnchannels()
        n_frames = wf.getnframes()
        raw_bytes = wf.readframes(n_frames)

    # 如果是立体声，取左声道
    raw_np = np.frombuffer(raw_bytes, dtype=np.int16)
    if n_channels > 1:
        raw_np = raw_np[::n_channels]

    duration_s = len(raw_np) / orig_sr

    print("=" * 60)
    print(" Brain Audio Perception — 音频分析")
    print("=" * 60)
    print(f"  文件:    {wav_path}")
    print(f"  时长:    {duration_s:.1f}s")
    print(f"  采样率:  {orig_sr}Hz -> {TARGET_RATE}Hz")
    print(f"  RMS:     {np.sqrt(np.mean((raw_np.astype(float)/32768)**2)):.4f}")
    print("=" * 60)

    # ── 重采样到 16kHz ──
    pcm_16k = resample_to_16k(raw_np, orig_sr)
    pcm_16k_bytes = pcm_16k.tobytes()

    # ── VAD: 标注语音/静音区间 ──
    print(f"\n[VAD] 检测语音活动...")
    from brain.perception.audio.vad import SileroVAD, VADConfig
    vad = SileroVAD(VADConfig(min_speech_duration_s=0.3, min_rms=0.0))

    # 流式喂入，收集所有语音片段
    vad_segments = []
    chunk_size = 512 * 2  # 512 samples * 2 bytes (int16)
    for i in range(0, len(pcm_16k_bytes), chunk_size * 32):
        chunk = pcm_16k_bytes[i:i + chunk_size * 32]
        segs = vad.feed(chunk)
        vad_segments.extend(segs)
    # flush: 强制结束最后可能还在缓冲的片段
    last_seg = vad.flush()
    if last_seg:
        vad_segments.append(last_seg)

    total_speech_s = sum(s.duration_s for s in vad_segments)
    total_silence_s = duration_s - total_speech_s
    print(f"[VAD] 语音: {total_speech_s:.1f}s | 静音: {total_silence_s:.1f}s | 片段数: {len(vad_segments)}")

    # 构建时间轴标注
    vad_timeline = []
    prev_end = 0.0
    for seg in vad_segments:
        if seg.start_time_s > prev_end + 0.1:
            vad_timeline.append(("silence", prev_end, seg.start_time_s))
        vad_timeline.append(("speech", seg.start_time_s, seg.end_time_s))
        prev_end = seg.end_time_s
    if prev_end < duration_s - 0.1:
        vad_timeline.append(("silence", prev_end, duration_s))

    for label, start, end in vad_timeline:
        dur = end - start
        tag = "  [speech]" if label == "speech" else "  [------]"
        print(f"  {tag}  {start:>6.1f}s - {end:>6.1f}s  ({dur:.1f}s)")

    # ── 整段 ASR ──
    do_asr = HAS_ASR and not args.no_asr
    if do_asr:
        print(f"\n[ASR] 整段识别...")
        text = asr_recognize(pcm_16k)
        print(f"[ASR] {text}")

    # ── Diarization ──
    if not args.no_diarization:
        print(f"\n[Diarization] 分析中...")
        token = os.getenv("HF_TOKEN", "")
        dia_cfg = PyannoteDiarizationConfig(hf_token=token, device=args.device)
        turns = diarize_pyannote(pcm_16k_bytes, dia_cfg)

        speakers = sorted(set(t.speaker_id for t in turns))
        print(f"[Diarization] {len(speakers)} 个说话人: {', '.join(speakers)}, {len(turns)} 个片段\n")

        # 声纹
        speaker_memory = None
        if not args.no_voiceprint:
            speaker_memory = SpeakerMemory(VoiceprintConfig(device=args.device, threshold=args.vp_threshold))

        out_dir = os.path.splitext(wav_path)[0] + "_turns"

        for i, t in enumerate(turns):
            t_start = t.start_ms / 1000
            t_end = t.end_ms / 1000
            t_dur = t_end - t_start

            line = f"  [{i+1:2d}] {t.speaker_id}  {t_start:>6.1f}s - {t_end:>6.1f}s  ({t_dur:.1f}s)"

            # 逐段 ASR
            if do_asr and t_dur >= 0.5:
                start_sample = int(t.start_ms * TARGET_RATE / 1000)
                end_sample = int(t.end_ms * TARGET_RATE / 1000)
                seg_np = pcm_16k[start_sample:end_sample]
                seg_text = asr_recognize(seg_np)
                if seg_text:
                    line += f'  "{seg_text}"'

            # 逐段声纹
            if speaker_memory and t_dur >= 0.8:
                seg_pcm = slice_pcm(pcm_16k_bytes, t.start_ms, t.end_ms, TARGET_RATE)
                vp_name, vp_score = speaker_memory.identify_or_register(seg_pcm, duration_s=t_dur)
                if vp_name:
                    line += f"  [{vp_name} {vp_score:.2f}]"

            print(line)

            # 保存 turn
            if args.save_turns:
                turn_raw = slice_pcm(raw_bytes, t.start_ms, t.end_ms, orig_sr)
                if turn_raw:
                    turn_file = os.path.join(out_dir, f"turn_{i+1:02d}_{t.speaker_id}_{t.start_ms}ms.wav")
                    save_wav(turn_raw, turn_file, orig_sr)

        if args.save_turns:
            print(f"\n  Turn 文件保存: {out_dir}/")

        # 声纹统计
        if speaker_memory and speaker_memory.speaker_count > 0:
            print(f"\n  声纹识别: {speaker_memory.speaker_count} 人")
            for name in speaker_memory.known_speakers:
                print(f"    - {name}")

    print(f"\n{'='*60}")


if __name__ == "__main__":
    main()
