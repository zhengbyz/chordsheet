"""阶段 0：madmom 冒烟测试。

只回答一个问题：madmom 在这套环境（Python 3.12 + numpy 2.x）上
**能不能真的跑出结果**，而不只是能 import。

合成一段节拍明确的 4/4 音频，跑两条 madmom 流水线：
  1. downbeat 跟踪 —— 阶段 2 要用的小节线
  2. 和弦识别     —— 阶段 3 要用的和弦

用法：
    .venv/bin/python scripts/stage0_madmom_smoke.py
"""

import sys
import time
import warnings

import numpy as np

warnings.filterwarnings("ignore")

SAMPLE_RATE = 44100
BPM = 120
BEATS_PER_BAR = 4

# 一个小节一个和弦，两遍 I-IV-V-I。用 MIDI 音高表示。
PROGRESSION = [
    ("C:maj", [60, 64, 67]),
    ("F:maj", [65, 69, 72]),
    ("G:maj", [67, 71, 74]),
    ("C:maj", [60, 64, 67]),
] * 2


def midi_to_hz(note: int) -> float:
    return 440.0 * 2 ** ((note - 69) / 12)


def pluck(freq: float, duration: float, amp: float = 1.0) -> np.ndarray:
    """一个带谐波和指数衰减的音，够像乐器让 CNN 有东西可看。"""
    n = int(duration * SAMPLE_RATE)
    t = np.arange(n) / SAMPLE_RATE
    envelope = np.exp(-3.5 * t)
    wave = np.zeros(n)
    for harmonic, weight in enumerate([1.0, 0.5, 0.3, 0.15, 0.08], start=1):
        wave += weight * np.sin(2 * np.pi * freq * harmonic * t)
    return amp * envelope * wave


def synth_audio() -> np.ndarray:
    """合成 8 小节 4/4、120 BPM 的和弦进行。"""
    beat_dur = 60.0 / BPM
    bar_dur = beat_dur * BEATS_PER_BAR
    total = int(len(PROGRESSION) * bar_dur * SAMPLE_RATE)
    audio = np.zeros(total)

    for bar_idx, (_label, notes) in enumerate(PROGRESSION):
        for beat in range(BEATS_PER_BAR):
            start = int((bar_idx * bar_dur + beat * beat_dur) * SAMPLE_RATE)
            # 第一拍重音 + 低八度根音，给 downbeat 跟踪器可抓的线索
            accent = 1.0 if beat == 0 else 0.55
            voices = list(notes)
            if beat == 0:
                voices.append(notes[0] - 12)
            for note in voices:
                tone = pluck(midi_to_hz(note), beat_dur * 0.95, amp=accent / len(voices))
                end = min(start + len(tone), total)
                audio[start:end] += tone[: end - start]

    peak = np.abs(audio).max()
    return (audio / peak * 0.9).astype(np.float32)


def main() -> int:
    print("=" * 60)
    print("madmom 冒烟测试 — Python", ".".join(map(str, sys.version_info[:3])))
    print("=" * 60)

    import madmom

    print(f"madmom  {madmom.__version__}")
    print(f"numpy   {np.__version__}")

    audio = synth_audio()
    duration = len(audio) / SAMPLE_RATE
    print(f"\n合成音频: {duration:.1f}s, {len(PROGRESSION)} 小节 @ {BPM} BPM")
    print(f"预期和弦: {' '.join(label for label, _ in PROGRESSION)}")

    signal = madmom.audio.Signal(audio, sample_rate=SAMPLE_RATE)
    failures = []

    # ---- 测试 1: downbeat 跟踪（阶段 2）----
    print("\n" + "-" * 60)
    print("[1/2] Downbeat 跟踪  RNNDownBeatProcessor + DBNDownBeatTrackingProcessor")
    print("-" * 60)
    try:
        from madmom.features.downbeats import (
            DBNDownBeatTrackingProcessor,
            RNNDownBeatProcessor,
        )

        t0 = time.perf_counter()
        activations = RNNDownBeatProcessor()(signal)
        tracker = DBNDownBeatTrackingProcessor(beats_per_bar=[3, 4], fps=100)
        beats = tracker(activations)
        elapsed = time.perf_counter() - t0

        downbeats = beats[beats[:, 1] == 1][:, 0]
        print(f"激活值 shape: {activations.shape}")
        print(f"检出 {len(beats)} 个拍点，其中 {len(downbeats)} 个小节线")
        print(f"小节线时间: {np.round(downbeats, 2).tolist()}")
        if len(beats) > 1:
            est_bpm = 60.0 / np.median(np.diff(beats[:, 0]))
            print(f"推算速度: {est_bpm:.1f} BPM (真值 {BPM})")
        print(f"耗时 {elapsed:.1f}s  (实时率 {duration / elapsed:.1f}x)")
        print("=> 通过")
    except Exception as exc:
        failures.append(("downbeat", exc))
        print(f"=> 失败: {type(exc).__name__}: {exc}")

    # ---- 测试 2: 和弦识别（阶段 3）----
    print("\n" + "-" * 60)
    print("[2/2] 和弦识别  CNNChordFeatureProcessor + CRFChordRecognitionProcessor")
    print("-" * 60)
    try:
        from madmom.features.chords import (
            CNNChordFeatureProcessor,
            CRFChordRecognitionProcessor,
        )

        t0 = time.perf_counter()
        features = CNNChordFeatureProcessor()(signal)
        chords = CRFChordRecognitionProcessor()(features)
        elapsed = time.perf_counter() - t0

        print(f"特征 shape: {features.shape}")
        print(f"检出 {len(chords)} 段和弦:")
        for start, end, label in chords:
            print(f"  {start:6.2f} - {end:6.2f}s   {label}")
        print(f"耗时 {elapsed:.1f}s  (实时率 {duration / elapsed:.1f}x)")
        print("=> 通过")
    except Exception as exc:
        failures.append(("chords", exc))
        print(f"=> 失败: {type(exc).__name__}: {exc}")

    print("\n" + "=" * 60)
    if failures:
        print(f"结论：{len(failures)} 项失败")
        for name, exc in failures:
            print(f"  - {name}: {type(exc).__name__}: {exc}")
        return 1
    print("结论：madmom 在 Python 3.12 上可用，阶段 2/3 的技术路线成立")
    return 0


if __name__ == "__main__":
    sys.exit(main())
