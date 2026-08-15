"""阶段 2：节拍与小节线检测的准确率。

扫「速度 × 拍号」的组合，每组合成一段真值已知的音频，检查三件事：
  1. 拍号选对没有（3/4 还是 4/4）
  2. 速度准不准，以及有没有倍速/半速错误
  3. 小节线的相位对不对（是否落在真正的强拍上）

第 3 条最要紧。速度对、拍号对，但小节线整体偏移一拍，下游按小节切和弦就全错位。

用法：
    .venv/bin/python scripts/stage2_beats_smoke.py
"""

import sys
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from transcribe.beats import MADMOM_SAMPLE_RATE, track_beats  # noqa: E402

SR = MADMOM_SAMPLE_RATE
BARS = 12
TEMPOS = (72, 96, 120, 144)
METERS = (3, 4)


def click(freq: float, duration: float, amp: float, decay: float) -> np.ndarray:
    """一个短促的打击音。节拍跟踪主要靠瞬态起音，所以衰减要快。"""
    n = int(duration * SR)
    t = np.arange(n) / SR
    noise = np.random.default_rng(0).standard_normal(n) * 0.3
    return amp * np.exp(-decay * t) * (np.sin(2 * np.pi * freq * t) + noise)


def synth(bpm: float, meter: int) -> np.ndarray:
    """合成 BARS 个小节，强拍用低频重音，弱拍用高频轻击。"""
    beat_dur = 60.0 / bpm
    total = int(BARS * meter * beat_dur * SR)
    audio = np.zeros(total)

    for bar in range(BARS):
        for beat in range(meter):
            start = int((bar * meter + beat) * beat_dur * SR)
            if beat == 0:
                hit = click(60.0, 0.25, 1.0, 30.0)  # 底鼓式重音标记小节线
            else:
                hit = click(900.0, 0.08, 0.35, 80.0)  # 轻击标记普通拍
            end = min(start + len(hit), total)
            audio[start:end] += hit[: end - start]

        # 每小节一个持续和弦，给 RNN 一点和声节奏的线索
        chord_start = int(bar * meter * beat_dur * SR)
        chord_len = int(meter * beat_dur * SR)
        t = np.arange(chord_len) / SR
        for note in (48, 52, 55):
            freq = 440.0 * 2 ** ((note - 69) / 12)
            audio[chord_start : chord_start + chord_len] += (
                0.12 * np.exp(-1.2 * t) * np.sin(2 * np.pi * freq * t)
            )

    return (audio / np.abs(audio).max() * 0.9).astype(np.float32)


def phase_error(downbeats: np.ndarray, bar_dur: float) -> float:
    """小节线偏离真值的平均绝对误差（秒）。真值是 0, bar_dur, 2*bar_dur, ...

    对每个检出的小节线，找最近的真值小节线算距离。
    """
    if len(downbeats) == 0:
        return float("inf")
    nearest = np.round(downbeats / bar_dur) * bar_dur
    return float(np.mean(np.abs(downbeats - nearest)))


def main() -> int:
    print("=" * 78)
    print("阶段 2：节拍与小节线检测准确率")
    print("=" * 78)
    print(f"每组合成 {BARS} 小节，强拍低频重音、弱拍高频轻击、每小节一个持续和弦")
    print(
        f"{'真值':<14} {'检出拍号':<9} {'检出BPM':<10} {'速度误差':<10} {'相位误差':<11} {'判定'}"
    )
    print("-" * 78)

    failures = []
    for meter in METERS:
        for bpm in TEMPOS:
            truth = f"{bpm} BPM {meter}/4"
            beat_dur = 60.0 / bpm
            bar_dur = beat_dur * meter

            result = track_beats(synth(bpm, meter), SR, meters=METERS)

            meter_ok = result.meter == meter
            ratio = result.tempo / bpm
            tempo_err = abs(ratio - 1)
            # 倍速/半速是节拍跟踪最经典的失败模式，单独识别出来
            octave = (
                ""
                if tempo_err < 0.05
                else (
                    " [倍速]"
                    if abs(ratio - 2) < 0.1
                    else (" [半速]" if abs(ratio - 0.5) < 0.1 else "")
                )
            )
            phase_err = phase_error(result.downbeats, bar_dur)
            # 相位误差超过 1/4 拍就算错位
            phase_ok = phase_err < beat_dur * 0.25

            ok = meter_ok and tempo_err < 0.05 and phase_ok
            if not ok:
                failures.append((truth, result.meter, result.tempo, phase_err))

            print(
                f"{truth:<14} {str(result.meter) + '/4':<9} {result.tempo:<10.1f} "
                f"{tempo_err:<10.1%} {phase_err * 1000:>7.1f}ms   "
                f"{'通过' if ok else '失败'}{octave}"
                f"{'' if meter_ok else ' [拍号错]'}{'' if phase_ok else ' [相位错]'}"
            )

    print("-" * 78)
    total = len(TEMPOS) * len(METERS)
    print(f"通过 {total - len(failures)}/{total}")
    print()
    if failures:
        print("结论：存在失败用例，见上方明细")
        return 1
    print("结论：madmom 的节拍与小节线检测在合成音频上全部通过")
    print("注意这是理想条件（严格等间距、重音清晰、不变速），真实录音尤其是")
    print("自由速度的独奏钢琴会难得多——看 tempo_stability 判断可信度")
    return 0


if __name__ == "__main__":
    sys.exit(main())
