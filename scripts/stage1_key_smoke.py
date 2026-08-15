"""阶段 1：调式识别在 24 个调上的准确率。

对 12 大调 + 12 小调各合成一段确立调性的和弦进行（真值已知），
跑识别，统计准确率，并单独盯住关系大小调的混淆情况。

小调用**和声小调**（升 VII 级导音）。这一点很要紧：自然小调的音级集合
和它的关系大调完全相同，用自然小调测等于在测运气。真实音乐里正是那个
升高的导音在区分 A 小调和 C 大调。

用法：
    .venv/bin/python scripts/stage1_key_smoke.py
"""

import sys
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from chordsheet.key import PITCH_CLASSES, PROFILES, detect_key, mean_chroma  # noqa: E402

SAMPLE_RATE = 22050
CHORD_DUR = 1.2

# 相对主音的和弦进行，(根音半音数, 三和弦音程结构)
# 大调 I-IV-V-I，小调 i-iv-V-i（V 用大三和弦，即和声小调的升导音）
MAJOR_PROGRESSION = [(0, (0, 4, 7)), (5, (0, 4, 7)), (7, (0, 4, 7)), (0, (0, 4, 7))]
MINOR_PROGRESSION = [(0, (0, 3, 7)), (5, (0, 3, 7)), (7, (0, 4, 7)), (0, (0, 3, 7))]


def pluck(freq: float, duration: float, amp: float = 1.0) -> np.ndarray:
    n = int(duration * SAMPLE_RATE)
    t = np.arange(n) / SAMPLE_RATE
    wave = sum(
        w * np.sin(2 * np.pi * freq * h * t)
        for h, w in enumerate([1.0, 0.5, 0.3, 0.15, 0.08], start=1)
    )
    return amp * np.exp(-2.0 * t) * wave


def synth_key(tonic: int, is_minor: bool) -> np.ndarray:
    """合成某个调的 I-IV-V-I，tonic 为 0-11 的音级序号。"""
    progression = MINOR_PROGRESSION if is_minor else MAJOR_PROGRESSION
    chunks = []
    for degree, intervals in progression:
        n = int(CHORD_DUR * SAMPLE_RATE)
        chord = np.zeros(n)
        root_midi = 60 + tonic + degree
        for interval in intervals:
            note = root_midi + interval
            freq = 440.0 * 2 ** ((note - 69) / 12)
            chord += pluck(freq, CHORD_DUR, amp=0.3)
        # 低八度根音，强调功能
        chord += pluck(440.0 * 2 ** ((root_midi - 12 - 69) / 12), CHORD_DUR, amp=0.3)
        chunks.append(chord)
    audio = np.concatenate(chunks)
    return (audio / np.abs(audio).max() * 0.9).astype(np.float32)


def main() -> int:
    print("=" * 64)
    print("阶段 1：调式识别 24 调准确率")
    print("=" * 64)
    print(f"每个调合成 I-IV-V-I，共 {4 * CHORD_DUR:.1f}s；小调用和声小调")

    cases = [
        (f"{PITCH_CLASSES[t]} {'minor' if m else 'major'}", t, m) for m in (0, 1) for t in range(12)
    ]

    # chroma 只算一次，两个模板复用
    chromas = {name: mean_chroma(synth_key(t, bool(m)), SAMPLE_RATE) for name, t, m in cases}

    overall_ok = True
    for profile in PROFILES:
        print("\n" + "-" * 64)
        print(f"模板: {profile}")
        print("-" * 64)

        wrong, relative_confusions, margins = [], [], []
        for truth, _t, _m in cases:
            result = detect_key(chromas[truth], profile=profile)
            margins.append(result.margin)
            if result.key != truth:
                wrong.append((truth, result.key, result.margin))
                if result.relative_key == truth:
                    relative_confusions.append(truth)

        correct = len(cases) - len(wrong)
        print(f"准确率: {correct}/{len(cases)}  ({correct / len(cases):.0%})")
        print(f"平均领先幅度: {np.mean(margins):+.3f}  最小 {np.min(margins):+.3f}")
        if wrong:
            overall_ok = False
            print(f"错判 {len(wrong)} 个（其中 {len(relative_confusions)} 个是关系调混淆）:")
            for truth, got, margin in wrong:
                tag = " [关系调]" if truth in relative_confusions else ""
                print(f"  {truth:<10} → 判成 {got:<10} (领先 {margin:+.3f}){tag}")
        else:
            print("全对")

    print("\n" + "=" * 64)
    if overall_ok:
        print("结论：两个模板在合成音频上均 24/24，算法实现正确")
        print("注意这是理想条件（音高精确、无鼓无人声、不转调），真实录音会明显更难")
        return 0
    print("结论：存在错判，见上方明细")
    return 1


if __name__ == "__main__":
    sys.exit(main())
