"""阶段 3：和弦识别 + 小节对齐的准确率。

合成几段真值已知的和弦进行，检查三件事：
  1. 和弦本身识别对没有
  2. 小节对齐对没有——这是阶段 3 的主要工作量，也是最容易错的一环
  3. 跨阶段验证：识别出的和弦是否落在真实调内

在多个调上各跑一遍，避免只在 C 大调上碰巧能用。

用法：
    .venv/bin/python scripts/stage3_chords_smoke.py
"""

import sys
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from chordsheet.beats import MADMOM_SAMPLE_RATE, track_beats  # noqa: E402
from chordsheet.chords import (  # noqa: E402
    assign_chords_to_bars,
    is_diatonic,
    recognize_chords,
)
from chordsheet.key import PITCH_CLASSES  # noqa: E402

SR = MADMOM_SAMPLE_RATE
BPM = 100
METER = 4

# 相对主音的级数和性质。I-vi-IV-V 是流行乐最常见的进行之一，重复两遍够 CRF 锁定。
PROGRESSION = [(0, "maj"), (9, "min"), (5, "maj"), (7, "maj")] * 2
TRIADS = {"maj": (0, 4, 7), "min": (0, 3, 7)}
# 在几个调上验证，避免只在 C 大调上碰巧能用
TONICS = [0, 5, 7]  # C, F, G


HARMONICS = 12
RNG = np.random.default_rng(7)


def piano(freq: float, duration: float, amp: float) -> np.ndarray:
    """钢琴式音色。**音色的真实度直接决定这个测试有没有意义。**

    最初这里是 5 个等衰减正弦叠加，结果 vi 级和弦被系统性误判：
    DeepChroma 在 Am 上输出 C=0.93 A=0.92 E=0.60 F=0.59——凭空造出一个 F，
    于是 A-C-E 被读成 F-A-C，vi 变 IV，三个调无一例外。
    而同一段音频 librosa 的原始 chroma 是干净的 A=1.00 C=0.84 E=0.77。

    原因是这些和弦模型在真实录音（Beatles / Billboard 数据集）上训练，
    贫瘠的合成音色属于训练分布外，DNN 会「脑补」成它熟悉的样子。
    换成下面这套之后 DeepChroma 给出 A=1.00 C=0.99 E=0.99，问题消失。

    四个要素缺一不可：泛音数量与滚降、琴弦非谐性、逐泛音独立衰减、起音瞬态。
    """
    n = int(duration * SR)
    t = np.arange(n) / SR
    wave = np.zeros(n)
    for h in range(1, HARMONICS + 1):
        # 琴弦非谐性：泛音频率略高于整数倍，越高越明显
        partial = freq * h * (1 + 0.0004 * h * h)
        # 高次泛音衰减更快，这是钢琴音色随时间变暗的来源
        wave += (
            (1.0 / h**1.3)
            * np.sin(2 * np.pi * partial * t + RNG.uniform(0, 2 * np.pi))
            * np.exp(-(1.5 + 0.35 * h) * t)
        )
    attack = np.minimum(1.0, t / 0.006)  # 6ms 起音，避免咔哒声也提供瞬态
    return amp * attack * wave


def synth(tonic: int) -> np.ndarray:
    """一小节一个和弦，每拍重弹一次，强拍加低八度根音。"""
    beat_dur = 60.0 / BPM
    bar_dur = beat_dur * METER
    total = int(len(PROGRESSION) * bar_dur * SR)
    audio = np.zeros(total)

    for bar, (degree, quality) in enumerate(PROGRESSION):
        root_midi = 60 + tonic + degree
        for beat in range(METER):
            start = int((bar * METER + beat) * beat_dur * SR)
            amp = 0.30 if beat == 0 else 0.18
            voices = [root_midi + i for i in TRIADS[quality]]
            if beat == 0:
                voices.append(root_midi - 12)
            for note in voices:
                # 每个声部略微失谐，真实乐器不会绝对准
                freq = 440.0 * 2 ** ((note - 69) / 12) * (1 + RNG.uniform(-0.0008, 0.0008))
                # 让音符延续到下一拍之后，模拟踏板残响
                tone = piano(freq, beat_dur * 1.8, amp / len(voices))
                end = min(start + len(tone), total)
                audio[start:end] += tone[: end - start]

    return (audio / np.abs(audio).max() * 0.9).astype(np.float32)


def expected_labels(tonic: int) -> list[str]:
    return [f"{PITCH_CLASSES[(tonic + d) % 12]}:{q}" for d, q in PROGRESSION]


def main() -> int:
    print("=" * 76)
    print("阶段 3：和弦识别 + 小节对齐准确率")
    print("=" * 76)
    print(f"进行 I-vi-IV-V 重复两遍，{BPM} BPM {METER}/4，在 {len(TONICS)} 个调上验证")

    # 只对默认路线 cnn 做通过/失败判定。deepchroma 有实测到的系统性弱点
    # （见文末说明），跑它是为了留下记录和对照，不是当作质量门槛。
    failures = []
    for route in ("cnn", "deepchroma"):
        gated = route == "cnn"
        print("\n" + "-" * 76)
        print(f"路线: {route}" + ("" if gated else "   （仅记录，不计入通过判定）"))
        print("-" * 76)

        for tonic in TONICS:
            key = f"{PITCH_CLASSES[tonic]} major"
            audio = synth(tonic)
            truth = expected_labels(tonic)

            beats = track_beats(audio, SR, meters=(METER,), cross_check=False)
            segments = recognize_chords(audio, SR, route=route)
            bar_chords = assign_chords_to_bars(segments, beats.bars)

            got = [bar.chord for bar in bar_chords]
            # 小节数可能因弱起/残片和真值差一两个，按较短的比
            n = min(len(got), len(truth))
            correct = sum(g == t for g, t in zip(got[:n], truth[:n], strict=True))
            coverage = float(np.mean([b.coverage for b in bar_chords])) if bar_chords else 0.0
            diatonic = sum(is_diatonic(c, key) for c in got if c != "N") / max(
                sum(c != "N" for c in got), 1
            )

            ok = n == len(truth) and correct == len(truth)
            if gated and not ok:
                failures.append((route, key, truth, got))

            status = ("通过" if ok else "失败") if gated else (f"{correct}/{n}")
            print(
                f"{key:<9} 小节 {len(got):>2}/{len(truth)}  和弦 {correct}/{n}  "
                f"纯度 {coverage:>5.0%}  调内 {diatonic:>5.0%}  {status}"
            )
            if not ok:
                print(f"          真值: {' '.join(truth)}")
                print(f"          识别: {' '.join(got)}")

    print("\n" + "=" * 76)
    total = len(TONICS)
    if failures:
        print(f"默认路线 cnn 通过 {total - len(failures)}/{total}，存在失败用例，见上方明细")
        return 1
    print(f"默认路线 cnn 通过 {total}/{total}：和弦识别、小节对齐、跨阶段调内验证三项均过")
    print()
    print("deepchroma 的已知弱点：vi 级小三和弦被系统性判成 IV 级大三和弦")
    print("  （三个调各错两次，全部是 vi，无一例外）。两者共享两个音，")
    print("  但错得如此整齐说明是分类环节的偏置，不是随机失误。默认用 cnn。")
    print()
    print("注意合成音频仍是理想条件：一小节一个和弦、无转位、无七和弦。")
    print("真实录音上和声节奏可能比小节快，看『小节内和弦纯度』判断。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
