"""阶段 1：调式识别（Krumhansl-Schmuckler 模板匹配）。

思路三步：
  1. 音频 → chroma（12 维音级轮廓），沿时间轴取平均得到全曲音级分布
  2. 把 24 个调（12 大 + 12 小）的模板逐一和它算皮尔逊相关
  3. 相关系数最高的就是答案

用法：
    .venv/bin/python -m chordsheet.key 素材.mp3
    .venv/bin/python -m chordsheet.key 素材.mp3 --profile both --top 8
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass

import numpy as np

PITCH_CLASSES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

# --- 调性模板 ---------------------------------------------------------------
#
# Krumhansl & Kessler (1982)，来自探针音心理学实验：
# 给被试听一段确立调性的进行，再放单音问「有多契合」，1-7 打分。
# 数值是 C 大调 / C 小调的结果，其余 23 个调靠循环移位得到。
KRUMHANSL_MAJOR = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
KRUMHANSL_MINOR = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])

# Temperley (2001)，从 Kostka-Payne 和声教材的实际乐谱语料统计而来。
# 不是心理学实验，是真实作曲实践的频次统计，在真实录音上通常比原版准。
TEMPERLEY_MAJOR = np.array([5.0, 2.0, 3.5, 2.0, 4.5, 4.0, 2.0, 4.5, 2.0, 3.5, 1.5, 4.0])
TEMPERLEY_MINOR = np.array([5.0, 2.0, 3.5, 4.5, 2.0, 4.0, 2.0, 4.5, 3.5, 2.0, 1.5, 4.0])

PROFILES = {
    "krumhansl": (KRUMHANSL_MAJOR, KRUMHANSL_MINOR),
    "temperley": (TEMPERLEY_MAJOR, TEMPERLEY_MINOR),
}

# 调号：按大调主音的音级序号索引，值为 (记号数, 大调名, 关系小调名)。
# 用常规记谱法拼写（降号调写成降号），而不是硬套 PITCH_CLASSES 的全升号表示。
#
# 单独列出来是因为实测发现：**调号比大小调可靠得多**。
# 7 首真实钢琴曲上，两套模板的调号 7/7 一致，大小调只有 4/7 一致，
# 且 3 次分歧全部是关系大小调混淆。调号只取决于音级集合（哪 7 个音在响），
# 大小调却要靠权重分布里主音的细微优势——主音常常还没三音响。
KEY_SIGNATURES: list[tuple[int, str, str]] = [
    (0, "C major", "A minor"),  # C
    (-5, "Db major", "Bb minor"),  # C#
    (2, "D major", "B minor"),  # D
    (-3, "Eb major", "C minor"),  # D#
    (4, "E major", "C# minor"),  # E
    (-1, "F major", "D minor"),  # F
    (6, "F# major", "D# minor"),  # F#
    (1, "G major", "E minor"),  # G
    (-4, "Ab major", "F minor"),  # G#
    (3, "A major", "F# minor"),  # A
    (-2, "Bb major", "G minor"),  # A#
    (5, "B major", "G# minor"),  # B
]


@dataclass
class KeyResult:
    """一次调式识别的完整结果。"""

    ranking: list[tuple[str, float]]  # (调名, 相关系数)，从高到低
    chroma: np.ndarray  # 全曲平均 chroma，12 维
    profile: str

    @property
    def key(self) -> str:
        return self.ranking[0][0]

    @property
    def score(self) -> float:
        return self.ranking[0][1]

    @property
    def margin(self) -> float:
        """第一名领先第二名多少。

        这是可信度的关键指标：领先 0.30 是笃定，领先 0.02 基本等于抛硬币。
        """
        return self.ranking[0][1] - self.ranking[1][1]

    @property
    def relative_key(self) -> str:
        """关系大小调（共用同一组音，是最主要的混淆来源）。"""
        root, quality = self.key.rsplit(" ", 1)
        i = PITCH_CLASSES.index(root)
        if quality == "major":
            return f"{PITCH_CLASSES[(i + 9) % 12]} minor"
        return f"{PITCH_CLASSES[(i + 3) % 12]} major"

    @property
    def key_signature(self) -> tuple[int, str, str]:
        """调号，返回 (记号数, 大调名, 关系小调名)，正数为升号、负数为降号。

        这是本算法**真正可靠**的输出。它只依赖「哪 7 个音在响」，
        大调和它的关系小调共享同一个调号，所以关系调混淆完全不影响它。
        """
        root, quality = self.key.rsplit(" ", 1)
        i = PITCH_CLASSES.index(root)
        if quality == "minor":
            i = (i + 3) % 12  # 换算成关系大调
        return KEY_SIGNATURES[i]

    def key_signature_label(self) -> str:
        count, major_name, minor_name = self.key_signature
        if count == 0:
            marks = "无升降号"
        else:
            marks = f"{abs(count)}{'♯' if count > 0 else '♭'}"
        return f"{marks}（{major_name} / {minor_name}）"


def mean_chroma(
    y: np.ndarray,
    sr: float,
    *,
    harmonic: bool = True,
    hop_length: int = 512,
) -> np.ndarray:
    """算全曲平均 chroma。

    harmonic=True 时先做 HPSS 只保留谐波成分。鼓是宽带噪声，会在 12 个音级上
    均匀铺能量、把真实分布压平，带鼓的编曲上这一步影响很大。
    """
    import librosa

    if harmonic:
        y = librosa.effects.harmonic(y)

    # chroma_cqt 而非 chroma_stft：CQT 的频率轴按对数分布，天然对齐半音，
    # 不需要把线性频率 bin 硬塞进音级格子。tuning=None 让它自己估调音偏差，
    # 应付跑调的录音或非 A440 的老唱片。
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr, hop_length=hop_length, tuning=None)
    return chroma.mean(axis=1)


def detect_key(chroma: np.ndarray, profile: str = "krumhansl") -> KeyResult:
    """把 12 维 chroma 匹配到 24 个调，返回完整排名。"""
    if profile not in PROFILES:
        raise ValueError(f"未知模板 {profile!r}，可选：{list(PROFILES)}")
    if chroma.shape != (12,):
        raise ValueError(f"chroma 应为 12 维，实际 {chroma.shape}")
    if not np.isfinite(chroma).all() or np.ptp(chroma) == 0:
        raise ValueError("chroma 无效（全零或含 NaN），音频可能是静音")

    major, minor = PROFILES[profile]
    ranking = []
    for i in range(12):
        # 皮尔逊相关而非欧氏距离：我们关心分布的「形状」像不像，
        # 不关心录音音量。相关系数对幅度缩放和整体偏移天然免疫。
        ranking.append(
            (f"{PITCH_CLASSES[i]} major", float(np.corrcoef(chroma, np.roll(major, i))[0, 1]))
        )
        ranking.append(
            (f"{PITCH_CLASSES[i]} minor", float(np.corrcoef(chroma, np.roll(minor, i))[0, 1]))
        )

    ranking.sort(key=lambda item: item[1], reverse=True)
    return KeyResult(ranking=ranking, chroma=chroma, profile=profile)


def analyze_file(
    path: str,
    *,
    profile: str = "krumhansl",
    harmonic: bool = True,
    sr: int = 22050,
    duration: float | None = None,
    offset: float = 0.0,
) -> KeyResult:
    """加载音频文件并识别调式。"""
    import librosa

    y, actual_sr = librosa.load(path, sr=sr, mono=True, duration=duration, offset=offset)
    if len(y) == 0:
        raise ValueError(f"{path} 读出来是空的")
    return detect_key(mean_chroma(y, actual_sr, harmonic=harmonic), profile=profile)


def format_report(result: KeyResult, top: int = 5) -> str:
    """把结果排成人能读的样子。"""
    lines = [f"模板: {result.profile}", ""]

    bar_scale = 40 / max(result.chroma.max(), 1e-9)
    lines.append("音级分布 (全曲平均 chroma):")
    for name, value in zip(PITCH_CLASSES, result.chroma, strict=True):
        lines.append(f"  {name:<2} {'█' * int(value * bar_scale):<40} {value:.3f}")

    lines += ["", f"Top {top}:"]
    for rank, (name, score) in enumerate(result.ranking[:top], start=1):
        marker = " ←" if rank == 1 else ""
        lines.append(f"  {rank}. {name:<10} {score:+.3f}{marker}")

    margin = result.margin
    if margin >= 0.15:
        verdict = "高 — 第一名明显甩开第二名"
    elif margin >= 0.05:
        verdict = "中 — 可信，但值得人耳复核"
    else:
        verdict = "低 — 和第二名几乎并列，别当结论用"

    # 分两级报告：调号可靠，大小调不可靠。混在一起说会高估后者的可信度。
    lines += [
        "",
        f"调号 (可靠): {result.key_signature_label()}",
        f"大小调 (不可靠): {result.key}",
        f"  领先第二名 {margin:+.3f}  可信度: {verdict}",
        f"  另一种可能: {result.relative_key}（关系调，共用同一组音）",
    ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="识别音频的调式")
    parser.add_argument("audio", help="音频文件（wav/flac/mp3/ogg）")
    parser.add_argument(
        "--profile",
        default="krumhansl",
        choices=[*PROFILES, "both"],
        help="调性模板，both 表示两个都跑并对比",
    )
    parser.add_argument("--top", type=int, default=5, help="显示前几名（默认 5）")
    parser.add_argument("--no-harmonic", action="store_true", help="跳过 HPSS 谐波分离")
    parser.add_argument("--duration", type=float, help="只分析前 N 秒")
    parser.add_argument("--offset", type=float, default=0.0, help="从第 N 秒开始")
    args = parser.parse_args(argv)

    profiles = list(PROFILES) if args.profile == "both" else [args.profile]
    results = {}
    for name in profiles:
        result = analyze_file(
            args.audio,
            profile=name,
            harmonic=not args.no_harmonic,
            duration=args.duration,
            offset=args.offset,
        )
        results[name] = result
        print("=" * 56)
        print(format_report(result, top=args.top))

    if len(results) > 1:
        keys = {r.key for r in results.values()}
        signatures = {r.key_signature for r in results.values()}
        print("=" * 56)
        if len(keys) == 1:
            print(f"两个模板一致: {keys.pop()}")
        else:
            for name, result in results.items():
                print(f"  {name:<10} → {result.key:<10} (领先 {result.margin:+.3f})")
            if len(signatures) == 1:
                # 最常见的分歧形态：调号一致，只在大小调上打架
                sample = next(iter(results.values()))
                print(f"\n调号一致: {sample.key_signature_label()} —— 这部分可以采信")
                print("分歧只在大小调上，是 K-S 算法的固有弱点，建议人耳复核")
            else:
                print("\n调号都不一致，这段音频调性本身就很模糊，结果不可用")
    return 0


if __name__ == "__main__":
    sys.exit(main())
