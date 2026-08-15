"""阶段 2：节拍与小节线检测（madmom RNN + DBN）。

两段式：
  1. RNNDownBeatProcessor —— 双向 RNN 集成，每 10ms 输出一帧，
     每帧两个概率：这一帧是拍点的概率、是小节线的概率。只看局部，有噪声。
  2. DBNDownBeatTrackingProcessor —— 在「速度 × 小节内位置」的联合状态空间上
     做维特比解码，找全局最优路径。强加三条音乐常识：拍点近似等间距、
     速度平滑变化、小节线每 N 拍出现一次。

单靠 RNN 会给出局部合理但全局矛盾的结果；单靠先验无法处理真实音乐的弹性。

用法：
    .venv/bin/python -m chordsheet.beats 素材.mp3
    .venv/bin/python -m chordsheet.beats 素材.mp3 --meter 3 4 --bars 16
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, field

import numpy as np

# madmom 的模型是 44100 Hz 训练的，写死在 RNNDownBeatProcessor 的前处理链里。
# 喂别的采样率速度会算错，所以这里硬性对齐。
MADMOM_SAMPLE_RATE = 44100

# madmom 不能自动处理任意拍号，必须给候选列表。5/4、7/8 直接没戏，这是硬限制。
DEFAULT_METERS = (3, 4)


@dataclass
class BeatResult:
    """一次节拍检测的结果。

    beats 是 (N, 2) 数组：第一列时间（秒），第二列小节内位置（从 1 开始）。
    位置为 1 的就是小节线。
    """

    beats: np.ndarray
    duration: float
    candidate_meters: tuple[int, ...] = DEFAULT_METERS
    activations: np.ndarray | None = field(default=None, repr=False)
    # librosa 的独立速度估计，用来交叉验证倍速/半速错误。None 表示没做这项检查。
    reference_tempo: float | None = None

    def __post_init__(self) -> None:
        if self.beats.ndim != 2 or self.beats.shape[1] != 2:
            raise ValueError(f"beats 应为 (N, 2) 数组，实际 {self.beats.shape}")

    @property
    def times(self) -> np.ndarray:
        return self.beats[:, 0]

    @property
    def positions(self) -> np.ndarray:
        return self.beats[:, 1].astype(int)

    @property
    def downbeats(self) -> np.ndarray:
        """小节线时间。"""
        return self.times[self.positions == 1]

    @property
    def meter(self) -> int:
        """DBN 实际选中的拍号（每小节几拍）。"""
        return int(self.positions.max()) if len(self.beats) else 0

    @property
    def intervals(self) -> np.ndarray:
        """相邻拍点间隔（秒）。"""
        return np.diff(self.times)

    @property
    def tempo(self) -> float:
        """速度，取间隔的中位数而非均值——中位数不受个别漏检/误检拖动。"""
        if len(self.intervals) == 0:
            return 0.0
        return float(60.0 / np.median(self.intervals))

    @property
    def tempo_stability(self) -> float:
        """间隔的变异系数（标准差 / 均值），越小越稳。

        这是本阶段的可信度指标，作用等同阶段 1 的 margin。
        机械的电子乐接近 0；自由速度的独奏钢琴会明显偏高。
        高不代表错——可能曲子本身就是 rubato——但意味着「小节线」这个
        概念在这段音乐上本来就模糊，下游按小节切和弦会跟着糊。
        """
        if len(self.intervals) < 2:
            return 0.0
        return float(np.std(self.intervals) / np.mean(self.intervals))

    @property
    def tempo_agreement(self) -> str | None:
        """和 librosa 独立估计的对照结论，没做交叉验证时返回 None。

        倍速/半速是节拍跟踪最经典的失败模式：拍点全落在正确位置，
        只是把每两个真拍算成一拍（或反过来）。madmom 自己看不出区别——
        两种解读在它的状态空间里都自洽。只能靠一个方法完全不同的估计来对照。
        librosa 走的是起音包络自相关，和 RNN+DBN 没有共同的失败模式。
        """
        if self.reference_tempo is None or self.reference_tempo <= 0:
            return None
        ratio = self.tempo / self.reference_tempo
        if abs(ratio - 1) < 0.06:
            return "一致"
        if abs(ratio - 2) < 0.15:
            return "madmom 可能倍速（librosa 认为慢一半）"
        if abs(ratio - 0.5) < 0.08:
            return "madmom 可能半速（librosa 认为快一倍）"
        if abs(ratio - 1.5) < 0.1 or abs(ratio - 2 / 3) < 0.07:
            return "相差三比二，可能是附点/三连音的解读差异"
        return "两者不一致，速度不可信"

    # 末尾残片的判定阈值：短于中位小节长度的这个比例就不算小节。
    FRAGMENT_RATIO = 0.25
    # 弱起短于这个秒数就不单独成小节。第一条小节线常检测在 0.02s 这种位置，
    # 硬补一个 20 毫秒的「第 0 小节」会在 MIDI 里造出一个 20ms 的和弦、
    # 在卷帘里造出一条看不见的缝。真正的弱起至少是零点几秒。
    PICKUP_MIN = 0.15

    @property
    def bars(self) -> list[tuple[int, float, float]]:
        """小节列表 [(序号, 起点, 终点)]，序号从 1 开始。

        这是阶段 3「小节 → 和弦」直接要用的东西。

        末尾残片会被丢掉：最后一条小节线若落在音频结束前一丁点，会切出一个
        零点几秒的空壳，下游照样给它配个和弦、算个 100% 纯度，纯属噪声。
        丢掉多少秒可以从 `dropped_tail` 查到，不是静默行为。
        """
        marks = self.downbeats
        if len(marks) == 0:
            return []
        edges = [*marks.tolist(), max(self.duration, float(marks[-1]))]
        candidates = [
            (i + 1, start, end)
            for i, (start, end) in enumerate(zip(edges[:-1], edges[1:], strict=True))
        ]
        if len(candidates) > 1:
            lengths = [end - start for _, start, end in candidates]
            if lengths[-1] < np.median(lengths[:-1]) * self.FRAGMENT_RATIO:
                return candidates[:-1]
        return candidates

    @property
    def full_bars(self) -> list[tuple[int, float, float]]:
        """覆盖整段音频的小节列表，从第 0 秒开始，一秒不漏。

        和 `bars` 的区别只在两头：
          - 开头的弱起补成第 0 小节（`bars` 从第一条小节线才开始）
          - 末尾残片并进上一小节（`bars` 直接丢掉，会漏掉那段音频）

        两头都遵循同一条规则：**太短的残片并进邻居，而不是单独成格或丢弃**。
        单独成格会造出 10 毫秒的小节——MIDI 里是个瞬间和弦，卷帘里是条看不见的缝；
        丢弃则让那段音频凭空消失。合并两头都不占。

        `bars` 在音乐上更正确——小节本来就从强拍起算，弱起不属于任何编号小节，
        评测数字也建立在它上面。但界面上「开头几秒凭空消失」是明显的缺陷，
        所以展示用这个，两者并存而不是改掉 `bars` 的语义。
        """
        marks = self.downbeats
        if len(marks) == 0:
            return [(0, 0.0, self.duration)] if self.duration > 0 else []

        edges = [*marks.tolist(), max(self.duration, float(marks[-1]))]
        result = [
            (i + 1, start, end)
            for i, (start, end) in enumerate(zip(edges[:-1], edges[1:], strict=True))
        ]

        # 末尾残片并进上一小节
        if len(result) > 1:
            lengths = [end - start for _, start, end in result]
            if lengths[-1] < np.median(lengths[:-1]) * self.FRAGMENT_RATIO:
                tail_end = result[-1][2]
                index, start, _ = result[-2]
                result = [*result[:-2], (index, start, tail_end)]

        if marks[0] > self.PICKUP_MIN:
            result.insert(0, (0, 0.0, float(marks[0])))
        elif marks[0] > 1e-6:
            # 弱起太短，并进第 1 小节
            first = result[0]
            result[0] = (first[0], 0.0, first[2])
        return result

    @property
    def dropped_tail(self) -> float:
        """末尾被判为残片、未纳入任何小节的秒数。"""
        kept = self.bars
        if not kept:
            return self.duration
        return max(0.0, self.duration - kept[-1][2])

    @property
    def anacrusis_beats(self) -> int:
        """第一条小节线之前的拍点数，即弱起（不从强拍开始）的长度。

        这段音频不属于任何编号小节——`bars` 从第一条小节线才开始。
        必须显式暴露，否则阶段 3 按小节切和弦时开头会凭空少一段而无人察觉。
        """
        positions = self.positions
        first = np.flatnonzero(positions == 1)
        return int(first[0]) if len(first) else len(positions)

    @property
    def uncovered_head(self) -> float:
        """开头有多少秒不在任何小节内（弱起那一段）。"""
        marks = self.downbeats
        return float(marks[0]) if len(marks) else self.duration

    @property
    def incomplete_bars(self) -> list[int]:
        """拍数不等于拍号的小节序号。

        DBN 保证小节线每 N 拍出现一次，所以正常只有最后一小节可能不齐（结尾被截断）。
        中间出现就是检测出了问题。开头的弱起不在这里报——它不是编号小节，
        见 `anacrusis_beats`。
        """
        meter = self.meter
        counts: dict[int, int] = {}
        bar_index = 0
        for position in self.positions:
            if position == 1:
                bar_index += 1
            if bar_index:
                counts[bar_index] = counts.get(bar_index, 0) + 1
        return [index for index, count in sorted(counts.items()) if count != meter]


def track_beats(
    y: np.ndarray,
    sr: int,
    *,
    meters: tuple[int, ...] = DEFAULT_METERS,
    fps: int = 100,
    min_bpm: float = 55.0,
    max_bpm: float = 215.0,
    cross_check: bool = True,
) -> BeatResult:
    """跑 madmom 的两段式流水线。

    cross_check=True 时额外用 librosa 独立估一次速度，用来发现倍速/半速错误。
    很便宜（相对 RNN 可以忽略），默认开着。
    """
    import librosa
    from madmom.audio import Signal
    from madmom.features.downbeats import (
        DBNDownBeatTrackingProcessor,
        RNNDownBeatProcessor,
    )

    if y.ndim != 1:
        raise ValueError(f"需要单声道一维数组，实际 {y.shape}")
    if sr != MADMOM_SAMPLE_RATE:
        y = librosa.resample(y, orig_sr=sr, target_sr=MADMOM_SAMPLE_RATE)
        sr = MADMOM_SAMPLE_RATE

    duration = len(y) / sr
    signal = Signal(y.astype(np.float32), sample_rate=sr)

    activations = RNNDownBeatProcessor()(signal)
    tracker = DBNDownBeatTrackingProcessor(
        beats_per_bar=list(meters), fps=fps, min_bpm=min_bpm, max_bpm=max_bpm
    )
    beats = tracker(activations)

    reference_tempo = None
    if cross_check:
        # 起音包络自相关，方法上和 RNN+DBN 完全不同，没有共同的失败模式
        reference_tempo = float(np.atleast_1d(librosa.feature.tempo(y=y, sr=sr))[0])

    return BeatResult(
        beats=np.asarray(beats, dtype=float),
        duration=duration,
        candidate_meters=tuple(meters),
        activations=activations,
        reference_tempo=reference_tempo,
    )


def analyze_file(
    path: str,
    *,
    meters: tuple[int, ...] = DEFAULT_METERS,
    duration: float | None = None,
    offset: float = 0.0,
    **kwargs: float,
) -> BeatResult:
    """加载音频文件并检测节拍。

    madmom 自己读 mp3 要靠 ffmpeg（本机没装），所以走 librosa 读成数组再喂进去。
    直接按 44100 加载，省掉一次重采样。
    """
    import librosa

    y, sr = librosa.load(path, sr=MADMOM_SAMPLE_RATE, mono=True, duration=duration, offset=offset)
    if len(y) == 0:
        raise ValueError(f"{path} 读出来是空的")
    return track_beats(y, sr, meters=meters, **kwargs)


def format_report(result: BeatResult, max_bars: int = 12) -> str:
    """把结果排成人能读的样子。"""
    lines = [
        f"时长: {result.duration:.1f}s   拍点: {len(result.beats)}   小节: {len(result.bars)}",
        f"拍号: {result.meter}/4  (候选 {'/'.join(str(m) for m in result.candidate_meters)})",
        f"速度: {result.tempo:.1f} BPM",
    ]

    agreement = result.tempo_agreement
    if agreement is not None:
        lines.append(f"交叉验证: librosa 独立估计 {result.reference_tempo:.1f} BPM —— {agreement}")

    cv = result.tempo_stability
    if cv < 0.03:
        verdict = "很稳 — 接近机械节拍"
    elif cv < 0.08:
        verdict = "正常 — 有自然律动"
    elif cv < 0.15:
        verdict = "偏松 — rubato 或检测有抖动"
    else:
        verdict = "很松 — 小节线在这段音乐上本身就模糊，下游按小节切会跟着糊"
    lines.append(f"速度稳定性: {cv:.3f} (变异系数)  {verdict}")

    if result.anacrusis_beats:
        lines.append(
            f"弱起: 开头 {result.anacrusis_beats} 拍（{result.uncovered_head:.3f}s）"
            f"在第一条小节线之前，不属于任何编号小节"
        )
    if result.dropped_tail > 0.01:
        lines.append(f"末尾残片: {result.dropped_tail:.3f}s 太短，未计入小节")

    bad = result.incomplete_bars
    if bad:
        # 首尾不齐是正常的（弱起或结尾截断），中间不齐才是检测出了问题
        interior = [i for i in bad if i not in (1, len(result.bars))]
        note = (
            "首尾不齐属正常（弱起或结尾截断）"
            if not interior
            else f"中间第 {interior} 小节不齐，可疑"
        )
        lines.append(f"拍数不齐的小节: {bad}  —— {note}")

    lines += ["", f"前 {max_bars} 小节:"]
    for index, start, end in result.bars[:max_bars]:
        lines.append(f"  第 {index:>3} 小节  {start:7.3f} - {end:7.3f}s   ({end - start:.3f}s)")
    if len(result.bars) > max_bars:
        lines.append(f"  ... 还有 {len(result.bars) - max_bars} 小节")

    lines += ["", "拍点分布（| 为小节线，· 为普通拍，每行一小节）:"]
    row: list[str] = []
    for time, position in zip(result.times, result.positions, strict=True):
        if position == 1 and row:
            lines.append("    " + " ".join(row))
            row = []
        row.append(f"|{time:.2f}" if position == 1 else f"·{time:.2f}")
        if len(lines) > max_bars + 40:
            break
    if row:
        lines.append("    " + " ".join(row))

    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="检测音频的节拍与小节线")
    parser.add_argument("audio", help="音频文件（wav/flac/mp3/ogg）")
    parser.add_argument(
        "--meter",
        type=int,
        nargs="+",
        default=list(DEFAULT_METERS),
        metavar="N",
        help="候选拍号，默认 3 4。madmom 不能自动处理任意拍号",
    )
    parser.add_argument("--bars", type=int, default=12, help="显示前几小节（默认 12）")
    parser.add_argument("--min-bpm", type=float, default=55.0)
    parser.add_argument("--max-bpm", type=float, default=215.0)
    parser.add_argument("--duration", type=float, help="只分析前 N 秒")
    parser.add_argument("--offset", type=float, default=0.0, help="从第 N 秒开始")
    parser.add_argument(
        "--no-cross-check", action="store_true", help="跳过 librosa 的独立速度交叉验证"
    )
    args = parser.parse_args(argv)

    result = analyze_file(
        args.audio,
        meters=tuple(args.meter),
        duration=args.duration,
        offset=args.offset,
        min_bpm=args.min_bpm,
        max_bpm=args.max_bpm,
        cross_check=not args.no_cross_check,
    )
    print(format_report(result, max_bars=args.bars))

    if result.tempo_agreement not in (None, "一致"):
        print(
            "\n速度存疑。倍速/半速错误时拍点位置仍然是对的，只是每两个真拍算成了一拍，"
            "\n下游按小节切和弦会切成两倍长。用 --min-bpm/--max-bpm 收窄搜索范围重试。"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
