"""阶段 3：和弦识别，输出「小节 → 和弦」。

madmom 提供两条方法完全不同的路线，可以互相验证：
  cnn         CNNChordFeatureProcessor + CRFChordRecognitionProcessor
              CNN 直接从频谱学特征，跳过 chroma 这个中间表示。更准，更慢。
  deepchroma  DeepChromaProcessor + DeepChromaChordRecognitionProcessor
              先用 DNN 提「洗干净的」chroma（去泛音、去打击噪声），再匹配。更快。

两条路线里 CRF 的角色和阶段 2 的 DBN 完全同构：神经网络逐帧输出有噪声，
图模型强加「和弦有持续性、不会每帧都换」的先验，做时序上的全局最优解码。
这个「网络看局部 + 图模型管全局」的模式在 MIR 里反复出现。

**硬限制：词汇表只有 25 类**——12 大三 + 12 小三 + 无和弦（N）。
七和弦、挂留、减、增全部会被映射到最近的大/小三和弦。民谣和简单钢琴曲影响不大，
爵士是毁灭性的。这是模型本身的限制，换不了。

用法：
    .venv/bin/python -m transcribe.chords 素材.mp3
    .venv/bin/python -m transcribe.chords 素材.mp3 --route deepchroma --min-bpm 120
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass

import numpy as np

from transcribe.key import PITCH_CLASSES

ROUTES = ("cnn", "deepchroma")
NO_CHORD = "N"

# 自然音级三和弦，键为相对主音的半音数，值为和弦性质。
# vii 级在大调是减三和弦、ii 级在小调也是减三和弦，madmom 的词汇表里没有减三和弦，
# 所以这两个位置无论识别成什么都不会被判为调内——这是词汇表限制的直接后果。
MAJOR_DIATONIC = {0: {"maj"}, 2: {"min"}, 4: {"min"}, 5: {"maj"}, 7: {"maj"}, 9: {"min"}}
# 小调按和声小调算：V 级用大三和弦（升导音），同时保留自然小调的 v 级小三和弦。
MINOR_DIATONIC = {0: {"min"}, 3: {"maj"}, 5: {"min"}, 7: {"min", "maj"}, 8: {"maj"}, 10: {"maj"}}


def parse_chord(label: str) -> tuple[int, str] | None:
    """把 'C#:maj' 拆成 (音级序号, 性质)。无和弦或无法解析返回 None。"""
    if label == NO_CHORD or ":" not in label:
        return None
    root, _, quality = label.partition(":")
    if root not in PITCH_CLASSES:
        return None
    return PITCH_CLASSES.index(root), quality


def is_diatonic(label: str, key: str) -> bool:
    """这个和弦是不是给定调的自然音级和弦。

    用来把阶段 1 和阶段 3 串起来互证：和弦应该大部分落在调内。
    调内比例低说明调错了、和弦错了、或曲子转调了，三者必居其一。
    """
    parsed = parse_chord(label)
    if parsed is None:
        return False
    root, quality = parsed
    tonic_name, _, mode = key.rpartition(" ")
    if tonic_name not in PITCH_CLASSES:
        return False
    table = MAJOR_DIATONIC if mode == "major" else MINOR_DIATONIC
    degree = (root - PITCH_CLASSES.index(tonic_name)) % 12
    return quality in table.get(degree, set())


@dataclass(frozen=True)
class BarChord:
    """一个小节配一个和弦。"""

    index: int
    start: float
    end: float
    chord: str
    # 主和弦占该小节的时长比例。低于阈值说明这小节里和弦换了，
    # 硬塞一个进去就是信息损失——这是本阶段的可信度指标。
    coverage: float
    shares: tuple[tuple[str, float], ...]  # 全部和弦及占比，降序

    @property
    def duration(self) -> float:
        return self.end - self.start


def chord_change_times(segments: list[tuple[float, float, str]]) -> list[float]:
    """和弦真正发生变化的时刻。

    只取标签不同的分界，相同标签被切成两段的接缝不算变化。
    """
    changes = []
    for (_, end, label), (next_start, _, next_label) in zip(segments, segments[1:], strict=False):
        if label != next_label:
            changes.append((end + next_start) / 2)
    return changes


def downbeat_phase_votes(
    beat_times: np.ndarray,
    beat_positions: np.ndarray,
    meter: int,
    change_times: list[float],
    tolerance: float,
) -> dict[int, int]:
    """统计每个和弦变化点落在小节内的哪一拍上。

    和声几乎总在强拍换。所以票数最多的那一拍，就应该是「1」。
    这是把阶段 3 的信息反馈给阶段 2——madmom 的 downbeat 模型只看声学特征，
    看不到和弦。
    """
    votes = dict.fromkeys(range(1, meter + 1), 0)
    if len(beat_times) == 0:
        return votes
    for time in change_times:
        index = int(np.argmin(np.abs(beat_times - time)))
        if abs(beat_times[index] - time) <= tolerance:
            votes[int(beat_positions[index])] += 1
    return votes


def rephase_by_chord_changes(beats, segments, *, min_share: float = 0.5, min_votes: int = 3):
    """用和弦变化点重新选择哪一拍是小节线，返回新的 BeatResult。

    只改小节内位置的编号，**不动拍点时间**——拍点本身 F 值 0.788 已经不错，
    问题出在选错了相位。实测 60 段里有 12 段是「拍点 F>0.7 但小节线 F<0.3」，
    正是这种纯相位错误。

    证据不足时原样返回：和弦变化太少（短片段可能只有两三次转换），
    或者票数分散（没有哪一拍明显占优）。宁可不动也不要瞎改。
    """
    from transcribe.beats import BeatResult

    meter = beats.meter
    if meter < 2 or len(beats.beats) == 0:
        return beats

    intervals = beats.intervals
    if len(intervals) == 0:
        return beats
    tolerance = float(np.median(intervals)) * 0.5

    changes = chord_change_times(segments)
    votes = downbeat_phase_votes(beats.times, beats.positions, meter, changes, tolerance)
    total = sum(votes.values())
    if total < min_votes:
        return beats

    best = max(votes, key=lambda p: votes[p])
    if votes[best] / total < min_share or best == 1:
        return beats

    positions = ((beats.positions - best) % meter) + 1
    return BeatResult(
        beats=np.column_stack([beats.times, positions]),
        duration=beats.duration,
        candidate_meters=beats.candidate_meters,
        activations=beats.activations,
        reference_tempo=beats.reference_tempo,
    )


def assign_chords_to_bars(
    segments: list[tuple[float, float, str]],
    bars: list[tuple[int, float, float]],
) -> list[BarChord]:
    """把和弦时间段对齐到小节，按重叠时长加权投票。

    madmom 输出的是 (起点, 终点, 和弦名)，和小节线毫无关系——
    这个对齐就是阶段 3 的主要工作量。
    """
    result = []
    for index, bar_start, bar_end in bars:
        span = bar_end - bar_start
        if span <= 0:
            result.append(BarChord(index, bar_start, bar_end, NO_CHORD, 0.0, ()))
            continue

        totals: dict[str, float] = {}
        for seg_start, seg_end, label in segments:
            overlap = min(bar_end, seg_end) - max(bar_start, seg_start)
            if overlap > 0:
                totals[label] = totals.get(label, 0.0) + overlap

        if not totals:
            result.append(BarChord(index, bar_start, bar_end, NO_CHORD, 0.0, ()))
            continue

        shares = sorted(
            ((label, dur / span) for label, dur in totals.items()), key=lambda kv: -kv[1]
        )
        chord, coverage = shares[0]
        result.append(BarChord(index, bar_start, bar_end, chord, coverage, tuple(shares)))
    return result


@dataclass
class ChordResult:
    """一次和弦识别的完整结果。"""

    segments: list[tuple[float, float, str]]
    bar_chords: list[BarChord]
    route: str
    key: str | None = None

    @property
    def progression(self) -> list[str]:
        return [bar.chord for bar in self.bar_chords]

    @property
    def mean_coverage(self) -> float:
        """各小节主和弦占比的平均值，本阶段的整体可信度指标。"""
        if not self.bar_chords:
            return 0.0
        return float(np.mean([bar.coverage for bar in self.bar_chords]))

    def ambiguous_bars(self, threshold: float = 0.7) -> list[int]:
        """主和弦占比低于阈值的小节——这些小节里和弦换了，一个装不下。"""
        return [bar.index for bar in self.bar_chords if bar.coverage < threshold]

    @property
    def diatonic_ratio(self) -> float | None:
        """调内和弦占比。没有调式信息时返回 None。

        跨阶段验证：把阶段 1 的调式和阶段 3 的和弦对上。
        两个独立方法得出的结论互证，比任何单方面的置信度都可靠。
        """
        if self.key is None:
            return None
        real = [bar for bar in self.bar_chords if bar.chord != NO_CHORD]
        if not real:
            return None
        return sum(is_diatonic(bar.chord, self.key) for bar in real) / len(real)

    @property
    def non_diatonic(self) -> list[tuple[int, str]]:
        """调外和弦 [(小节号, 和弦)]。可能是借用和弦、转调，也可能是误识别。"""
        if self.key is None:
            return []
        return [
            (bar.index, bar.chord)
            for bar in self.bar_chords
            if bar.chord != NO_CHORD and not is_diatonic(bar.chord, self.key)
        ]


def recognize_chords(
    y: np.ndarray, sr: int, *, route: str = "cnn"
) -> list[tuple[float, float, str]]:
    """跑 madmom 的和弦识别，返回 (起点, 终点, 和弦名) 列表。"""
    if route not in ROUTES:
        raise ValueError(f"未知路线 {route!r}，可选：{list(ROUTES)}")

    import librosa
    from madmom.audio import Signal
    from madmom.features.chords import (
        CNNChordFeatureProcessor,
        CRFChordRecognitionProcessor,
        DeepChromaChordRecognitionProcessor,
    )

    from transcribe.beats import MADMOM_SAMPLE_RATE

    if y.ndim != 1:
        raise ValueError(f"需要单声道一维数组，实际 {y.shape}")
    if sr != MADMOM_SAMPLE_RATE:
        y = librosa.resample(y, orig_sr=sr, target_sr=MADMOM_SAMPLE_RATE)
        sr = MADMOM_SAMPLE_RATE

    signal = Signal(y.astype(np.float32), sample_rate=sr)
    if route == "cnn":
        segments = CRFChordRecognitionProcessor()(CNNChordFeatureProcessor()(signal))
    else:
        from madmom.audio.chroma import DeepChromaProcessor

        segments = DeepChromaChordRecognitionProcessor()(DeepChromaProcessor()(signal))

    return [(float(s), float(e), str(label)) for s, e, label in segments]


def analyze_file(
    path: str,
    *,
    route: str = "cnn",
    meters: tuple[int, ...] = (3, 4),
    min_bpm: float = 55.0,
    max_bpm: float = 215.0,
    duration: float | None = None,
    offset: float = 0.0,
    detect_key: bool = True,
    rephase: bool = True,
):
    """跑完整流水线：调式（阶段 1）+ 小节线（阶段 2）+ 和弦（阶段 3）。

    返回 (ChordResult, BeatResult)。这就是第一期目标「和弦谱生成器」的全部输出。

    rephase=True 时用和弦变化点回头修正小节线相位。GuitarSet 60 段实测
    小节线 F 从 0.514 提到 0.621（改动 13 段，修好 10、弄坏 3），
    和弦 majmin 从 0.678 提到 0.704。默认开着。
    """
    import librosa

    from transcribe.beats import MADMOM_SAMPLE_RATE, track_beats
    from transcribe.key import detect_key as detect_key_fn
    from transcribe.key import mean_chroma

    y, sr = librosa.load(path, sr=MADMOM_SAMPLE_RATE, mono=True, duration=duration, offset=offset)
    if len(y) == 0:
        raise ValueError(f"{path} 读出来是空的")

    beat_result = track_beats(y, sr, meters=meters, min_bpm=min_bpm, max_bpm=max_bpm)
    segments = recognize_chords(y, sr, route=route)

    if rephase:
        # 把阶段 3 的信息反馈给阶段 2：和声几乎总在强拍换，
        # 哪一拍上的和弦变化最多，那一拍就是「1」。madmom 的 downbeat 模型
        # 只看声学特征，看不到和弦——这是它拿不到的信息。
        beat_result = rephase_by_chord_changes(beat_result, segments)

    key = None
    if detect_key:
        # 阶段 1 默认开 HPSS，但实测在独奏钢琴上占 91% 时间且几乎无收益。
        # 这里已经为了 madmom 用 44100 加载了，直接复用不再重采样。
        #
        # 用 krumhansl 而非 temperley：阶段 1 曾根据合成音频的领先幅度认为
        # temperley 更稳，但 GuitarSet 真实标注上 60 段的结果正好相反——
        # krumhansl 加权分 0.672 / 全错 10 段，temperley 0.570 / 全错 22 段。
        # 领先幅度大只说明它对自己的答案更笃定，不代表答案更对。
        key = detect_key_fn(mean_chroma(y, sr, harmonic=False), profile="krumhansl").key

    return (
        ChordResult(
            segments=segments,
            bar_chords=assign_chords_to_bars(segments, beat_result.bars),
            route=route,
            key=key,
        ),
        beat_result,
    )


def format_report(result: ChordResult, beats, max_bars: int = 32) -> str:
    """把结果排成人能读的样子。"""
    lines = [
        f"路线: {result.route}   和弦段: {len(result.segments)}   小节: {len(result.bar_chords)}",
        f"拍号: {beats.meter}/4   速度: {beats.tempo:.1f} BPM",
    ]
    if result.key:
        lines.append(f"调式（阶段 1）: {result.key}")

    coverage = result.mean_coverage
    if coverage >= 0.85:
        verdict = "高 — 基本一小节一个和弦，和声节奏和小节线对得上"
    elif coverage >= 0.7:
        verdict = "中 — 部分小节内和弦有变化"
    else:
        verdict = "低 — 和声节奏比小节快，一小节一个和弦装不下"
    lines.append(f"小节内和弦纯度: {coverage:.1%}  {verdict}")

    ratio = result.diatonic_ratio
    if ratio is not None:
        if ratio >= 0.85:
            note = "和弦与调式互证，两个阶段的结论一致"
        elif ratio >= 0.6:
            note = "多数调内，少量调外——可能是借用和弦或局部离调"
        else:
            note = "调内比例偏低：调错了、和弦错了、或曲子转调了，三者必居其一"
        lines.append(f"调内和弦占比: {ratio:.1%}  —— {note}")
        if result.non_diatonic:
            shown = result.non_diatonic[:8]
            tail = "..." if len(result.non_diatonic) > 8 else ""
            lines.append("  调外和弦: " + ", ".join(f"第{i}小节 {c}" for i, c in shown) + tail)

    if beats.anacrusis_beats:
        lines.append(
            f"注意: 开头 {beats.anacrusis_beats} 拍（{beats.uncovered_head:.2f}s）是弱起，不在任何小节内"
        )

    ambiguous = result.ambiguous_bars()
    if ambiguous:
        shown = ambiguous[:12]
        tail = "..." if len(ambiguous) > 12 else ""
        lines.append(f"和弦不纯的小节（占比<70%）: {shown}{tail}")

    lines += ["", "小节 → 和弦:"]
    for bar in result.bar_chords[:max_bars]:
        mark = (
            ""
            if bar.coverage >= 0.7
            else f"  ← 混杂 ({', '.join(f'{c} {s:.0%}' for c, s in bar.shares[:3])})"
        )
        outside = (
            ""
            if not result.key or bar.chord == NO_CHORD or is_diatonic(bar.chord, result.key)
            else "  [调外]"
        )
        lines.append(
            f"  第 {bar.index:>3} 小节  {bar.start:7.2f} - {bar.end:7.2f}s   "
            f"{bar.chord:<8} {bar.coverage:>5.0%}{outside}{mark}"
        )
    if len(result.bar_chords) > max_bars:
        lines.append(f"  ... 还有 {len(result.bar_chords) - max_bars} 小节")

    lines += ["", "和弦进行（每行 4 小节）:"]
    progression = result.progression
    for i in range(0, min(len(progression), max_bars), 4):
        row = progression[i : i + 4]
        lines.append(f"  {i + 1:>3}| " + " | ".join(f"{c:<7}" for c in row) + " |")

    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="识别音频的和弦进行，按小节输出")
    parser.add_argument("audio", help="音频文件（wav/flac/mp3/ogg）")
    parser.add_argument("--route", default="cnn", choices=[*ROUTES, "both"], help="识别路线")
    parser.add_argument("--meter", type=int, nargs="+", default=[3, 4], metavar="N")
    parser.add_argument("--bars", type=int, default=32, help="显示前几小节（默认 32）")
    parser.add_argument("--min-bpm", type=float, default=55.0)
    parser.add_argument("--max-bpm", type=float, default=215.0)
    parser.add_argument("--duration", type=float, help="只分析前 N 秒")
    parser.add_argument("--offset", type=float, default=0.0, help="从第 N 秒开始")
    parser.add_argument("--no-key", action="store_true", help="跳过调式识别")
    parser.add_argument("--no-rephase", action="store_true", help="跳过用和弦变化点修正小节线相位")
    args = parser.parse_args(argv)

    routes = list(ROUTES) if args.route == "both" else [args.route]
    results = {}
    for route in routes:
        result, beats = analyze_file(
            args.audio,
            route=route,
            meters=tuple(args.meter),
            min_bpm=args.min_bpm,
            max_bpm=args.max_bpm,
            duration=args.duration,
            offset=args.offset,
            detect_key=not args.no_key,
            rephase=not args.no_rephase,
        )
        results[route] = result
        print("=" * 68)
        print(format_report(result, beats, max_bars=args.bars))

    if len(results) > 1:
        # 两条路线方法不同，一致的小节可以采信，分歧的小节需要人耳
        a, b = (results[r].progression for r in ROUTES)
        pairs = list(zip(a, b, strict=False))
        same = sum(x == y for x, y in pairs)
        print("=" * 68)
        print(f"两条路线逐小节一致率: {same}/{len(pairs)} ({same / max(len(pairs), 1):.0%})")
        diff = [(i + 1, x, y) for i, (x, y) in enumerate(pairs) if x != y]
        if diff:
            print("分歧小节（cnn vs deepchroma）:")
            for index, x, y in diff[:15]:
                print(f"  第 {index:>3} 小节  {x:<8} vs {y}")
            if len(diff) > 15:
                print(f"  ... 还有 {len(diff) - 15} 处")
    return 0


if __name__ == "__main__":
    sys.exit(main())
