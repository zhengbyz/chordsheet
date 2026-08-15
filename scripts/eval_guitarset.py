"""用 GuitarSet 的人工标注做真实 ground truth 评测。

这是第一次用**外部真值**而非自己合成的音频或「两个方法互相同意」来打分。
后者只能证明一致，不能证明正确。

GuitarSet（CC-BY-4.0）是唯一音频和标注都免费的选择，360 段 30 秒原声吉他，
标注了 key_mode / beat_position / chord，正好覆盖阶段 1/2/3。

**局限必须先说**：全是独奏原声吉他，比「一般音乐」窄得多。
跑出来的数字代表「干净原声吉他上的表现」，不代表流行乐编曲。

三个方法论决定：
  1. 用 mir_eval 而不是自己写指标。加权和弦召回率、节拍容差 F 值都有公认定义，
     自己实现容易在细节上偷偷放水。
  2. comp（伴奏扫弦）和 solo（单音即兴）分开。让和弦识别器从单音旋律里认和弦
     不公平，结论以 comp 为准，solo 只作对照。
  3. 两套和弦标注都评。「指示演奏」是让弹的简单三和弦（majmin 覆盖 95.5%），
     「实际演奏」是真弹出来的带扩展音的（覆盖 66.7%，(1,5) 强力和弦没有三音
     无所谓大小调，被 mir_eval 正确排除）。

用法：
    .venv/bin/python scripts/eval_guitarset.py --limit 12      # 先小样本试
    .venv/bin/python scripts/eval_guitarset.py                 # 全部 comp
    .venv/bin/python scripts/eval_guitarset.py --part both
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import mir_eval  # noqa: E402

from transcribe.beats import MADMOM_SAMPLE_RATE, track_beats  # noqa: E402
from transcribe.chords import (  # noqa: E402
    assign_chords_to_bars,
    recognize_chords,
    rephase_by_chord_changes,
)
from transcribe.key import PROFILES, detect_key, mean_chroma  # noqa: E402

DATA_ROOT = Path.home() / "datasets" / "guitarset"


def stratified_sample(names: list[str], n: int) -> list[str]:
    """按（演奏者 × 风格）分层抽样。

    直接按字典序取前 N 个是错的：文件名以演奏者编号开头，取前 60 段
    只会覆盖演奏者 00 和 01（每人 30 段），6 位里漏掉 4 位。
    数据集是 6 演奏者 × 5 风格完全均衡的，分成 30 格每格均匀取，
    顺带把速度（68-200 BPM）和调（13 个）也铺开。
    """
    cells: dict[tuple[str, str], list[str]] = {}
    for name in names:
        player, _, rest = name.partition("_")
        style = re.match(r"[A-Za-z]+", rest)
        cells.setdefault((player, style.group() if style else "?"), []).append(name)

    def tempo_of(name: str) -> int:
        match = re.search(r"-(\d+)-", name)
        return int(match.group(1)) if match else 0

    keys = sorted(cells)
    base, extra = divmod(n, len(keys))
    picked: list[str] = []
    for i, key in enumerate(keys):
        # 格内按速度排序而非文件名，linspace 取两个就自然拿到最慢和最快。
        # 按文件名取会把速度挤在中段（实测 98-154，而数据集是 68-200），
        # 而倍速/半速错误恰恰在速度极端处最容易发生。
        items = sorted(cells[key], key=tempo_of)
        take = min(base + (1 if i < extra else 0), len(items))
        if take:
            idx = np.linspace(0, len(items) - 1, take).round().astype(int)
            picked += [items[j] for j in dict.fromkeys(idx.tolist())]
    return sorted(picked)


def load_annotation(path: Path) -> dict:
    """从 JAMS（其实就是 JSON）里取出我们要的四项标注。"""
    doc = json.loads(path.read_text())
    out: dict = {"duration": doc["file_metadata"]["duration"]}

    chord_anns = [a for a in doc["annotations"] if a["namespace"] == "chord"]
    for name, ann in (("instructed", chord_anns[0]), ("performed", chord_anns[-1])):
        intervals = np.array([[o["time"], o["time"] + o["duration"]] for o in ann["data"]])
        out[f"chord_{name}"] = (intervals, [o["value"] for o in ann["data"]])

    beat_ann = next(a for a in doc["annotations"] if a["namespace"] == "beat_position")
    out["beats"] = np.array([o["time"] for o in beat_ann["data"]])
    out["downbeats"] = np.array(
        [o["time"] for o in beat_ann["data"] if o["value"]["position"] == 1]
    )
    out["meter"] = int(beat_ann["data"][0]["value"]["num_beats"])

    key_ann = next(a for a in doc["annotations"] if a["namespace"] == "key_mode")
    # 标注写成 'Eb:major'，mir_eval 要 'Eb major'
    out["key"] = key_ann["data"][0]["value"].replace(":", " ")

    tempo_ann = next(a for a in doc["annotations"] if a["namespace"] == "tempo")
    out["tempo"] = float(tempo_ann["data"][0]["value"])
    return out


def evaluate_track(audio_path: Path, ann: dict) -> dict:
    """跑一遍完整流水线并对着标注打分。"""
    import librosa

    y, sr = librosa.load(str(audio_path), sr=MADMOM_SAMPLE_RATE, mono=True)
    row: dict = {"name": audio_path.stem}

    # --- 阶段 1: 调式 ---
    chroma = mean_chroma(y, sr, harmonic=False)
    for profile in PROFILES:
        est = detect_key(chroma, profile=profile)
        row[f"key_{profile}"] = float(mir_eval.key.weighted_score(ann["key"], est.key))
        row[f"key_{profile}_est"] = est.key
    row["key_ref"] = ann["key"]

    # --- 阶段 2: 节拍与小节线 ---
    beats = track_beats(y, sr, cross_check=False)
    # 同时记相位敏感和相位宽容两类指标。只报 F 值会把「跟得极准但整体踩在反拍」
    # 和「根本没跟上」混为一谈——实测 Bossa Nova 是前者（F=0.00 而 AnyLvl=0.97），
    # Funk 是后者（两者都低）。这是完全不同的两个问题。
    scores = mir_eval.beat.evaluate(ann["beats"], beats.times)
    row["beat_f"] = float(scores["F-measure"])
    row["beat_any_level"] = float(scores["Any Metric Level Total"])
    row["beat_cemgil_best"] = float(scores["Cemgil Best Metric Level"])
    row["downbeat_f"] = float(mir_eval.beat.f_measure(ann["downbeats"], beats.downbeats))

    # 每个检出拍点到最近标注拍点的偏移，用拍长归一化。
    # 中位数接近 ±0.5 说明整体踩在反拍，接近 0 说明相位是对的。
    if len(beats.times) and len(ann["beats"]) > 1:
        deltas = beats.times[:, None] - ann["beats"][None, :]
        nearest = deltas[np.arange(len(beats.times)), np.abs(deltas).argmin(axis=1)]
        row["beat_phase_offset"] = float(np.median(nearest) / np.median(np.diff(ann["beats"])))
    else:
        row["beat_phase_offset"] = float("nan")
    row["meter_ok"] = bool(beats.meter == ann["meter"])
    row["tempo_est"] = beats.tempo
    row["tempo_ref"] = ann["tempo"]
    ratio = beats.tempo / ann["tempo"] if ann["tempo"] else 0.0
    row["tempo_ok"] = bool(abs(ratio - 1) < 0.08)
    row["tempo_octave"] = bool(abs(ratio - 2) < 0.15 or abs(ratio - 0.5) < 0.08)

    # --- 阶段 3: 和弦 ---
    segments = recognize_chords(y, sr, route="cnn")
    raw_intervals = np.array([[s, e] for s, e, _ in segments])
    raw_labels = [c for _, _, c in segments]

    # 用和弦变化点重选小节线相位，再测一次。必须和修正前对照报告，
    # 只报修正后的数字看不出这一步到底有没有用、有没有把对的改坏。
    rephased = rephase_by_chord_changes(beats, segments)
    row["downbeat_f_rephased"] = float(
        mir_eval.beat.f_measure(ann["downbeats"], rephased.downbeats)
    )
    row["rephased"] = bool(not np.array_equal(rephased.positions, beats.positions))

    # 量化到小节之后的版本——这才是工具实际输出的东西。
    # 和原始分段对比，就能看出小节对齐这一步本身损失了多少准确率。
    bar_chords = assign_chords_to_bars(segments, beats.bars)
    rephased_bars = assign_chords_to_bars(segments, rephased.bars)

    def as_arrays(chords):
        if not chords:
            return raw_intervals, raw_labels
        return np.array([[b.start, b.end] for b in chords]), [b.chord for b in chords]

    bar_intervals, bar_labels = as_arrays(bar_chords)
    rephased_intervals, rephased_labels = as_arrays(rephased_bars)

    for ref_kind in ("instructed", "performed"):
        ref_intervals, ref_labels = ann[f"chord_{ref_kind}"]
        for est_kind, (est_intervals, est_labels) in (
            ("raw", (raw_intervals, raw_labels)),
            ("bar", (bar_intervals, bar_labels)),
            ("rebar", (rephased_intervals, rephased_labels)),
        ):
            scores = mir_eval.chord.evaluate(ref_intervals, ref_labels, est_intervals, est_labels)
            for metric in ("root", "thirds", "majmin"):
                row[f"chord_{ref_kind}_{est_kind}_{metric}"] = float(scores[metric])
    return row


def summarize(rows: list[dict]) -> str:
    lines = []

    def mean(key: str) -> float:
        values = [r[key] for r in rows if key in r]
        return float(np.mean(values)) if values else float("nan")

    lines.append("=" * 74)
    lines.append(f"GuitarSet 真实标注评测   样本 {len(rows)} 段")
    lines.append("=" * 74)

    lines.append("\n--- 阶段 1: 调式 (mir_eval 加权分：1.0 全对 / 0.5 属调 / 0.3 关系调) ---")
    for profile in PROFILES:
        scores = [r[f"key_{profile}"] for r in rows]
        exact = sum(s == 1.0 for s in scores)
        relative = sum(abs(s - 0.3) < 1e-9 for s in scores)
        fifth = sum(abs(s - 0.5) < 1e-9 for s in scores)
        wrong = sum(s == 0.0 for s in scores)
        lines.append(
            f"  {profile:<10} 加权分 {np.mean(scores):.3f}   "
            f"全对 {exact}/{len(rows)} ({exact / len(rows):.0%})   "
            f"关系调 {relative}   属调 {fifth}   全错 {wrong}"
        )

    lines.append("\n--- 阶段 2: 节拍与小节线 ---")
    lines.append(f"  拍点 F 值（相位敏感）      {mean('beat_f'):.3f}")
    lines.append(f"  Any Metric Level（宽容）   {mean('beat_any_level'):.3f}   允许倍速/半速/反拍")
    lines.append(f"  Cemgil Best Metric Level   {mean('beat_cemgil_best'):.3f}")
    lines.append(f"  小节线 F 值                {mean('downbeat_f'):.3f}")

    # 和弦变化点重定相位的效果。必须报「修好几段 / 弄坏几段」，
    # 只看平均值会掩盖「改动了很多但净收益接近零」这种情况。
    changed = [r for r in rows if r["rephased"]]
    fixed = sum(r["downbeat_f_rephased"] > r["downbeat_f"] + 0.05 for r in rows)
    broken = sum(r["downbeat_f_rephased"] < r["downbeat_f"] - 0.05 for r in rows)
    lines.append(
        f"  小节线 F（和弦重定相位）   {mean('downbeat_f_rephased'):.3f}   "
        f"改动 {len(changed)}/{len(rows)} 段，修好 {fixed}，弄坏 {broken}"
    )

    # 相位偏移接近 ±0.5 拍 = 整体踩在反拍，拍子跟对了但位置错了
    offsets = [abs(r["beat_phase_offset"]) for r in rows if not np.isnan(r["beat_phase_offset"])]
    offbeat = sum(0.35 < o < 0.65 for o in offsets)
    lines.append(f"  相位偏半拍（踩反拍）       {offbeat}/{len(rows)} ({offbeat / len(rows):.0%})")
    lines.append(f"  拍号正确                   {sum(r['meter_ok'] for r in rows)}/{len(rows)}")
    tempo_ok = sum(r["tempo_ok"] for r in rows)
    octave = sum(r["tempo_octave"] for r in rows)
    lines.append(
        f"  速度正确       {tempo_ok}/{len(rows)} ({tempo_ok / len(rows):.0%})"
        f"   其中倍速/半速错误 {octave}"
    )

    lines.append("\n--- 阶段 3: 和弦 (mir_eval 加权和弦召回率) ---")
    lines.append(f"  {'真值':<12} {'输出':<8} {'root':>8} {'thirds':>8} {'majmin':>8}")
    for ref_kind, ref_name in (("instructed", "指示演奏"), ("performed", "实际演奏")):
        for est_kind, est_name in (
            ("raw", "原始分段"),
            ("bar", "量化到小节"),
            ("rebar", "重定相位后"),
        ):
            lines.append(
                f"  {ref_name:<12} {est_name:<8} "
                f"{mean(f'chord_{ref_kind}_{est_kind}_root'):>8.3f} "
                f"{mean(f'chord_{ref_kind}_{est_kind}_thirds'):>8.3f} "
                f"{mean(f'chord_{ref_kind}_{est_kind}_majmin'):>8.3f}"
            )

    delta = mean("chord_instructed_raw_majmin") - mean("chord_instructed_bar_majmin")
    lines.append(f"\n  小节量化的代价: majmin 下降 {delta:+.3f}")

    # 分风格拆解。总平均会掩盖「某一种风格系统性失败」这种最值得知道的情况。
    lines.append("\n--- 分风格 ---")
    lines.append(
        f"  {'风格':<6} {'段数':>4} {'调式':>6} {'拍点F':>7} {'AnyLvl':>7} "
        f"{'小节线F':>8} {'速度对':>7} {'反拍':>5} {'和弦majmin':>11}"
    )
    styles: dict[str, list[dict]] = {}
    for row in rows:
        match = re.search(r"_([A-Za-z]+)", row["name"])
        styles.setdefault(match.group(1) if match else "?", []).append(row)
    for style in sorted(styles):
        group = styles[style]

        def gmean(key: str, g: list[dict] = group) -> float:
            return float(np.mean([r[key] for r in g]))

        offbeat = sum(0.35 < abs(r["beat_phase_offset"]) < 0.65 for r in group)
        lines.append(
            f"  {style:<6} {len(group):>4} {gmean('key_krumhansl'):>6.2f} "
            f"{gmean('beat_f'):>7.3f} {gmean('beat_any_level'):>7.3f} "
            f"{gmean('downbeat_f'):>8.3f} "
            f"{sum(r['tempo_ok'] for r in group):>4}/{len(group):<2} "
            f"{offbeat:>5} {gmean('chord_performed_bar_majmin'):>11.3f}"
        )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="用 GuitarSet 标注评测流水线")
    parser.add_argument("--part", default="comp", choices=["comp", "solo", "both"])
    parser.add_argument(
        "--sample", type=int, help="按（演奏者 × 风格）分层抽 N 段，而非字典序取前 N"
    )
    parser.add_argument("--limit", type=int, help="字典序取前 N 段（仅用于快速冒烟）")
    parser.add_argument("--root", type=Path, default=DATA_ROOT)
    parser.add_argument("--out", type=Path, help="逐段结果写入 JSON")
    args = parser.parse_args(argv)

    ann_dir = args.root / "annotation"
    audio_dir = args.root / "audio_mono-mic"
    if not audio_dir.is_dir():
        print(f"找不到音频目录 {audio_dir}", file=sys.stderr)
        return 2

    names = sorted(p.stem for p in ann_dir.glob("*.jams"))
    if args.part != "both":
        names = [n for n in names if n.endswith(f"_{args.part}")]
    if args.sample:
        names = stratified_sample(names, args.sample)
    elif args.limit:
        names = names[: args.limit]

    players = sorted({n.partition("_")[0] for n in names})
    styles = sorted({m.group() for n in names if (m := re.search(r"_([A-Za-z]+)", n))})
    print(f"评测 {len(names)} 段（{args.part}）")
    print(f"  覆盖演奏者 {len(players)} 位: {' '.join(players)}")
    print(f"  覆盖风格 {len(styles)} 种: {' '.join(s.lstrip('_') for s in styles)}")
    rows, missing = [], []
    started = time.perf_counter()
    for i, name in enumerate(names, start=1):
        audio_path = audio_dir / f"{name}_mic.wav"
        if not audio_path.exists():
            missing.append(name)
            continue
        rows.append(evaluate_track(audio_path, load_annotation(ann_dir / f"{name}.jams")))
        elapsed = time.perf_counter() - started
        eta = elapsed / i * (len(names) - i)
        print(
            f"  [{i}/{len(names)}] {name}  已用 {elapsed / 60:.1f}min  剩余 ~{eta / 60:.1f}min",
            flush=True,
        )

    if missing:
        print(f"\n缺少音频，跳过 {len(missing)} 段: {missing[:5]}")
    if not rows:
        print("没有可评测的样本", file=sys.stderr)
        return 1

    print()
    print(summarize(rows))
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(rows, ensure_ascii=False, indent=2))
        print(f"\n逐段结果已写入 {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
