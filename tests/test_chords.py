"""和弦识别的纯函数测试。

只测「时间段 → 小节」的对齐和调内判断，不跑神经网络。
端到端验证在 scripts/stage3_chords_smoke.py。
"""

import pytest

from chordsheet.chords import (
    NO_CHORD,
    BarChord,
    ChordResult,
    assign_chords_to_bars,
    is_diatonic,
    parse_chord,
)


def bars(n: int, length: float = 2.0) -> list[tuple[int, float, float]]:
    return [(i + 1, i * length, (i + 1) * length) for i in range(n)]


# --- 标签解析 ---------------------------------------------------------------


@pytest.mark.parametrize(
    ("label", "expected"),
    [
        ("C:maj", (0, "maj")),
        ("C#:min", (1, "min")),
        ("B:maj", (11, "maj")),
        ("N", None),
        ("", None),
        ("H:maj", None),  # 不存在的音名
        ("Cb:maj", None),  # madmom 只用升号，降号拼写不认
    ],
)
def test_parse_chord(label: str, expected) -> None:
    assert parse_chord(label) == expected


# --- 小节对齐 ---------------------------------------------------------------


def test_chord_exactly_filling_bar() -> None:
    segments = [(0.0, 2.0, "C:maj"), (2.0, 4.0, "G:maj")]
    result = assign_chords_to_bars(segments, bars(2))

    assert [b.chord for b in result] == ["C:maj", "G:maj"]
    assert all(b.coverage == pytest.approx(1.0) for b in result)


def test_majority_vote_by_overlap_duration() -> None:
    """一小节里两个和弦，占时长的那个赢，占比如实反映。"""
    segments = [(0.0, 1.4, "C:maj"), (1.4, 2.0, "G:maj")]
    result = assign_chords_to_bars(segments, bars(1))

    assert result[0].chord == "C:maj"
    assert result[0].coverage == pytest.approx(0.7)
    assert dict(result[0].shares) == pytest.approx({"C:maj": 0.7, "G:maj": 0.3})


def test_shares_are_sorted_descending() -> None:
    segments = [(0.0, 0.4, "C:maj"), (0.4, 1.2, "G:maj"), (1.2, 2.0, "F:maj")]
    result = assign_chords_to_bars(segments, bars(1))

    values = [share for _, share in result[0].shares]
    assert values == sorted(values, reverse=True)
    assert result[0].chord == result[0].shares[0][0]


def test_segment_spanning_many_bars() -> None:
    """一个长和弦横跨多个小节，每个小节都应拿到它且占比 100%。"""
    result = assign_chords_to_bars([(0.0, 6.0, "C:maj")], bars(3))
    assert [b.chord for b in result] == ["C:maj"] * 3
    assert all(b.coverage == pytest.approx(1.0) for b in result)


def test_bar_with_no_overlapping_segment() -> None:
    """和弦段没覆盖到的小节记为无和弦，而不是继承上一个。"""
    result = assign_chords_to_bars([(0.0, 2.0, "C:maj")], bars(2))
    assert result[1].chord == NO_CHORD
    assert result[1].coverage == 0.0


def test_zero_length_bar_is_safe() -> None:
    result = assign_chords_to_bars([(0.0, 2.0, "C:maj")], [(1, 1.0, 1.0)])
    assert result[0].chord == NO_CHORD
    assert result[0].coverage == 0.0


def test_touching_segments_do_not_double_count() -> None:
    """段首尾相接时边界那一点不能被算两次，占比总和必须是 1。"""
    segments = [(0.0, 1.0, "C:maj"), (1.0, 2.0, "G:maj")]
    result = assign_chords_to_bars(segments, bars(1))
    assert sum(share for _, share in result[0].shares) == pytest.approx(1.0)


def test_same_chord_in_two_segments_is_merged() -> None:
    """同一个和弦被切成两段时占比要累加，不能各算各的。"""
    segments = [(0.0, 0.8, "C:maj"), (0.8, 1.2, "G:maj"), (1.2, 2.0, "C:maj")]
    result = assign_chords_to_bars(segments, bars(1))

    assert result[0].chord == "C:maj"
    assert result[0].coverage == pytest.approx(0.8)
    assert len(result[0].shares) == 2


def test_bar_indices_are_preserved() -> None:
    """小节序号来自阶段 2，不能重新编号——否则和 BeatResult 对不上。"""
    custom = [(5, 0.0, 2.0), (6, 2.0, 4.0)]
    result = assign_chords_to_bars([(0.0, 4.0, "C:maj")], custom)
    assert [b.index for b in result] == [5, 6]


# --- 调内判断 ---------------------------------------------------------------


@pytest.mark.parametrize("label", ["C:maj", "D:min", "E:min", "F:maj", "G:maj", "A:min"])
def test_diatonic_triads_of_c_major(label: str) -> None:
    assert is_diatonic(label, "C major")


@pytest.mark.parametrize(
    "label",
    [
        "C:min",  # 主音变小三，调外
        "D:maj",  # ii 级应为小三
        "A:maj",  # vi 级应为小三
        "A#:maj",  # 降 VII，借用和弦
        "B:maj",  # vii 级应为减三，madmom 输出不了
        NO_CHORD,
    ],
)
def test_non_diatonic_in_c_major(label: str) -> None:
    assert not is_diatonic(label, "C major")


@pytest.mark.parametrize("label", ["A:min", "C:maj", "D:min", "E:min", "E:maj", "F:maj", "G:maj"])
def test_diatonic_triads_of_a_minor(label: str) -> None:
    """小调按和声小调算：V 级（E:maj）和自然小调的 v 级（E:min）都接受。"""
    assert is_diatonic(label, "A minor")


def test_transposition_invariance_of_diatonic_check() -> None:
    """把调和和弦一起移调，调内判断结果必须不变。"""
    from chordsheet.key import PITCH_CLASSES

    for shift in range(12):
        for degree, quality in [(0, "maj"), (2, "min"), (5, "maj"), (7, "maj"), (9, "min")]:
            key = f"{PITCH_CLASSES[shift]} major"
            chord = f"{PITCH_CLASSES[(shift + degree) % 12]}:{quality}"
            assert is_diatonic(chord, key), f"{chord} 应属于 {key}"


def test_unknown_key_is_not_diatonic() -> None:
    assert not is_diatonic("C:maj", "H major")
    assert not is_diatonic("C:maj", "乱写")


# --- ChordResult 汇总 -------------------------------------------------------


def make_result(chords: list[str], coverages: list[float] | None = None, key=None) -> ChordResult:
    coverages = coverages or [1.0] * len(chords)
    bar_chords = [
        BarChord(i + 1, i * 2.0, (i + 1) * 2.0, c, cov, ((c, cov),))
        for i, (c, cov) in enumerate(zip(chords, coverages, strict=True))
    ]
    return ChordResult(segments=[], bar_chords=bar_chords, route="cnn", key=key)


def test_progression_and_mean_coverage() -> None:
    result = make_result(["C:maj", "G:maj"], [1.0, 0.6])
    assert result.progression == ["C:maj", "G:maj"]
    assert result.mean_coverage == pytest.approx(0.8)


def test_ambiguous_bars_uses_threshold() -> None:
    result = make_result(["C:maj", "G:maj", "F:maj"], [1.0, 0.6, 0.75])
    assert result.ambiguous_bars(0.7) == [2]
    assert result.ambiguous_bars(0.8) == [2, 3]


def test_diatonic_ratio_ignores_no_chord() -> None:
    """无和弦的小节不该拉低调内比例——它既不调内也不调外。"""
    result = make_result(["C:maj", "F:maj", NO_CHORD, "C:min"], key="C major")
    assert result.diatonic_ratio == pytest.approx(2 / 3)
    assert result.non_diatonic == [(4, "C:min")]


def test_diatonic_ratio_needs_a_key() -> None:
    assert make_result(["C:maj"]).diatonic_ratio is None
    assert make_result(["C:maj"]).non_diatonic == []


def test_all_no_chord_gives_no_ratio() -> None:
    assert make_result([NO_CHORD, NO_CHORD], key="C major").diatonic_ratio is None


def test_empty_result_degrades_gracefully() -> None:
    result = ChordResult(segments=[], bar_chords=[], route="cnn", key="C major")
    assert result.progression == []
    assert result.mean_coverage == 0.0
    assert result.ambiguous_bars() == []
    assert result.diatonic_ratio is None
