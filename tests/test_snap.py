"""和弦边界吸附到拍点网格的测试。

核心断言：每段和弦的起止都必须**正好落在拍点上**，长度是整数个拍。
madmom 原始输出是 10fps 的时间轴产物，边界经常落在拍与拍之间。
"""

import numpy as np
import pytest

from chordsheet.chords import NO_CHORD, snap_chords_to_beats


def grid(bpm=120, meter=4, bars_n=4, offset=0.0):
    """造拍点、小节两套网格。120 BPM 下一拍 0.5s，一小节 2.0s。"""
    interval = 60.0 / bpm
    times = offset + np.arange(bars_n * meter) * interval
    bars = [
        (i + 1, offset + i * interval * meter, offset + (i + 1) * interval * meter)
        for i in range(bars_n)
    ]
    return times, bars, interval


def test_boundaries_land_exactly_on_beats() -> None:
    """和弦边界偏离拍点时必须被拉回拍上。"""
    times, bars, _ = grid()
    # 故意让和弦在 1.37 秒换——落在第 3 拍(1.0s)和第 4 拍(1.5s)之间
    segs = [(0.0, 1.37, "C:maj"), (1.37, 8.0, "G:maj")]

    cells = snap_chords_to_beats(segs, times, bars)
    beat_set = set(np.round(times, 6))

    for cell in cells:
        assert round(cell.start, 6) in beat_set or cell.start == 0.0
        assert round(cell.end, 6) in beat_set or cell.end == pytest.approx(8.0)


@pytest.mark.parametrize("beats_per_chord", [1, 2, 3, 4])
def test_chord_length_is_a_whole_number_of_beats(beats_per_chord: int) -> None:
    """1 拍、2 拍、3 拍、4 拍一换都应被如实还原。"""
    times, bars, interval = grid(bars_n=6)
    labels = ["C:maj", "G:maj", "F:maj", "A:min"]
    segs = []
    for i in range(0, len(times), beats_per_chord):
        segs.append(
            (
                float(times[i]),
                float(times[min(i + beats_per_chord, len(times) - 1)]) + interval,
                labels[(i // beats_per_chord) % 4],
            )
        )

    cells = snap_chords_to_beats(segs, times, bars)

    for cell in cells:
        span = (cell.end - cell.start) / interval
        assert span == pytest.approx(round(span), abs=1e-6), f"{span} 不是整数拍"


def test_slightly_early_change_is_pulled_to_the_beat() -> None:
    """比拍点早一点点的变化，应归到它占多数的那一拍。"""
    times, bars, _ = grid()
    # 1.95s 换和弦，第 4 拍是 [1.5, 2.0)，C 占了 0.45/0.5，所以第 4 拍仍算 C
    segs = [(0.0, 1.95, "C:maj"), (1.95, 8.0, "G:maj")]

    cells = snap_chords_to_beats(segs, times, bars)
    first = [c for c in cells if c.chord == "C:maj"][0]
    assert first.end == pytest.approx(2.0)


def test_adjacent_same_chord_beats_are_merged() -> None:
    """连续同名的拍要合并成一段，而不是一拍一格。"""
    times, bars, _ = grid()
    cells = snap_chords_to_beats([(0.0, 8.0, "C:maj")], times, bars)

    # 每小节一格（跨小节不合并），而不是 16 格
    assert len(cells) == len(bars)
    assert all(c.chord == "C:maj" for c in cells)


def test_merge_stops_at_bar_lines() -> None:
    """同名和弦跨小节时切成两段——cell 带小节号，也符合和弦谱写法。"""
    times, bars, _ = grid(bars_n=2)
    cells = snap_chords_to_beats([(0.0, 4.0, "C:maj")], times, bars)

    assert [c.bar for c in cells] == [1, 2]
    assert cells[0].end == pytest.approx(cells[1].start)


def test_cells_tile_the_timeline_without_gaps() -> None:
    times, bars, _ = grid()
    segs = [(0.0, 2.3, "C:maj"), (2.3, 5.1, "F:maj"), (5.1, 8.0, "G:maj")]
    cells = snap_chords_to_beats(segs, times, bars)

    for prev, nxt in zip(cells[:-1], cells[1:], strict=True):
        assert prev.end == pytest.approx(nxt.start)


def test_region_before_first_beat_becomes_its_own_cell() -> None:
    """第一个拍点之前没有拍可依附，单独成格而不是硬凑进第一拍。"""
    times, bars, _ = grid(offset=0.9)
    cells = snap_chords_to_beats([(0.0, 8.0, "C:maj")], times, bars)

    assert cells[0].start == pytest.approx(0.0)
    assert cells[0].end == pytest.approx(0.9)


def test_tail_after_last_beat_is_covered() -> None:
    """最后一拍之后到曲末的内容不能丢。"""
    times, bars, interval = grid(bars_n=2)
    cells = snap_chords_to_beats([(0.0, 4.0, "C:maj")], times, bars, duration=4.0)

    assert cells[-1].end == pytest.approx(float(times[-1]) + interval)


def test_empty_segments_give_no_chord() -> None:
    times, bars, _ = grid()
    cells = snap_chords_to_beats([], times, bars)
    assert all(c.chord == NO_CHORD for c in cells)


def test_falls_back_when_there_are_no_beats() -> None:
    """没有拍点时退回按小节切，而不是崩掉。"""
    bars = [(1, 0.0, 2.0), (2, 2.0, 4.0)]
    cells = snap_chords_to_beats([(0.0, 4.0, "C:maj")], [], bars)
    assert [c.bar for c in cells] == [1, 2]


def test_no_bars_is_safe() -> None:
    assert snap_chords_to_beats([(0.0, 4.0, "C:maj")], [0.0, 0.5], []) == []


def test_chord_shorter_than_one_beat_is_absorbed() -> None:
    """短于一拍的和弦没有独立的拍可占，会被吸收——这是吸附到拍网格的必然代价。"""
    times, bars, _ = grid()
    segs = [(0.0, 1.9, "C:maj"), (1.9, 2.05, "D:min"), (2.05, 8.0, "G:maj")]

    cells = snap_chords_to_beats(segs, times, bars)
    assert "D:min" not in [c.chord for c in cells]
