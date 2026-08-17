"""音高集合 → 和弦标签，以及编辑后按音高导出 MIDI 的测试。

这是「音块是事实、标签是投影」这个模型的核心：标签必须始终反映实际音高，
说不出名字就老实显示 `?`，绝不硬套一个最接近的名字。
"""

import pytest

from chordsheet.chords import NO_CHORD, UNKNOWN_CHORD, triad_label
from chordsheet.key import PITCH_CLASSES
from chordsheet.midi import chord_notes, pitch_cells_to_notes

# --- 标签派生 ---------------------------------------------------------------


@pytest.mark.parametrize("root", range(12))
@pytest.mark.parametrize(("quality", "intervals"), [("maj", (0, 4, 7)), ("min", (0, 3, 7))])
def test_exact_triad_gets_its_name(root: int, quality: str, intervals) -> None:
    pitches = [60 + (root + i) % 12 for i in intervals]
    assert triad_label(pitches) == f"{PITCH_CLASSES[root]}:{quality}"


@pytest.mark.parametrize("label", ["C:maj", "F#:min", "A#:maj", "E:min"])
def test_chord_notes_round_trips_through_the_label(label: str) -> None:
    """chord_notes 生成的音高（含低八度根音）必须能还原出同一个标签。

    这条把两个方向锁在一起：改和弦 → 音高 → 标签，必须回到原点。
    """
    assert triad_label(chord_notes(label)) == label


def test_octaves_and_duplicates_do_not_matter() -> None:
    """八度和重复音都折叠到同一个音级，不影响判定。"""
    assert triad_label([60, 64, 67]) == "C:maj"
    assert triad_label([36, 60, 64, 67]) == "C:maj"  # 加低八度根音
    assert triad_label([60, 64, 67, 72, 76, 79]) == "C:maj"  # 整体加倍
    assert triad_label([67, 64, 60]) == "C:maj"  # 顺序无关


def test_added_foreign_note_makes_it_unknown() -> None:
    """C-E-G 加一个 G#，标签必须变成 ? 而不是继续显示 C。

    这正是不用最近邻匹配的理由：最近邻会让标签撒谎。
    """
    assert triad_label([60, 64, 67]) == "C:maj"
    assert triad_label([60, 64, 67, 68]) == UNKNOWN_CHORD


def test_missing_note_makes_it_unknown() -> None:
    """缺一个音也不算三和弦。C-G 是强力和弦，没有三音，无从判断大小调。"""
    assert triad_label([60, 67]) == UNKNOWN_CHORD
    assert triad_label([60]) == UNKNOWN_CHORD


@pytest.mark.parametrize(
    "pitches",
    [
        [60, 64, 68],  # 增三和弦
        [60, 63, 66],  # 减三和弦
        [60, 62, 67],  # 挂二
        [60, 65, 67],  # 挂四
        [60, 64, 67, 71],  # 大七
    ],
)
def test_non_triad_chords_are_unknown(pitches) -> None:
    """词汇表只有大三和小三。其余和弦老实显示 ?，不硬塞进最近的三和弦。"""
    assert triad_label(pitches) == UNKNOWN_CHORD


def test_empty_pitches_mean_no_chord() -> None:
    """一个音都没有 = 无和弦，和「有音但认不出」要区分开。"""
    assert triad_label([]) == NO_CHORD
    assert triad_label([]) != UNKNOWN_CHORD


def test_augmented_triad_is_not_mislabelled() -> None:
    """增三和弦的音级间距对称，最容易被误判成某个大三和弦。"""
    assert triad_label([60, 64, 68]) == UNKNOWN_CHORD


# --- 按音高导出 -------------------------------------------------------------


def cell(start, end, pitches, bar=1):
    return {"bar": bar, "start": start, "end": end, "pitches": pitches}


def test_notes_span_the_whole_cell() -> None:
    """音块长度由格子决定，这是界面上「拖和弦块，音块跟着变」的体现。"""
    notes = pitch_cells_to_notes([cell(0.0, 2.0, [60, 64, 67])])

    assert len(notes) == 3
    assert all(n.start == pytest.approx(0.0) and n.end == pytest.approx(2.0) for n in notes)
    assert sorted(n.pitch for n in notes) == [60, 64, 67]


def test_lowest_pitch_gets_bass_velocity() -> None:
    notes = pitch_cells_to_notes([cell(0.0, 1.0, [36, 60, 64, 67])])
    lowest = min(notes, key=lambda n: n.pitch)

    assert lowest.pitch == 36
    assert lowest.velocity > max(n.velocity for n in notes if n.pitch != 36)


def test_arbitrary_pitch_sets_export_fine() -> None:
    """标签是 ? 的音高集合照样能导出——导出以音块为准，不看标签。"""
    notes = pitch_cells_to_notes([cell(0.0, 1.0, [60, 64, 67, 68])])
    assert sorted(n.pitch for n in notes) == [60, 64, 67, 68]


def test_every_cell_retriggers() -> None:
    """相邻格子即使音高相同也各自触发，不连成长音。"""
    notes = pitch_cells_to_notes([cell(0.0, 1.0, [60]), cell(1.0, 2.0, [60], bar=2)])

    assert len(notes) == 2
    assert sorted(n.start for n in notes) == pytest.approx([0.0, 1.0])


def test_duplicate_pitches_are_deduped() -> None:
    notes = pitch_cells_to_notes([cell(0.0, 1.0, [60, 60, 64])])
    assert sorted(n.pitch for n in notes) == [60, 64]


@pytest.mark.parametrize(
    "bad",
    [
        {"bar": 1, "start": 0.0, "end": 0.0, "pitches": [60]},  # 零长度
        {"bar": 1, "start": 2.0, "end": 1.0, "pitches": [60]},  # 倒序
        {"bar": 1, "start": 0.0, "end": 1.0, "pitches": []},  # 无音高
        {"bar": 1, "start": 0.0, "end": 1.0},  # 缺字段
    ],
)
def test_degenerate_cells_are_skipped(bad) -> None:
    assert pitch_cells_to_notes([bad]) == []
