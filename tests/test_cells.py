"""小节内和弦细分与 MIDI 导出的纯函数测试。"""

import numpy as np
import pytest

from chordsheet.beats import BeatResult
from chordsheet.chords import NO_CHORD, ChordCell, split_bars_by_chords
from chordsheet.midi import cells_to_notes, chord_notes, write_midi


def bars(n: int, length: float = 2.0, start_index: int = 1):
    return [(start_index + i, i * length, (i + 1) * length) for i in range(n)]


# --- 切分 -------------------------------------------------------------------


def test_one_chord_per_bar_stays_one_cell() -> None:
    segs = [(0.0, 2.0, "C:maj"), (2.0, 4.0, "G:maj")]
    cells = split_bars_by_chords(segs, bars(2))

    assert [(c.bar, c.chord) for c in cells] == [(1, "C:maj"), (2, "G:maj")]


def test_chord_change_inside_bar_produces_two_cells() -> None:
    """这是本功能的核心：小节内换和弦要切成两段，而不是压成一个标签。"""
    segs = [(0.0, 1.2, "C:maj"), (1.2, 2.0, "G:maj")]
    cells = split_bars_by_chords(segs, bars(1))

    assert len(cells) == 2
    assert [c.chord for c in cells] == ["C:maj", "G:maj"]
    assert cells[0].start == pytest.approx(0.0)
    assert cells[0].end == pytest.approx(1.2)
    assert cells[1].end == pytest.approx(2.0)


def test_long_chord_is_cut_at_bar_lines() -> None:
    """跨小节的长和弦要在小节线处切开，每个小节各得一个 cell。"""
    cells = split_bars_by_chords([(0.0, 6.0, "C:maj")], bars(3))

    assert [c.bar for c in cells] == [1, 2, 3]
    assert all(c.chord == "C:maj" for c in cells)
    assert [c.start for c in cells] == pytest.approx([0.0, 2.0, 4.0])


def test_cells_tile_each_bar_without_gaps() -> None:
    """同一小节内的 cell 必须首尾相接、完整铺满，不留缝也不重叠。"""
    segs = [(0.0, 0.7, "C:maj"), (0.7, 1.5, "F:maj"), (1.5, 2.0, "G:maj")]
    cells = split_bars_by_chords(segs, bars(1))

    assert cells[0].start == pytest.approx(0.0)
    assert cells[-1].end == pytest.approx(2.0)
    for prev, nxt in zip(cells[:-1], cells[1:], strict=True):
        assert prev.end == pytest.approx(nxt.start)


def test_tiny_fragment_is_absorbed_not_dropped() -> None:
    """短于阈值的碎片是 10fps 的边界伪影，要并进邻居而不是丢掉。

    丢掉会在时间轴上留下空洞，界面上就是一条缝。
    """
    segs = [(0.0, 0.05, "N"), (0.05, 2.0, "C:maj")]
    cells = split_bars_by_chords(segs, bars(1))

    assert len(cells) == 1
    assert cells[0].chord == "C:maj"
    assert cells[0].start == pytest.approx(0.0)
    assert cells[0].end == pytest.approx(2.0)


def test_fragment_absorbed_into_longer_neighbour() -> None:
    segs = [(0.0, 0.9, "C:maj"), (0.9, 0.95, "D:min"), (0.95, 2.0, "G:maj")]
    cells = split_bars_by_chords(segs, bars(1), min_duration=0.2)

    assert "D:min" not in [c.chord for c in cells]
    assert [c.chord for c in cells] == ["C:maj", "G:maj"]
    assert cells[0].end == pytest.approx(cells[1].start)


def test_bar_without_any_segment_becomes_no_chord() -> None:
    cells = split_bars_by_chords([(0.0, 2.0, "C:maj")], bars(2))
    assert cells[-1].bar == 2
    assert cells[-1].chord == NO_CHORD


def test_pickup_bar_zero_is_supported() -> None:
    """第 0 小节（弱起）也要能正常切分。"""
    custom = [(0, 0.0, 1.0), (1, 1.0, 3.0)]
    cells = split_bars_by_chords([(0.0, 3.0, "C:maj")], custom)
    assert [c.bar for c in cells] == [0, 1]


def test_zero_length_bar_is_skipped() -> None:
    assert split_bars_by_chords([(0.0, 2.0, "C:maj")], [(1, 1.0, 1.0)]) == []


# --- full_bars（从第 0 秒起）------------------------------------------------


def make_beats(bpm=120, meter=4, bars_n=4, offset=0.0) -> BeatResult:
    interval = 60.0 / bpm
    times = offset + np.arange(bars_n * meter) * interval
    positions = np.tile(np.arange(1, meter + 1), bars_n)
    return BeatResult(
        beats=np.column_stack([times, positions]), duration=float(times[-1] + interval)
    )


def test_full_bars_starts_at_zero_when_there_is_a_pickup() -> None:
    beats = make_beats(offset=1.3)

    assert beats.bars[0][1] == pytest.approx(1.3)  # 原来的语义：从第一条小节线起
    assert beats.full_bars[0] == (0, 0.0, pytest.approx(1.3))  # 新增：补上第 0 小节
    assert len(beats.full_bars) == len(beats.bars) + 1


def test_full_bars_adds_nothing_when_starting_on_a_downbeat() -> None:
    beats = make_beats(offset=0.0)
    assert beats.full_bars[0][0] == 1
    assert len(beats.full_bars) == len(beats.bars)


def test_full_bars_covers_the_whole_audio() -> None:
    beats = make_beats(offset=0.7)
    full = beats.full_bars

    assert full[0][1] == pytest.approx(0.0)
    assert full[-1][2] == pytest.approx(beats.duration)
    for prev, nxt in zip(full[:-1], full[1:], strict=True):
        assert prev[2] == pytest.approx(nxt[1])


def test_full_bars_without_any_downbeat() -> None:
    empty = BeatResult(beats=np.zeros((0, 2)), duration=12.0)
    assert empty.full_bars == [(0, 0.0, 12.0)]


# --- MIDI -------------------------------------------------------------------


def test_chord_notes_major_and_minor() -> None:
    assert chord_notes("C:maj", with_bass=False) == [60, 64, 67]
    assert chord_notes("C:min", with_bass=False) == [60, 63, 67]
    assert chord_notes("A:maj", with_bass=False) == [69, 73, 76]


def test_chord_notes_include_bass_an_octave_below() -> None:
    notes = chord_notes("C:maj", with_bass=True)
    assert notes[0] == 36
    assert notes[1:] == [60, 64, 67]


def test_no_chord_yields_no_notes() -> None:
    assert chord_notes(NO_CHORD) == []
    assert chord_notes("乱写") == []


def test_every_cell_retriggers_the_whole_chord() -> None:
    """每个格子都完整重触发，同一和弦跨三个小节 = 三组音块。"""
    cells = [ChordCell(i + 1, i * 2.0, (i + 1) * 2.0, "C:maj") for i in range(3)]

    assert len(cells_to_notes(cells, with_bass=False)) == 9  # 3 格 × 3 个音
    assert len(cells_to_notes(cells, with_bass=False, sustain=True)) == 3


def test_common_tone_also_retriggers_on_chord_change() -> None:
    """共有音也要断开重弹，不能连成一条长音块。

    A# 大三是 70-74-77、D# 大三是 63-67-70，共有 70。连着不断看起来像
    voice leading，但在卷帘里会把两个和弦的边界糊掉——记谱要的是
    「这里换和弦了」这个信息。
    """
    cells = [ChordCell(1, 0.0, 2.0, "A#:maj"), ChordCell(1, 2.0, 4.0, "D#:maj")]
    notes = cells_to_notes(cells, with_bass=False)

    assert {n.pitch for n in notes if n.start == pytest.approx(0.0)} == {70, 74, 77}
    assert {n.pitch for n in notes if n.start == pytest.approx(2.0)} == {63, 67, 70}
    # 共有的 70 必须是两个独立音符，而不是一个 0→4 秒的长音
    seventies = sorted((n.start, n.end) for n in notes if n.pitch == 70)
    assert seventies == [
        (pytest.approx(0.0), pytest.approx(2.0)),
        (pytest.approx(2.0), pytest.approx(4.0)),
    ]


def test_sustain_mode_still_holds_common_tones() -> None:
    """sustain=True 保留旧行为，供需要 voice leading 的场合。"""
    cells = [ChordCell(1, 0.0, 2.0, "A#:maj"), ChordCell(1, 2.0, 4.0, "D#:maj")]
    notes = cells_to_notes(cells, with_bass=False, sustain=True)

    held = [n for n in notes if n.pitch == 70]
    assert len(held) == 1
    assert held[0].start == pytest.approx(0.0)
    assert held[0].end == pytest.approx(4.0)


def test_relative_keys_do_not_share_absolute_pitches() -> None:
    """原位记法的代价：C 和 Am 虽共有 C、E 两个音级，绝对音高却不重合。

    C=60,64,67 而 A:min=69,72,76。这是「同一和弦名永远画在同一高度」的取舍
    换来的——不做转位去凑最小移动。
    """
    cells = [ChordCell(1, 0.0, 2.0, "C:maj"), ChordCell(2, 2.0, 4.0, "A:min")]
    notes = cells_to_notes(cells, with_bass=False)

    assert {n.pitch for n in notes if n.start == pytest.approx(2.0)} == {69, 72, 76}
    assert all(n.end == pytest.approx(2.0) for n in notes if n.start == pytest.approx(0.0))


def test_no_chord_cells_produce_silence() -> None:
    cells = [ChordCell(1, 0.0, 2.0, NO_CHORD), ChordCell(2, 2.0, 4.0, "C:maj")]
    notes = cells_to_notes(cells, with_bass=False)
    assert all(n.start >= 2.0 - 1e-6 for n in notes)


def test_written_midi_is_readable_and_ordered(tmp_path) -> None:
    """写出的文件要能被读回来，且 delta time 不为负。

    音符是重叠的，按音符顺序直接写会算错 delta——必须先展平成事件再排序。

    这里直接 import mido 而不用 importorskip：缺依赖时应当报错，
    而不是悄悄跳过——那样 CI 是绿的但这条断言根本没跑，比红着更危险。
    """
    import mido

    cells = [
        ChordCell(1, 0.0, 2.0, "C:maj"),
        ChordCell(2, 2.0, 3.0, "G:maj"),
        ChordCell(3, 3.0, 5.0, "A:min"),
    ]
    path = tmp_path / "out.mid"
    write_midi(cells_to_notes(cells), path, tempo_bpm=120)

    loaded = mido.MidiFile(str(path))
    messages = [m for m in loaded.tracks[0] if m.type in ("note_on", "note_off")]
    assert messages
    assert all(m.time >= 0 for m in loaded.tracks[0])
    # 开与关必须配平
    assert sum(m.type == "note_on" for m in messages) == sum(m.type == "note_off" for m in messages)
    assert loaded.length == pytest.approx(5.0, abs=0.05)


def test_tiny_pickup_is_merged_into_bar_one() -> None:
    """第一条小节线检测在 0.02s 这种位置时，不该造出一个 20 毫秒的第 0 小节。

    那会在 MIDI 里变成一个 20ms 的和弦、在卷帘里变成一条看不见的缝。
    并进第 1 小节，既不留缝也不产生垃圾格子。
    """
    beats = make_beats(offset=0.02)
    full = beats.full_bars

    assert full[0][0] == 1
    assert full[0][1] == pytest.approx(0.0)
    assert all(index != 0 for index, _, _ in full)


def test_real_pickup_still_becomes_bar_zero() -> None:
    """够长的弱起仍然单独成第 0 小节。"""
    beats = make_beats(offset=0.6)
    assert beats.full_bars[0][0] == 0
    assert beats.full_bars[0][2] == pytest.approx(0.6)


def test_tiny_tail_is_merged_into_previous_bar() -> None:
    """最后一条小节线贴着曲末时，不该切出一个 10 毫秒的小节。

    和开头的弱起同一条规则：太短的残片并进邻居，既不单独成格也不丢弃。
    """
    beats = make_beats(bars_n=3)
    beats.duration = float(beats.times[-1]) + 0.51  # 末拍后再多 0.01s
    full = beats.full_bars

    assert len(full) == 3
    assert full[-1][2] == pytest.approx(beats.duration)
    assert min(end - start for _, start, end in full) > 0.5


def test_full_bars_still_covers_everything_after_merging() -> None:
    """两头合并之后仍然一秒不漏。"""
    beats = make_beats(offset=0.03, bars_n=3)
    beats.duration = float(beats.times[-1]) + 0.52
    full = beats.full_bars

    assert full[0][1] == pytest.approx(0.0)
    assert full[-1][2] == pytest.approx(beats.duration)
    for prev, nxt in zip(full[:-1], full[1:], strict=True):
        assert prev[2] == pytest.approx(nxt[1])
