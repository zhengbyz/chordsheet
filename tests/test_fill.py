"""补全无和弦格子、以及 MIDI 逐小节重触发的测试。"""

import numpy as np
import pytest

from chordsheet.chords import NO_CHORD, ChordCell, best_triad, fill_no_chord_cells
from chordsheet.key import PITCH_CLASSES
from chordsheet.midi import cells_to_notes

SR = 22050


# --- 三和弦模板匹配 ---------------------------------------------------------


@pytest.mark.parametrize("root", range(12))
@pytest.mark.parametrize(("quality", "intervals"), [("maj", (0, 4, 7)), ("min", (0, 3, 7))])
def test_clean_triad_is_identified(root: int, quality: str, intervals) -> None:
    """干净的三和弦 chroma 必须被认出来，12 个根音都要对。"""
    chroma = np.zeros(12)
    chroma[[(root + i) % 12 for i in intervals]] = 1.0

    assert best_triad(chroma) == f"{PITCH_CLASSES[root]}:{quality}"


def test_major_and_minor_are_distinguished() -> None:
    """大小三和弦只差一个音，不能混。"""
    c_major = np.zeros(12)
    c_major[[0, 4, 7]] = 1.0
    c_minor = np.zeros(12)
    c_minor[[0, 3, 7]] = 1.0

    assert best_triad(c_major) == "C:maj"
    assert best_triad(c_minor) == "C:min"


def test_extra_energy_on_a_foreign_note_is_penalised() -> None:
    """相关系数会惩罚不该响的音——这正是不用「三个音能量和」的原因。

    C-E-G 都响但 D 也一样响时，纯能量和分不出来，相关系数能。
    """
    noisy = np.zeros(12)
    noisy[[0, 4, 7]] = 1.0
    noisy[2] = 1.0  # D 也满响

    clean = np.zeros(12)
    clean[[0, 4, 7]] = 1.0
    assert best_triad(clean) == "C:maj"
    # 加了 D 之后 C:maj 的相关系数必然下降
    template = np.zeros(12)
    template[[0, 4, 7]] = 1.0
    assert np.corrcoef(noisy, template)[0, 1] < np.corrcoef(clean, template)[0, 1]


def test_invalid_chroma_returns_none() -> None:
    assert best_triad(np.zeros(12)) is None  # 全零无方差
    assert best_triad(np.ones(12)) is None  # 常量无方差
    assert best_triad(np.full(12, np.nan)) is None
    assert best_triad(np.arange(11, dtype=float)) is None


def test_scale_invariance() -> None:
    """音量不该影响判断。"""
    chroma = np.zeros(12)
    chroma[[0, 4, 7]] = 1.0
    assert best_triad(chroma * 0.01) == best_triad(chroma * 100) == "C:maj"


# --- 补全 -------------------------------------------------------------------


def tone(pcs, seconds: float, amp: float = 0.5) -> np.ndarray:
    """合成一个由若干音级构成的和弦音块。"""
    n = int(seconds * SR)
    t = np.arange(n) / SR
    wave = np.zeros(n)
    for pc in pcs:
        freq = 261.63 * 2 ** (pc / 12)
        for harmonic in (1, 2, 3):
            wave += np.sin(2 * np.pi * freq * harmonic * t) / harmonic
    return (amp * wave / max(np.abs(wave).max(), 1e-9)).astype(np.float32)


def test_silent_blank_stays_no_chord() -> None:
    """真正的空拍必须保持 N——模型不确定和这里没声音是两回事。"""
    audio = np.concatenate([tone([0, 4, 7], 1.0), np.zeros(SR, dtype=np.float32)])
    cells = [
        ChordCell(1, 0.0, 1.0, "C:maj"),
        ChordCell(1, 1.0, 2.0, NO_CHORD),
    ]

    out = fill_no_chord_cells(cells, audio, SR)
    assert out[-1].chord == NO_CHORD


def test_blank_with_audio_gets_a_chord() -> None:
    """有声音却标成 N 的格子，要补上最接近的和弦。"""
    audio = np.concatenate([tone([0, 4, 7], 1.0), tone([5, 9, 0], 1.0)])
    cells = [
        ChordCell(1, 0.0, 1.0, "C:maj"),
        ChordCell(1, 1.0, 2.0, NO_CHORD),
    ]

    out = fill_no_chord_cells(cells, audio, SR)
    assert out[-1].chord != NO_CHORD
    assert out[-1].chord == "F:maj"  # F-A-C


def test_filling_does_not_touch_recognised_cells() -> None:
    audio = tone([0, 4, 7], 2.0)
    cells = [ChordCell(1, 0.0, 1.0, "C:maj"), ChordCell(1, 1.0, 2.0, "G:maj")]

    assert [c.chord for c in fill_no_chord_cells(cells, audio, SR)] == ["C:maj", "G:maj"]


def test_adjacent_identical_fills_are_merged() -> None:
    """补出来的相邻同名格子要合并，不然会出现一串重复。"""
    audio = tone([0, 4, 7], 3.0)
    cells = [
        ChordCell(1, 0.0, 1.0, NO_CHORD),
        ChordCell(1, 1.0, 2.0, NO_CHORD),
        ChordCell(1, 2.0, 3.0, NO_CHORD),
    ]

    out = fill_no_chord_cells(cells, audio, SR)
    assert len(out) == 1
    assert out[0].start == pytest.approx(0.0)
    assert out[0].end == pytest.approx(3.0)


def test_merge_still_respects_bar_lines() -> None:
    audio = tone([0, 4, 7], 2.0)
    cells = [ChordCell(1, 0.0, 1.0, NO_CHORD), ChordCell(2, 1.0, 2.0, NO_CHORD)]

    out = fill_no_chord_cells(cells, audio, SR)
    assert [c.bar for c in out] == [1, 2]


def test_empty_inputs_are_safe() -> None:
    assert fill_no_chord_cells([], np.zeros(100, dtype=np.float32), SR) == []
    cells = [ChordCell(1, 0.0, 1.0, "C:maj")]
    assert fill_no_chord_cells(cells, np.zeros(0, dtype=np.float32), SR) == cells


# --- MIDI 逐小节重触发 -------------------------------------------------------


def test_same_chord_retriggers_at_each_bar() -> None:
    """同一和弦连续四小节应触发四次，而不是连成一个长音。"""
    cells = [ChordCell(i + 1, i * 2.0, (i + 1) * 2.0, "C:maj") for i in range(4)]
    notes = cells_to_notes(cells, with_bass=False)

    starts = sorted({n.start for n in notes})
    assert starts == pytest.approx([0.0, 2.0, 4.0, 6.0])
    assert len(notes) == 12  # 4 小节 × 3 个音


def test_sustain_mode_merges_repeats() -> None:
    cells = [ChordCell(i + 1, i * 2.0, (i + 1) * 2.0, "C:maj") for i in range(4)]
    notes = cells_to_notes(cells, with_bass=False, sustain=True)

    assert len(notes) == 3
    assert all(n.start == pytest.approx(0.0) and n.end == pytest.approx(8.0) for n in notes)


def test_no_note_spans_a_cell_boundary() -> None:
    """任何音符都不跨越格子边界——不管那是小节线还是小节内的和弦变化。

    共有音连着不断会把和弦边界在卷帘里糊掉。
    """
    cells = [
        ChordCell(1, 0.0, 2.0, "C:maj"),
        ChordCell(1, 2.0, 3.0, "G:maj"),  # 同小节内换和弦，共有 67
        ChordCell(2, 3.0, 5.0, "C:maj"),  # 跨小节
        ChordCell(3, 5.0, 7.0, "C:maj"),  # 跨小节且同和弦
    ]
    notes = cells_to_notes(cells)
    boundaries = [2.0, 3.0, 5.0]

    for note in notes:
        crossed = [b for b in boundaries if note.start < b < note.end]
        assert not crossed, f"音符 {note.pitch} 跨过了 {crossed}"
