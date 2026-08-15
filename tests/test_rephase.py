"""用和弦变化点重定小节线相位的纯函数测试。

核心断言分两类：能修的要修对，证据不足的**必须原样返回**。
后者更重要——一个乱改的启发式比不改更糟。
"""

import numpy as np
import pytest

from transcribe.beats import BeatResult
from transcribe.chords import (
    chord_change_times,
    downbeat_phase_votes,
    rephase_by_chord_changes,
)


def make_beats(bpm: float = 120, meter: int = 4, bars: int = 8, shift: int = 0) -> BeatResult:
    """造拍点。shift 表示把「1」错标到第 shift+1 拍上，模拟相位错误。"""
    interval = 60.0 / bpm
    times = np.arange(bars * meter) * interval
    positions = np.roll(np.tile(np.arange(1, meter + 1), bars), shift)
    return BeatResult(
        beats=np.column_stack([times, positions]),
        duration=float(times[-1] + interval),
    )


def chords_on_bars(bpm: float = 120, meter: int = 4, bars: int = 8, labels=None):
    """每小节一个和弦，和弦变化正好落在真正的小节线上。"""
    labels = labels or ["C:maj", "F:maj", "G:maj", "A:min"]
    bar_dur = 60.0 / bpm * meter
    return [(i * bar_dur, (i + 1) * bar_dur, labels[i % len(labels)]) for i in range(bars)]


# --- 变化点提取 -------------------------------------------------------------


def test_chord_change_times_ignores_same_label_seams() -> None:
    """同一个和弦被切成两段，接缝不算变化。"""
    segments = [(0.0, 1.0, "C:maj"), (1.0, 2.0, "C:maj"), (2.0, 3.0, "G:maj")]
    assert chord_change_times(segments) == pytest.approx([2.0])


def test_chord_change_times_on_empty_and_single() -> None:
    assert chord_change_times([]) == []
    assert chord_change_times([(0.0, 4.0, "C:maj")]) == []


# --- 投票 -------------------------------------------------------------------


def test_votes_concentrate_on_true_downbeat() -> None:
    beats = make_beats(shift=0)
    changes = chord_change_times(chords_on_bars())
    votes = downbeat_phase_votes(beats.times, beats.positions, 4, changes, tolerance=0.25)

    assert votes[1] == len(changes)
    assert sum(votes[p] for p in (2, 3, 4)) == 0


@pytest.mark.parametrize("shift", [1, 2, 3])
def test_votes_land_on_wrong_position_when_phase_is_off(shift: int) -> None:
    """相位错了的时候，票会集中到某个非 1 的位置——这正是可修的信号。

    np.roll 右移 shift 位后，真正的小节起点上标的是第 ((-shift) % meter) + 1 拍。
    """
    beats = make_beats(shift=shift)
    changes = chord_change_times(chords_on_bars())
    votes = downbeat_phase_votes(beats.times, beats.positions, 4, changes, tolerance=0.25)

    expected = ((-shift) % 4) + 1
    assert max(votes, key=lambda p: votes[p]) == expected
    assert votes[expected] == len(changes)


def test_votes_ignore_changes_far_from_any_beat() -> None:
    beats = make_beats()
    votes = downbeat_phase_votes(beats.times, beats.positions, 4, [100.0], tolerance=0.25)
    assert sum(votes.values()) == 0


# --- 重定相位 ---------------------------------------------------------------


@pytest.mark.parametrize("shift", [1, 2, 3])
def test_rephase_recovers_correct_downbeats(shift: int) -> None:
    """和弦每小节换一次时，任意相位错误都应被纠正回来。"""
    beats = make_beats(shift=shift)
    segments = chords_on_bars()

    fixed = rephase_by_chord_changes(beats, segments)

    assert fixed.downbeats == pytest.approx(np.arange(8) * 2.0)
    # 拍点时间一个都不许动，只改位置编号
    assert fixed.times == pytest.approx(beats.times)


def test_rephase_leaves_correct_phase_alone() -> None:
    beats = make_beats(shift=0)
    fixed = rephase_by_chord_changes(beats, chords_on_bars())
    assert np.array_equal(fixed.positions, beats.positions)


def test_rephase_preserves_meter_and_metadata() -> None:
    beats = make_beats(shift=2, meter=3)
    fixed = rephase_by_chord_changes(beats, chords_on_bars(meter=3))
    assert fixed.meter == 3
    assert fixed.duration == beats.duration
    assert fixed.candidate_meters == beats.candidate_meters


# --- 证据不足时必须不动 -----------------------------------------------------


def test_too_few_changes_leaves_phase_alone() -> None:
    """只有一两次和弦变化时不足以判断相位，宁可不动。"""
    beats = make_beats(shift=1)
    segments = [(0.0, 8.0, "C:maj"), (8.0, 16.0, "F:maj")]  # 只有 1 个变化点
    fixed = rephase_by_chord_changes(beats, segments)
    assert np.array_equal(fixed.positions, beats.positions)


def test_scattered_votes_leave_phase_alone() -> None:
    """和弦变化均匀散在各拍上（没有哪一拍占优）时不动。"""
    beats = make_beats(shift=1, bars=8)
    # 每拍换一次和弦，票会平均散到 4 个位置
    labels = ["C:maj", "F:maj", "G:maj", "A:min"]
    segments = [(i * 0.5, (i + 1) * 0.5, labels[i % 4]) for i in range(32)]
    fixed = rephase_by_chord_changes(beats, segments)
    assert np.array_equal(fixed.positions, beats.positions)


def test_no_chords_leaves_phase_alone() -> None:
    beats = make_beats(shift=1)
    assert np.array_equal(rephase_by_chord_changes(beats, []).positions, beats.positions)


def test_empty_beats_are_safe() -> None:
    empty = BeatResult(beats=np.zeros((0, 2)), duration=10.0)
    assert rephase_by_chord_changes(empty, chords_on_bars()) is empty


def test_min_share_threshold_is_respected() -> None:
    """把阈值调到 1.0（要求全票）时，只要有一票偏离就不该动。"""
    beats = make_beats(shift=1)
    segments = chords_on_bars()
    assert not np.array_equal(
        rephase_by_chord_changes(beats, segments, min_share=0.5).positions, beats.positions
    )
    # 混入一个落在别的拍上的变化点，全票要求就达不到了
    noisy = [*segments, (3.5, 4.0, "D:min"), (4.0, 4.5, "E:min")]
    assert np.array_equal(
        rephase_by_chord_changes(beats, sorted(noisy), min_share=1.0).positions, beats.positions
    )
