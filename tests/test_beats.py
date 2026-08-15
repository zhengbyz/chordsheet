"""节拍检测的纯函数测试。

只测 BeatResult 的推导属性——用手造的拍点数组，不跑 RNN。
端到端验证在 scripts/stage2_beats_smoke.py，那个要跑神经网络，一次 26 秒。
"""

import numpy as np
import pytest

from chordsheet.beats import BeatResult


def make_beats(bpm: float, meter: int, bars: int, start: float = 0.0) -> np.ndarray:
    """造一段严格等间距的拍点，位置从 1 循环到 meter。"""
    interval = 60.0 / bpm
    times = start + np.arange(bars * meter) * interval
    positions = np.tile(np.arange(1, meter + 1), bars)
    return np.column_stack([times, positions])


def make_result(bpm: float = 120, meter: int = 4, bars: int = 8, **kwargs) -> BeatResult:
    beats = make_beats(bpm, meter, bars)
    duration = kwargs.pop("duration", float(beats[-1, 0] + 60.0 / bpm))
    return BeatResult(beats=beats, duration=duration, **kwargs)


@pytest.mark.parametrize("bpm", [60, 72, 100, 120, 144, 180])
@pytest.mark.parametrize("meter", [3, 4])
def test_tempo_and_meter(bpm: float, meter: int) -> None:
    result = make_result(bpm, meter)
    assert result.tempo == pytest.approx(bpm)
    assert result.meter == meter


@pytest.mark.parametrize("meter", [3, 4])
def test_downbeats_and_bars(meter: int) -> None:
    """N 条小节线必须切出 N 个小节，且首尾相接不留缝。"""
    bars = 8
    result = make_result(120, meter, bars)

    assert len(result.downbeats) == bars
    assert len(result.bars) == bars

    indices = [i for i, _, _ in result.bars]
    assert indices == list(range(1, bars + 1))

    for (_, _, end), (_, next_start, _) in zip(result.bars[:-1], result.bars[1:], strict=True):
        assert end == pytest.approx(next_start)

    assert result.bars[0][1] == pytest.approx(result.downbeats[0])
    assert result.bars[-1][2] == pytest.approx(result.duration)


def test_last_bar_closes_at_audio_end() -> None:
    """最后一小节用音频结尾收口，即使它不完整。"""
    result = make_result(120, 4, 4, duration=99.0)
    assert result.bars[-1][2] == pytest.approx(99.0)


def test_bars_never_extend_backwards() -> None:
    """duration 比最后一个拍点还小时，最后一小节不能出现负长度。"""
    result = make_result(120, 4, 4, duration=0.5)
    for _, start, end in result.bars:
        assert end >= start


def test_perfect_grid_is_perfectly_stable() -> None:
    """严格等间距的拍点，变异系数必须是 0。"""
    assert make_result().tempo_stability == pytest.approx(0.0, abs=1e-12)


def test_jitter_raises_instability() -> None:
    """加抖动后变异系数必须变大——这是 rubato 检测的基础。"""
    beats = make_beats(120, 4, 16)
    steady = BeatResult(beats=beats, duration=float(beats[-1, 0]))

    jittered = beats.copy()
    rng = np.random.default_rng(42)
    jittered[:, 0] += rng.normal(0, 0.04, len(jittered))
    jittered = jittered[np.argsort(jittered[:, 0])]
    shaky = BeatResult(beats=jittered, duration=float(jittered[-1, 0]))

    assert shaky.tempo_stability > steady.tempo_stability
    assert shaky.tempo_stability > 0.05


def test_incomplete_bars_detects_short_bar() -> None:
    """中间少一拍的小节必须被揪出来。"""
    beats = make_beats(120, 4, 6)
    # 删掉第 3 小节的第 3 拍（下标 2*4 + 2 = 10）
    result = BeatResult(beats=np.delete(beats, 10, axis=0), duration=float(beats[-1, 0]))
    assert 3 in result.incomplete_bars


def test_complete_bars_report_nothing() -> None:
    assert make_result(120, 4, 8).incomplete_bars == []


def test_anacrusis_is_reported_not_swallowed() -> None:
    """弱起必须被显式报出来，不能静默丢掉。

    第一条小节线之前的拍点不属于任何编号小节，`bars` 也不覆盖那段音频。
    阶段 3 按小节切和弦时开头会少一段，所以必须让调用方看得见。
    """
    # 切掉前 2 拍，序列从「第 3 拍」开始，第一条小节线在 2 拍之后
    beats = make_beats(120, 4, 6)  # 0.5s 一拍，起点 0.0
    result = BeatResult(beats=beats[2:], duration=float(beats[-1, 0]))

    assert result.anacrusis_beats == 2
    assert result.uncovered_head == pytest.approx(2.0)
    # 编号小节本身都是齐的，弱起不该混进 incomplete_bars
    assert result.incomplete_bars == []
    # bars 从第一条小节线开始，开头那段确实没被覆盖
    assert result.bars[0][1] == pytest.approx(result.uncovered_head)


def test_no_anacrusis_when_starting_on_downbeat() -> None:
    result = make_result(120, 4, 8)
    assert result.anacrusis_beats == 0
    assert result.uncovered_head == pytest.approx(0.0)


@pytest.mark.parametrize(
    ("tempo_bpm", "reference", "expected"),
    [
        (120, 120, "一致"),
        (120, 118, "一致"),
        (120, 60, "madmom 可能倍速（librosa 认为慢一半）"),
        (60, 120, "madmom 可能半速（librosa 认为快一倍）"),
        (180, 120, "相差三比二，可能是附点/三连音的解读差异"),
        (120, 97, "两者不一致，速度不可信"),
    ],
)
def test_tempo_agreement(tempo_bpm: float, reference: float, expected: str) -> None:
    result = make_result(tempo_bpm, reference_tempo=reference)
    assert result.tempo_agreement == expected


def test_no_cross_check_means_no_verdict() -> None:
    assert make_result().tempo_agreement is None
    assert make_result(reference_tempo=0.0).tempo_agreement is None


@pytest.mark.parametrize(
    "beats",
    [
        np.zeros((5,)),  # 一维
        np.zeros((5, 3)),  # 三列
        np.zeros((5, 1)),  # 一列
    ],
)
def test_malformed_beats_rejected(beats: np.ndarray) -> None:
    with pytest.raises(ValueError, match=r"\(N, 2\)"):
        BeatResult(beats=beats, duration=10.0)


def test_empty_beats_degrade_gracefully() -> None:
    """一个拍点都没检出时不能崩，各项返回空/零。"""
    result = BeatResult(beats=np.zeros((0, 2)), duration=10.0)
    assert result.meter == 0
    assert result.tempo == 0.0
    assert result.tempo_stability == 0.0
    assert result.bars == []
    assert len(result.downbeats) == 0
