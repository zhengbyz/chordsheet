"""调式识别的纯函数测试。

不碰音频、不跑 CQT，毫秒级。音频端到端的验证在 scripts/stage1_key_smoke.py，
那个慢且依赖 librosa，不适合每次改代码都跑。
"""

import numpy as np
import pytest

from chordsheet.key import KEY_SIGNATURES, PITCH_CLASSES, PROFILES, KeyResult, detect_key


def _result(key: str) -> KeyResult:
    """造一个只有 key 有意义的 KeyResult，用来测纯派生属性。"""
    return KeyResult(ranking=[(key, 1.0), ("dummy", 0.0)], chroma=np.zeros(12), profile="test")


@pytest.mark.parametrize("profile", list(PROFILES))
@pytest.mark.parametrize("tonic", range(12))
@pytest.mark.parametrize("is_minor", [False, True])
def test_profile_identifies_itself(profile: str, tonic: int, is_minor: bool) -> None:
    """自洽性：把模板本身当 chroma 喂回去，必须认出对应的调。

    这是算法的下限。连自己的模板都认不出来，说明相关系数或移位方向写反了。
    """
    major, minor = PROFILES[profile]
    chroma = np.roll(minor if is_minor else major, tonic)
    expected = f"{PITCH_CLASSES[tonic]} {'minor' if is_minor else 'major'}"

    result = detect_key(chroma, profile=profile)

    assert result.key == expected
    # 完美匹配，相关系数应该正好是 1
    assert result.score == pytest.approx(1.0)


@pytest.mark.parametrize("profile", list(PROFILES))
@pytest.mark.parametrize("shift", range(1, 12))
def test_shift_invariance(profile: str, shift: int) -> None:
    """移位不变性：chroma 循环右移 N 位，识别出的主音也该移 N 位，调式不变。

    调性结构是平移不变的，这正是「两条模板 × 12 个旋转 = 24 个调」的前提。
    """
    major, _ = PROFILES[profile]
    base = detect_key(major, profile=profile)
    shifted = detect_key(np.roll(major, shift), profile=profile)

    base_root, base_quality = base.key.rsplit(" ", 1)
    shifted_root, shifted_quality = shifted.key.rsplit(" ", 1)

    assert shifted_quality == base_quality
    expected_root = PITCH_CLASSES[(PITCH_CLASSES.index(base_root) + shift) % 12]
    assert shifted_root == expected_root
    # 移位不改变分布形状，分数应当完全一致
    assert shifted.score == pytest.approx(base.score)


def test_ranking_is_complete_and_sorted() -> None:
    """24 个调全部出现在排名里，且严格降序。"""
    result = detect_key(PROFILES["krumhansl"][0])

    assert len(result.ranking) == 24
    assert len({name for name, _ in result.ranking}) == 24
    scores = [score for _, score in result.ranking]
    assert scores == sorted(scores, reverse=True)
    assert result.margin == pytest.approx(scores[0] - scores[1])


@pytest.mark.parametrize(
    ("key", "expected"),
    [
        ("C major", "A minor"),
        ("A minor", "C major"),
        ("G major", "E minor"),
        ("E minor", "G major"),
        # 跨 12 边界，验证取模没写错
        ("D# major", "C minor"),
        ("C minor", "D# major"),
        ("B major", "G# minor"),
        ("G# minor", "B major"),
    ],
)
def test_relative_key(key: str, expected: str) -> None:
    """关系大小调：大调往下小三度是关系小调，小调往上小三度是关系大调。"""
    assert _result(key).relative_key == expected


def test_relative_key_is_an_involution() -> None:
    """关系调取两次应该回到原点。"""
    for tonic in range(12):
        for quality in ("major", "minor"):
            key = f"{PITCH_CLASSES[tonic]} {quality}"
            once = _result(key).relative_key
            assert _result(once).relative_key == key


@pytest.mark.parametrize(
    ("chroma", "reason"),
    [
        (np.zeros(12), "全零（静音）"),
        (np.ones(12), "常量，无方差，相关系数无定义"),
        (np.full(12, np.nan), "含 NaN"),
        (np.arange(11, dtype=float), "只有 11 维"),
        (np.arange(24, dtype=float).reshape(2, 12), "二维，没沿时间轴聚合"),
    ],
)
def test_invalid_chroma_rejected(chroma: np.ndarray, reason: str) -> None:
    """无效输入要明确报错，不能悄悄返回 NaN 排名。"""
    with pytest.raises(ValueError):
        detect_key(chroma)


def test_unknown_profile_rejected() -> None:
    with pytest.raises(ValueError, match="未知模板"):
        detect_key(PROFILES["krumhansl"][0], profile="不存在的模板")


@pytest.mark.parametrize(
    ("key", "expected_count"),
    [
        ("C major", 0),
        ("A minor", 0),
        ("G major", 1),
        ("E minor", 1),
        ("B major", 5),
        ("G# minor", 5),  # 实测那首钢琴曲的两个候选，调号必须相同
        ("F major", -1),
        ("D minor", -1),
        ("D# major", -3),  # Eb major
        ("C minor", -3),
    ],
)
def test_key_signature_count(key: str, expected_count: int) -> None:
    assert _result(key).key_signature[0] == expected_count


def test_relative_keys_share_key_signature() -> None:
    """关系大小调必须共用同一个调号——这是「调号可靠」这个论断的基础。

    大小调判错时调号仍然对，正因为这两者的调号完全相同。
    """
    for tonic in range(12):
        for quality in ("major", "minor"):
            result = _result(f"{PITCH_CLASSES[tonic]} {quality}")
            relative = _result(result.relative_key)
            assert result.key_signature == relative.key_signature


def test_key_signature_table_is_the_circle_of_fifths() -> None:
    """调号表必须和五度圈一致：主音每上行五度，升号数 +1。"""
    for i in range(12):
        fifth_up = (i + 7) % 12
        delta = KEY_SIGNATURES[fifth_up][0] - KEY_SIGNATURES[i][0]
        # +1 是常规情况，-11 是绕过等音换写边界（6♯ → 5♭）
        assert delta in (1, -11), f"{KEY_SIGNATURES[i][1]} → {KEY_SIGNATURES[fifth_up][1]}"

    counts = sorted(count for count, _, _ in KEY_SIGNATURES)
    assert counts == [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6]


def test_key_signature_label() -> None:
    assert _result("C major").key_signature_label() == "无升降号（C major / A minor）"
    assert _result("B major").key_signature_label() == "5♯（B major / G# minor）"
    assert _result("G# minor").key_signature_label() == "5♯（B major / G# minor）"
    assert _result("D minor").key_signature_label() == "1♭（F major / D minor）"


def test_scale_and_offset_invariance() -> None:
    """相关系数对幅度缩放和整体偏移免疫——这正是不用欧氏距离的原因。

    录音音量大小、chroma 归一化方式都不该影响调性判断。
    """
    chroma = PROFILES["krumhansl"][0]
    base = detect_key(chroma)
    scaled = detect_key(chroma * 7.3)
    offset = detect_key(chroma + 2.5)

    assert scaled.key == base.key == offset.key
    assert scaled.score == pytest.approx(base.score)
    assert offset.score == pytest.approx(base.score)
