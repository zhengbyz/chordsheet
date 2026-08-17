"""把识别出的和弦导出成 MIDI。

导出的是**和弦构成音**（根音、三音、五音），不是歌里实际弹的音符。
本项目做的是和弦识别，拿不到逐音符的音高与起止时间——那需要音源分离加
逐音转录，是另一个量级的工作。

音区选择上，根音单独低八度、三音五音在中音区，这样在钢琴卷帘里看起来
接近常规的左手根音、右手和声的写法。
"""

from __future__ import annotations

from dataclasses import dataclass

from chordsheet.chords import NO_CHORD, parse_chord

# 中央 C = 60。三和弦放在 C4 附近，根音再低一个八度当低音声部。
CHORD_OCTAVE = 60
BASS_OFFSET = -24
TICKS_PER_BEAT = 480
DEFAULT_VELOCITY = 72
BASS_VELOCITY = 84


@dataclass(frozen=True)
class MidiNote:
    """一个待写入的音符。"""

    pitch: int
    start: float  # 秒
    end: float  # 秒
    velocity: int


def chord_notes(label: str, *, octave: int = CHORD_OCTAVE, with_bass: bool = True) -> list[int]:
    """和弦标签 → MIDI 音高列表。无和弦返回空。

    **一律用原位**：根音取 `octave + 音级`，所以 C 是 60-64-67，A 小三是 69-72-76。
    不做转位去凑最小音程移动（voice leading）。

    代价是相邻和弦可能跨一个多八度地跳，卷帘里看着不如转位平滑。
    换来的是「同一个和弦名永远画在同一个高度」——读谱时和弦身份一眼可辨，
    而转位后同一个 C 和弦会因上下文出现在不同位置，反而更难认。
    我们输出的是和弦记谱不是演奏编配，可读性优先于平滑度。
    """
    parsed = parse_chord(label)
    if parsed is None:
        return []
    root, quality = parsed
    intervals = (0, 3, 7) if quality == "min" else (0, 4, 7)
    base = octave + root
    pitches = [base + i for i in intervals]
    if with_bass:
        pitches.insert(0, base + BASS_OFFSET)
    return pitches


def cells_to_notes(
    cells,
    *,
    octave: int = CHORD_OCTAVE,
    with_bass: bool = True,
    sustain: bool = False,
) -> list[MidiNote]:
    """和弦格子 → 音符列表。

    默认**每个格子都完整重触发**：没有任何音符跨越和弦变化或小节线。
    一格 = 一次和弦事件，卷帘里每个和弦是一组独立的音块，边界一眼可辨。

    曾经的做法是让相邻和弦共有的绝对音高连着不断（A# 大三是 70-74-77，
    D# 大三是 63-67-70，共有 70），看起来像 voice leading。但在卷帘里
    那条不断的长音块会把两个和弦的边界糊掉，反而更难读——记谱要的是
    「这里换和弦了」这个信息，不是演奏上的连贯。

    sustain=True 恢复旧行为：音高延续时连成长音符，同一和弦跨多格也合并。
    """
    notes: list[MidiNote] = []

    if not sustain:
        for cell in cells:
            pitches = chord_notes(cell.chord, octave=octave, with_bass=with_bass)
            for pitch in pitches:
                velocity = (
                    BASS_VELOCITY if pitch == min(pitches) and with_bass else DEFAULT_VELOCITY
                )
                notes.append(MidiNote(pitch, cell.start, cell.end, velocity))
        return sorted(notes, key=lambda n: (n.start, n.pitch))

    pending: dict[int, MidiNote] = {}
    for cell in cells:
        pitches = chord_notes(cell.chord, octave=octave, with_bass=with_bass)
        active = set(pitches)

        for pitch, note in list(pending.items()):
            if pitch in active and abs(note.end - cell.start) < 1e-6:
                pending[pitch] = MidiNote(pitch, note.start, cell.end, note.velocity)
            else:
                notes.append(note)
                del pending[pitch]

        for pitch in pitches:
            if pitch not in pending:
                velocity = (
                    BASS_VELOCITY if pitch == min(pitches) and with_bass else DEFAULT_VELOCITY
                )
                pending[pitch] = MidiNote(pitch, cell.start, cell.end, velocity)

    notes.extend(pending.values())
    return sorted(notes, key=lambda n: (n.start, n.pitch))


def pitch_cells_to_notes(cells) -> list[MidiNote]:
    """`[{start, end, pitches}]` → 音符。用于编辑后的导出。

    和 `cells_to_notes` 的区别是音高直接给定，不再从和弦名推导——编辑之后
    音高集合可能不构成任何三和弦，那时标签只是个 `?`，推导不出东西来。

    每格完整重触发，音符横跨整格：这是界面上「音块长度由和弦块决定」的直接体现。
    最低音当低音声部给更大力度，和 chord_notes 的处理一致。
    """
    notes: list[MidiNote] = []
    for cell in cells:
        pitches = sorted({int(p) for p in (cell.get("pitches") or [])})
        start, end = float(cell["start"]), float(cell["end"])
        if not pitches or end <= start:
            continue
        for pitch in pitches:
            velocity = BASS_VELOCITY if pitch == pitches[0] else DEFAULT_VELOCITY
            notes.append(MidiNote(pitch, start, end, velocity))
    return sorted(notes, key=lambda n: (n.start, n.pitch))


def write_midi(
    notes: list[MidiNote],
    path,
    *,
    tempo_bpm: float = 120.0,
    track_name: str = "chordsheet",
) -> None:
    """写出单轨 MIDI 文件。"""
    import mido

    midi = mido.MidiFile(ticks_per_beat=TICKS_PER_BEAT)
    track = mido.MidiTrack()
    midi.tracks.append(track)
    track.append(mido.MetaMessage("track_name", name=track_name, time=0))
    track.append(mido.MetaMessage("set_tempo", tempo=mido.bpm2tempo(max(tempo_bpm, 1.0)), time=0))

    seconds_per_tick = 60.0 / max(tempo_bpm, 1.0) / TICKS_PER_BEAT

    # 展开成 (时刻, 类型, 音高, 力度) 事件后按时间排序，再转成 delta time。
    # 直接按音符顺序写会算错 delta——音符是重叠的。
    events: list[tuple[float, int, int, int]] = []
    for note in notes:
        if note.end <= note.start:
            continue
        events.append((note.start, 1, note.pitch, note.velocity))  # 1 = note_on
        events.append((note.end, 0, note.pitch, 0))  # 0 = note_off，同刻先关后开
    events.sort(key=lambda e: (e[0], e[1]))

    previous = 0.0
    for when, kind, pitch, velocity in events:
        delta = int(round((when - previous) / seconds_per_tick))
        previous = when
        track.append(
            mido.Message(
                "note_on" if kind else "note_off",
                note=int(pitch),
                velocity=int(velocity),
                time=max(delta, 0),
            )
        )

    midi.save(str(path))


def export_chords(cells, path, *, tempo_bpm: float = 120.0, with_bass: bool = True) -> int:
    """一步导出：和弦格子 → MIDI 文件，返回写入的音符数。"""
    notes = cells_to_notes(cells, with_bass=with_bass)
    write_midi(notes, path, tempo_bpm=tempo_bpm)
    return len(notes)


__all__ = [
    "NO_CHORD",
    "MidiNote",
    "cells_to_notes",
    "chord_notes",
    "export_chords",
    "write_midi",
]
