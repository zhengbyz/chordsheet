"""chordsheet —— 从音频生成和弦谱：调式、节拍与小节线、按小节的和弦进行。

三个阶段各自独立可用，也可以串成一条流水线：

    from chordsheet.key import analyze_file as detect_key
    from chordsheet.beats import analyze_file as track_beats
    from chordsheet.chords import analyze_file as transcribe

    result, beats = transcribe("song.mp3")
    for bar in result.bar_chords:
        print(bar.index, bar.chord, bar.coverage)

设计上有一条贯穿始终的原则：**每个输出都带一个可信度指标，且如实报告**。
调式给领先幅度、节拍给速度稳定性、和弦给小节内纯度。真实评测数字见 README。
"""

__version__ = "0.1.0"

__all__ = ["beats", "chords", "key"]
