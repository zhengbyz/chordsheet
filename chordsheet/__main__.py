"""chordsheet 的统一命令行入口。

    chordsheet song.mp3              完整流水线：调式 + 小节线 + 和弦
    chordsheet chords song.mp3       同上，显式写法
    chordsheet key song.mp3          只做调式识别
    chordsheet beats song.mp3        只做节拍与小节线检测

各子命令的参数与 `python -m chordsheet.<模块>` 完全一致，
直接把剩余参数透传给对应模块的 main()。
"""

from __future__ import annotations

import sys

SUBCOMMANDS = {
    "key": "调式识别（Krumhansl-Schmuckler 模板匹配）",
    "beats": "节拍与小节线检测（madmom RNN + DBN）",
    "chords": "和弦识别，输出「小节 → 和弦」（完整流水线）",
}
DEFAULT = "chords"

USAGE = (
    f"""用法: chordsheet [子命令] <音频> [选项]

子命令（省略时默认 {DEFAULT}）:
"""
    + "\n".join(f"  {name:<8} {desc}" for name, desc in SUBCOMMANDS.items())
    + """

示例:
  chordsheet song.mp3                      完整和弦谱
  chordsheet song.mp3 --bars 32
  chordsheet key song.mp3 --profile both   两套调性模板交叉验证
  chordsheet beats song.mp3 --meter 3 4

各子命令的完整选项: chordsheet <子命令> --help
"""
)


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)

    if not args or args[0] in ("-h", "--help"):
        print(USAGE)
        return 0
    if args[0] in ("-V", "--version"):
        from chordsheet import __version__

        print(f"chordsheet {__version__}")
        return 0

    # 第一个参数是已知子命令就用它，否则当作音频路径走默认子命令。
    # 这样 `chordsheet song.mp3` 和 `chordsheet chords song.mp3` 都能用。
    if args[0] in SUBCOMMANDS:
        name, rest = args[0], args[1:]
    else:
        name, rest = DEFAULT, args

    import importlib

    module = importlib.import_module(f"chordsheet.{name}")
    return int(module.main(rest))


if __name__ == "__main__":
    sys.exit(main())
