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
    "serve": "启动本地网页界面（上传音频、钢琴键盘、和弦音轨）",
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
  chordsheet serve                         打开图形界面（推荐）
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

    if name == "serve":
        return _serve(rest)

    import importlib

    module = importlib.import_module(f"chordsheet.{name}")
    return int(module.main(rest))


def _serve(argv: list[str]) -> int:
    import argparse

    parser = argparse.ArgumentParser(prog="chordsheet serve", description="启动本地网页界面")
    parser.add_argument(
        "--host",
        default="0.0.0.0",  # noqa: S104
        help="监听地址。默认 0.0.0.0，因为 WSL 下浏览器在 Windows 侧、"
        "绑回环地址访问不到。要限制只允许本机访问传 127.0.0.1",
    )
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--no-browser", action="store_true", help="不自动打开浏览器")
    args = parser.parse_args(argv)

    try:
        from chordsheet.web.app import serve
    except ImportError as exc:
        print(f"缺少网页界面依赖：{exc}", file=sys.stderr)
        print('安装：pip install "chordsheet[web]"', file=sys.stderr)
        return 2
    return serve(host=args.host, port=args.port, open_browser=not args.no_browser)


if __name__ == "__main__":
    sys.exit(main())
