"""chordsheet 的本地网页界面。

madmom 是 Python + 编译扩展，只能在服务端跑，所以形态是「本地服务 + 浏览器界面」：
音频不出本机，不需要联网，也没有上传大小限制。

    chordsheet serve            启动后打开 http://127.0.0.1:8000
"""

__all__ = ["app"]
