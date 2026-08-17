"""前端静态检查：抓「引用了不存在的东西」这类错误。

起因是一次真实事故：用字符串替换往 index.html 里插一个函数，替换没匹配上，
只剩调用点没有定义。浏览器抛 ReferenceError，界面静默失效——音块消失、
按钮没反应，但看不出原因，白查了两轮。

本地没有 node，跑不了正经的 JS linter，所以用启发式覆盖最常犯的三类：

  1. 调用了未定义的顶层函数
  2. $("xxx") 引用了 HTML 里不存在的 id
  3. Edit.foo / Synth.foo 引用了对象里没有的成员

启发式意味着会有误报，所以维护一份白名单而不是放宽规则——
放宽规则等于把检查废掉。

用法：
    .venv/bin/python scripts/check_frontend.py
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

PAGE = Path(__file__).resolve().parent.parent / "chordsheet" / "web" / "static" / "index.html"

# 浏览器内置与语言关键字。启发式抓不到的一律列在这里，而不是放宽正则。
BUILTINS = {
    "if",
    "for",
    "while",
    "switch",
    "catch",
    "function",
    "return",
    "typeof",
    "new",
    "async",
    "await",
    "var",
    "let",
    "const",
    "do",
    "else",
    "Set",
    "Map",
    "Array",
    "Object",
    "JSON",
    "Math",
    "Number",
    "String",
    "Boolean",
    "parseInt",
    "parseFloat",
    "isNaN",
    "setTimeout",
    "clearTimeout",
    "requestAnimationFrame",
    "fetch",
    "alert",
    "confirm",
    "FormData",
    "URLSearchParams",
    "URL",
    "Promise",
    "AudioContext",
    "webkitAudioContext",
    "Error",
    "console",
    "encodeURIComponent",
    "addEventListener",
    "removeEventListener",
    "querySelector",
    "querySelectorAll",
    "getElementById",
    "createElement",
    "appendChild",
    "closest",
    "remove",
    "toFixed",
    "padStart",
    "replace",
    "split",
    "join",
    "map",
    "filter",
    "reduce",
    "forEach",
    "sort",
    "push",
    "pop",
    "shift",
    "splice",
    "indexOf",
    "includes",
    "some",
    "every",
    "find",
    "findIndex",
    "has",
    "add",
    "delete",
    "keys",
    "values",
    "entries",
    "stringify",
    "parse",
    "abs",
    "max",
    "min",
    "round",
    "floor",
    "ceil",
    "pow",
    "sqrt",
    "random",
    "log2",
    "toggle",
    "contains",
    "getBoundingClientRect",
    "setPointerCapture",
    "releasePointerCapture",
    "createGain",
    "createOscillator",
    "createBufferSource",
    "createBuffer",
    "getChannelData",
    "connect",
    "start",
    "stop",
    "resume",
    "setValueAtTime",
    "linearRampToValueAtTime",
    "exponentialRampToValueAtTime",
    "setTargetAtTime",
    "play",
    "pause",
    "blob",
    "json",
    "text",
    "createObjectURL",
    "revokeObjectURL",
    "click",
    "preventDefault",
    "stopPropagation",
    "from",
    "of",
    "isArray",
    "atLeast",
    "sin",
    "exp",
    "cos",
    "dispatchEvent",
    "focus",
    "blur",
    "reverse",
    "concat",
    "slice",
    "flat",
    "trim",
}


def extract_script(html: str) -> str:
    match = re.search(r"<script>(.*?)</script>", html, re.S)
    if not match:
        raise SystemExit("index.html 里找不到 <script> 块")
    return match.group(1)


def declared_names(js: str) -> set[str]:
    """已定义的名字：函数、变量、对象方法简写、形参。"""
    names = set(re.findall(r"\bfunction\s+([A-Za-z_$][\w$]*)", js))
    names |= set(re.findall(r"\b(?:const|let|var)\s+([A-Za-z_$][\w$]*)", js))
    # 对象字面量里的方法简写 `foo(a, b) {`。定义处也长得像调用，
    # 不收进来就会把每个方法都误报成「调用了未定义的函数」。
    names |= set(re.findall(r"^\s+([A-Za-z_$][\w$]*)\s*\([^)]*\)\s*\{", js, re.M))
    # const {a, b} = ... 这类解构
    for group in re.findall(r"\b(?:const|let|var)\s*\{([^}]*)\}", js):
        names |= {n.strip().split(":")[-1].strip() for n in group.split(",") if n.strip()}
    # 形参也算已定义。用 removeprefix 而不是 lstrip("...")——后者是逐字符剥离，
    # 会把 `..a` 也剥成 `a`，不是「去掉展开运算符」的意思。
    for group in re.findall(r"\(([^()]*)\)\s*=>", js):
        names |= {n.strip().removeprefix("...").split("=")[0].strip() for n in group.split(",")}
    names |= set(re.findall(r"\bfor\s*\(\s*(?:const|let|var)\s+([A-Za-z_$][\w$]*)", js))
    return {n for n in names if n}


def check_calls(js: str) -> list[str]:
    """调用了未定义的裸函数名。"""
    known = declared_names(js) | BUILTINS
    problems = []
    for name in sorted(set(re.findall(r"(?<![\w$.])([A-Za-z_$][\w$]*)\s*\(", js))):
        if name not in known:
            problems.append(f"调用了未定义的函数 {name}()")
    return problems


def check_element_ids(html: str, js: str) -> list[str]:
    """$("xxx") 引用的 id 必须在 HTML 里存在，或由 JS 动态创建。"""
    static_ids = set(re.findall(r'\bid="([^"]+)"', html))
    created = set(re.findall(r'\.id\s*=\s*"([^"]+)"', js))
    # 模板字符串里生成的 id（如 id="ruler-head"）也算
    template_ids = set(re.findall(r'id=\\?"([\w-]+)\\?"', js))
    available = static_ids | created | template_ids
    referenced = set(re.findall(r'\$\("([^"]+)"\)', js))
    return [f'$("{i}") 引用了不存在的元素 id' for i in sorted(referenced - available)]


def check_members(js: str) -> list[str]:
    """Edit.foo / Synth.foo 必须是对象里真有的成员。"""
    problems = []
    for obj in ("Edit", "Synth"):
        match = re.search(rf"\bconst {obj} = \{{(.*?)\n\}};", js, re.S)
        if not match:
            problems.append(f"找不到 {obj} 对象定义")
            continue
        body = match.group(1)
        # 属性可能挤在一行（`sel: -1, history: [], original: null,`），
        # 所以不能只匹配行首——要匹配每个 `名字:` 或 `名字(`。
        members = set(re.findall(r"(?:^|[\s,{])([A-Za-z_$][\w$]*)\s*[(:]", body, re.M))
        used = set(re.findall(rf"\b{obj}\.([A-Za-z_$][\w$]*)", js))
        used |= set(re.findall(r"\bthis\.([A-Za-z_$][\w$]*)", body))
        for name in sorted(used - members):
            problems.append(f"{obj}.{name} 不是 {obj} 的成员")
    return problems


def main() -> int:
    html = PAGE.read_text()
    js = extract_script(html)

    problems = check_calls(js) + check_element_ids(html, js) + check_members(js)
    if problems:
        print(f"发现 {len(problems)} 个问题：")
        for p in problems:
            print(f"  - {p}")
        return 1
    print(f"前端检查通过（{len(js.splitlines())} 行 JS）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
