# chordsheet

从音频生成和弦谱：**调式** + **节拍与小节线** + **按小节的和弦进行**。

*Generate chord sheets from audio: key detection, beat/downbeat tracking, and bar-aligned chord recognition. Built on madmom and librosa. Chinese docs.*

```console
$ chordsheet song.mp3

路线: cnn   和弦段: 8   小节: 8
拍号: 4/4   速度: 91.6 BPM
调式（阶段 1）: C major
小节内和弦纯度: 83.2%  中 — 部分小节内和弦有变化
调内和弦占比: 100.0%  —— 和弦与调式互证，两个阶段的结论一致
和弦不纯的小节（占比<70%）: [1, 5]

小节 → 和弦:
  第   1 小节     0.02 -    2.66s   F:maj      55%  ← 混杂 (F:maj 55%, D:min 45%)
  第   2 小节     2.66 -    5.25s   G:maj      73%
  第   3 小节     5.25 -    7.87s   C:maj     100%
  ...

和弦进行（每行 4 小节）:
    1| F:maj   | G:maj   | C:maj   | C:maj   |
    5| F:maj   | G:maj   | C:maj   | C:maj   |
```

## 这个项目的特点：如实报告可信度

自动音乐转录（AMT）是 MIR 领域的开放问题，**没有哪个工具是可靠的**。
chordsheet 的取向是：与其假装准确，不如**让你知道哪些输出可信、哪些不可信**。

每个阶段都输出一个可信度指标，并且在不可信时明说：

| 阶段 | 可信度指标 | 含义 |
|---|---|---|
| 调式 | 领先幅度（第一名减第二名） | <0.05 基本等于抛硬币 |
| 调式 | **调号 vs 大小调分两级报告** | 调号可靠得多，见下 |
| 节拍 | 速度稳定性（间隔变异系数） | 高 = rubato 或检测抖动 |
| 节拍 | librosa 独立交叉验证 | 抓倍速/半速错误 |
| 和弦 | 小节内和弦纯度 | <70% = 一小节装不下一个和弦 |
| 和弦 | 调内和弦占比 | 跨阶段互证，低了说明有一环错了 |

比如调式，工具**不会**只丢给你一个答案：

```
调号 (可靠): 5♯（B major / G# minor）
大小调 (不可靠): G# minor
  领先第二名 +0.137  可信度: 中 — 可信，但值得人耳复核
  另一种可能: B major（关系调，共用同一组音）
```

调号只取决于「哪 7 个音在响」，而关系大小调共用同一个调号——所以大小调判错完全
不影响调号。把两者混成一个结论输出，等于用 4/7 的可信度污染了 7/7 的结果。

## 真实评测数据

在 **GuitarSet**（人工标注的原声吉他数据集）上，60 段分层抽样，指标全部用
`mir_eval` 计算而非自己实现：

| 阶段 | 指标 | 数值 |
|---|---|---|
| 调式 | mir_eval 加权分 | 0.672（全对 58%） |
| 拍点 | F 值 | 0.788 |
| 小节线 | F 值 | 0.621 |
| 速度 | 正确率 | 67%（14/60 是倍速/半速错误） |
| 和弦 | majmin 加权召回率 | 0.746（原始）/ 0.704（量化到小节） |

**分风格的差异远大于总平均：**

| 风格 | 调式 | 拍点F | 小节线F | 和弦majmin |
|---|---|---|---|---|
| 创作歌手 | **1.00** | 0.762 | 0.666 | **0.895** |
| 摇滚 | 0.68 | **0.885** | **0.318** | 0.727 |
| 波萨诺瓦 | 0.96 | 0.792 | 0.633 | 0.703 |
| 爵士 | **0.32** | 0.748 | 0.501 | 0.618 |
| 放克 | 0.40 | 0.753 | 0.450 | **0.446** |

**结论：结构清晰的民谣/创作歌手作品效果好，爵士和放克基本不可用。**
爵士的扩展和声超出模型的 25 类词汇表，放克常是单一和弦上的静态律动。

摇滚的反差值得单说：拍点 F 全场最高，小节线 F 却全场最低——找到拍子容易，
判断哪一拍是「1」难，节奏型太均匀反而没有线索指示小节起点。

评测怎么做的、哪些改进被数据否决了，见 [docs/EVALUATION.md](docs/EVALUATION.md)。

## 安装

需要 Python ≥ 3.12 和一个 C 编译器（madmom 有 Cython 扩展）。

```bash
git clone https://github.com/zhengbyz/chordsheet.git
cd chordsheet
python3 -m venv .venv
.venv/bin/python -m pip install -U pip
```

**madmom 不能直接 `pip install`**，有三个坑，都不是 Python 3.12 的问题而是生态老化：

```bash
# 1. 构建依赖必须预装，且关掉 pip 的构建隔离
#    madmom 的 setup.py 顶层 import Cython，但它没有 pyproject.toml 声明构建依赖，
#    隔离环境里没有 Cython 会直接失败
# 2. setuptools 必须 <81
#    madmom 用了 pkg_resources，setuptools 81+ 已移除
.venv/bin/python -m pip install "Cython" "setuptools<81" wheel

# 3. 必须装 git main 分支，不能用 PyPI
#    PyPI 上的 0.16.1 停在 2018 年，processors.py 里是
#    `from collections import MutableSequence`，Python 3.10 已删除该别名。
#    main 分支（0.17.dev0）修了这个也跟进了新版 scipy，只是七年没发版。
#    用 git 装还有个好处：pip 会自动拉 submodule，预训练模型就在里面。
.venv/bin/python -m pip install --no-build-isolation \
    "git+https://github.com/CPJKU/madmom.git@main"

# 4. 本项目
.venv/bin/python -m pip install -e .
```

验证：

```bash
.venv/bin/python scripts/stage0_madmom_smoke.py   # madmom 能真跑出结果吗
.venv/bin/python -m pytest                        # 193 个纯函数单测，0.4 秒
```

**音频格式**：WAV / FLAC / MP3 / OGG 靠 libsndfile 直接读，不需要 ffmpeg。
M4A / AAC 不支持，需要另装 ffmpeg 转码。

## 用法

```bash
chordsheet song.mp3                      # 完整流水线
chordsheet song.mp3 --bars 32            # 显示更多小节
chordsheet song.mp3 --route both         # 两条和弦识别路线交叉验证
chordsheet song.mp3 --min-bpm 120        # 速度被判成一半时锁定搜索范围

chordsheet key song.mp3 --profile both   # 只做调式，两套模板对照
chordsheet beats song.mp3 --meter 3 4    # 只做节拍，指定候选拍号
```

作为库使用：

```python
from chordsheet.chords import analyze_file

result, beats = analyze_file("song.mp3")

print(result.key, beats.tempo, beats.meter)
for bar in result.bar_chords:
    print(f"第 {bar.index} 小节: {bar.chord}  纯度 {bar.coverage:.0%}")

# 可信度指标
print(result.mean_coverage)  # 整体小节内和弦纯度
print(result.diatonic_ratio)  # 调内和弦占比（跨阶段互证）
print(result.ambiguous_bars())  # 一个和弦装不下的小节
print(beats.tempo_stability)  # 速度稳定性
print(beats.tempo_agreement)  # librosa 交叉验证结论
```

## 工作原理

```
音频
 ├─ librosa chroma ────────────→ Krumhansl-Schmuckler 模板匹配 → 调式
 ├─ madmom RNN ── DBN 维特比解码 ──────────────────→ 拍点 + 小节线
 └─ madmom CNN ── CRF 时序解码 ────────────────────→ 和弦时间段
                                                          ↓
                              和弦变化点回头修正小节线相位 ←┘
                                                          ↓
                                        按重叠时长投票 → 小节 → 和弦
```

**「神经网络看局部 + 图模型管全局」这个模式出现了两次**：节拍是 RNN + DBN，
和弦是 CNN + CRF。网络逐帧输出有噪声，图模型强加音乐常识（拍点近似等间距、
和弦有持续性）做全局最优解码。这是 MIR 里的标准架构。

**唯一自己实现的算法是调式识别**（K-S 模板匹配，三行核心代码），其余全部用现成
预训练模型——这是项目的既定原则，不从零造轮子。

**一个反向的信息流**：和声几乎总在强拍换，所以用阶段 3 的和弦变化点回头修正
阶段 2 的小节线相位。madmom 的 downbeat 模型只看声学特征、看不到和弦，
这是它拿不到的信息。实测小节线 F 从 0.514 提到 0.621。

## 已知限制

1. **和弦词汇表只有 25 类**：12 大三 + 12 小三 + 无和弦。七和弦、挂留、减、增
   全部被压扁到最近的大/小三和弦。这是 madmom 模型本身的限制。
2. **倍速/半速错误 14/60**。工具会用 librosa 独立估计交叉验证并报警，但报警后
   需要人耳定夺，然后用 `--min-bpm` 锁定。感知先验仲裁已实测无效（见 EVALUATION）。
3. **假设整曲一个调**，不做分段调性分析。转调的曲子会被平均成四不像。
4. **拍号只支持 3/4 和 4/4**（madmom 需要预先给定候选，5/4、7/8 不支持）。
5. **爵士与放克基本不可用**，见上方分风格数据。
6. **纯 CPU**，约 3-6 倍实时率。一首 4 分钟的歌约 1-2 分钟。

## 许可证

代码采用 [MIT](LICENSE)。

⚠️ **但整条流水线不能商用**：madmom 的代码是 BSD，而它捆绑的**预训练模型是
CC BY-NC-SA 4.0，禁止商业用途**。本仓库不重新分发这些模型，但任何运行本流水线
的人都在使用它们。商用前请自行确认。

## 致谢

- [madmom](https://github.com/CPJKU/madmom) — 节拍、小节线、和弦的预训练模型
- [librosa](https://librosa.org/) — 音频加载与 chroma 特征
- [mir_eval](https://github.com/mir-evaluation/mir_eval) — 评测指标
- [GuitarSet](https://zenodo.org/records/3371780) — 评测用的人工标注数据集（CC BY-4.0）
- Krumhansl & Kessler (1982)、Temperley (2001) — 调性模板
