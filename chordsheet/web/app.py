"""本地网页服务：上传音频 → 分析 → 返回可视化用的 JSON。

分析一首 4 分钟的歌要 1-2 分钟 CPU，所以放后台线程跑，前端轮询进度。
干转圈等两分钟体验很差，而且看不出是卡住了还是在正常工作。
"""

from __future__ import annotations

import io
import shutil
import tempfile
import threading
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, HTTPException, Response, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

STATIC_DIR = Path(__file__).parent / "static"
# 上传的音频要留着给前端播放，放临时目录，进程退出时清理
UPLOAD_DIR = Path(tempfile.gettempdir()) / "chordsheet-uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

SUPPORTED = {".wav", ".flac", ".mp3", ".ogg", ".aiff", ".au"}


@dataclass
class Job:
    """一次分析任务。"""

    id: str
    filename: str
    audio_path: Path
    stage: str = "排队中"
    progress: float = 0.0
    done: bool = False
    error: str | None = None
    result: dict[str, Any] | None = field(default=None, repr=False)
    # 供 MIDI 导出用，不进 JSON
    cells: list = field(default_factory=list, repr=False)
    tempo: float = 120.0


JOBS: dict[str, Job] = {}
app = FastAPI(title="chordsheet")


def _analyze(job: Job, meters: tuple[int, ...], min_bpm: float, max_bpm: float) -> None:
    """后台线程里跑完整流水线，逐阶段更新进度。"""
    try:
        import librosa

        from chordsheet.beats import MADMOM_SAMPLE_RATE, track_beats
        from chordsheet.chords import (
            NO_CHORD,
            ChordResult,
            assign_chords_to_bars,
            fill_no_chord_cells,
            is_diatonic,
            recognize_chords,
            rephase_by_chord_changes,
            snap_chords_to_beats,
        )
        from chordsheet.key import detect_key, mean_chroma
        from chordsheet.midi import cells_to_notes, chord_notes

        job.stage, job.progress = "加载音频", 0.05
        y, sr = librosa.load(str(job.audio_path), sr=MADMOM_SAMPLE_RATE, mono=True)
        if len(y) == 0:
            raise ValueError("音频读出来是空的")
        duration = len(y) / sr

        job.stage, job.progress = "识别调式", 0.15
        chroma = mean_chroma(y, sr, harmonic=False)
        key_result = detect_key(chroma, profile="krumhansl")
        key_alt = detect_key(chroma, profile="temperley")

        job.stage, job.progress = "检测节拍与小节线", 0.30
        beats = track_beats(y, sr, meters=meters, min_bpm=min_bpm, max_bpm=max_bpm)

        job.stage, job.progress = "识别和弦", 0.60
        segments = recognize_chords(y, sr, route="cnn")

        job.stage, job.progress = "对齐到小节", 0.92
        beats = rephase_by_chord_changes(beats, segments)
        bar_chords = assign_chords_to_bars(segments, beats.bars)
        chords = ChordResult(
            segments=segments, bar_chords=bar_chords, route="cnn", key=key_result.key
        )
        # 界面用 full_bars：从第 0 秒起、一秒不漏。bars 从第一条小节线开始，
        # 音乐上更正确，但界面上开头几秒凭空消失是明显缺陷。
        #
        # 和弦边界吸附到拍点网格。madmom 的分段是 10fps 时间轴的产物，
        # 边界常落在拍与拍之间，看谱时就是「和弦换得不在拍上」。
        cells = snap_chords_to_beats(segments, beats.times, beats.full_bars, duration=duration)
        # CRF 把握不大时会输出 N，但「模型不确定」和「这里没声音」是两回事。
        # 有声音的 N 补一个最接近的三和弦，真正的空拍留空。
        cells = fill_no_chord_cells(cells, y, sr)
        job.cells = cells
        job.tempo = beats.tempo

        job.result = {
            "filename": job.filename,
            "audio_url": f"/api/audio/{job.id}",
            "duration": duration,
            "key": {
                "primary": key_result.key,
                "alternative": key_alt.key,
                "agree": key_result.key == key_alt.key,
                "relative": key_result.relative_key,
                "margin": key_result.margin,
                "signature": key_result.key_signature_label(),
                "chroma": [float(v) for v in chroma],
            },
            "beats": {
                "tempo": beats.tempo,
                "meter": beats.meter,
                "stability": beats.tempo_stability,
                "reference_tempo": beats.reference_tempo,
                "agreement": beats.tempo_agreement,
                "times": [float(t) for t in beats.times],
                "positions": [int(p) for p in beats.positions],
                "downbeats": [float(t) for t in beats.downbeats],
                "anacrusis_beats": beats.anacrusis_beats,
                "uncovered_head": beats.uncovered_head,
            },
            "chords": {
                "mean_coverage": chords.mean_coverage,
                "diatonic_ratio": chords.diatonic_ratio,
                "ambiguous_bars": chords.ambiguous_bars(),
                "bars": [
                    {
                        "index": bar.index,
                        "start": bar.start,
                        "end": bar.end,
                        "chord": bar.chord,
                        "coverage": bar.coverage,
                        "shares": [[label, float(s)] for label, s in bar.shares[:3]],
                        "diatonic": bar.chord != NO_CHORD
                        and is_diatonic(bar.chord, key_result.key),
                    }
                    for bar in bar_chords
                ],
                "segments": [{"start": s, "end": e, "chord": c} for s, e, c in segments],
                # 小节内细分。实测 45 个小节里 35 个内部其实有和弦变化，
                # 只给 bars 会把这些全抹平。
                "cells": [
                    {
                        "bar": cell.bar,
                        "start": cell.start,
                        "end": cell.end,
                        "chord": cell.chord,
                        "notes": chord_notes(cell.chord),
                        "diatonic": cell.chord != NO_CHORD
                        and is_diatonic(cell.chord, key_result.key),
                    }
                    for cell in cells
                ],
                "bar_lines": [{"index": i, "start": s, "end": e} for i, s, e in beats.full_bars],
            },
            # 卷帘画的和导出的 MIDI 用同一份数据，避免「看到的和导出的不一样」
            "notes": [
                {"pitch": n.pitch, "start": n.start, "end": n.end, "velocity": n.velocity}
                for n in cells_to_notes(cells)
            ],
            "midi_url": f"/api/midi/{job.id}",
        }
        job.stage, job.progress, job.done = "完成", 1.0, True
    except Exception as exc:  # noqa: BLE001 - 任何失败都要如实回报给前端
        job.error = f"{type(exc).__name__}: {exc}"
        job.stage, job.done = "失败", True


@app.post("/api/analyze")
async def analyze(
    file: UploadFile = File(...),
    meters: str = "3,4",
    min_bpm: float = 55.0,
    max_bpm: float = 215.0,
) -> JSONResponse:
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in SUPPORTED:
        raise HTTPException(
            400,
            f"不支持的格式 {suffix or '(无扩展名)'}。"
            f"支持 {'、'.join(sorted(SUPPORTED))}；M4A/AAC 需要先用 ffmpeg 转码。",
        )

    job_id = uuid.uuid4().hex[:12]
    audio_path = UPLOAD_DIR / f"{job_id}{suffix}"
    with audio_path.open("wb") as fh:
        shutil.copyfileobj(file.file, fh)

    job = Job(id=job_id, filename=file.filename or audio_path.name, audio_path=audio_path)
    JOBS[job_id] = job

    parsed = tuple(int(m) for m in meters.split(",") if m.strip())
    threading.Thread(
        target=_analyze, args=(job, parsed or (3, 4), min_bpm, max_bpm), daemon=True
    ).start()
    return JSONResponse({"job_id": job_id})


@app.get("/api/status/{job_id}")
async def status(job_id: str) -> JSONResponse:
    job = JOBS.get(job_id)
    if job is None:
        raise HTTPException(404, "任务不存在")
    return JSONResponse(
        {
            "stage": job.stage,
            "progress": job.progress,
            "done": job.done,
            "error": job.error,
            "result": job.result,
        }
    )


@app.post("/api/midi")
async def midi_from_edits(payload: dict) -> Response:
    """把编辑后的和弦导出成 MIDI。

    编辑状态留在前端，服务端不存——刷新页面就回到识别结果，语义清楚，
    也不用管过期任务的清理。前端把当前的格子发过来即可。
    """
    from chordsheet.chords import ChordCell
    from chordsheet.midi import cells_to_notes, write_midi

    raw = payload.get("cells") or []
    if not raw:
        raise HTTPException(400, "没有可导出的和弦")

    cells = [
        ChordCell(int(c["bar"]), float(c["start"]), float(c["end"]), str(c["chord"])) for c in raw
    ]
    buffer = io.BytesIO()
    with tempfile.NamedTemporaryFile(suffix=".mid") as tmp:
        write_midi(cells_to_notes(cells), tmp.name, tempo_bpm=float(payload.get("tempo") or 120))
        buffer.write(Path(tmp.name).read_bytes())

    name = str(payload.get("filename") or "chords")
    return Response(
        buffer.getvalue(),
        media_type="audio/midi",
        headers={"Content-Disposition": f'attachment; filename="{name}.mid"'},
    )


@app.get("/api/midi/{job_id}")
async def midi(job_id: str) -> FileResponse:
    """导出和弦 MIDI。

    导出的是和弦构成音，不是歌里实际弹的音符——本项目做的是和弦识别。
    """
    from chordsheet.midi import export_chords

    job = JOBS.get(job_id)
    if job is None or not job.cells:
        raise HTTPException(404, "结果不存在或还没分析完")

    path = UPLOAD_DIR / f"{job_id}.mid"
    export_chords(job.cells, path, tempo_bpm=job.tempo)
    stem = Path(job.filename).stem or "chords"
    return FileResponse(path, media_type="audio/midi", filename=f"{stem}_chords.mid")


@app.get("/api/audio/{job_id}")
async def audio(job_id: str) -> FileResponse:
    job = JOBS.get(job_id)
    if job is None or not job.audio_path.exists():
        raise HTTPException(404, "音频不存在")
    return FileResponse(job.audio_path)


class NoCacheStatic(StaticFiles):
    """禁用浏览器缓存。

    界面是随开发迭代改的，浏览器缓存住旧的 HTML/JS 会让人以为「功能没生效」，
    而实际上跑的是旧代码。本地工具没有带宽压力，直接每次都取新的。
    """

    def is_not_modified(self, *args, **kwargs) -> bool:  # noqa: ARG002
        return False

    async def get_response(self, path: str, scope):
        response = await super().get_response(path, scope)
        response.headers["Cache-Control"] = "no-store, must-revalidate"
        return response


app.mount("/", NoCacheStatic(directory=STATIC_DIR, html=True), name="static")


def _is_wsl() -> bool:
    try:
        return "microsoft" in Path("/proc/version").read_text().lower()
    except OSError:
        return False


def _open_browser(url: str) -> None:
    """打开浏览器。WSL 需要特殊处理。

    WSL 里通常没有 xdg-open / wslview，Python 的 webbrowser.open 找不到浏览器会
    **静默失败**——功能等于没有，还不报错。所以在 WSL 下改成调 Windows 侧的
    cmd.exe 打开宿主浏览器，并且无论成败都把网址打印出来让用户能手动复制。
    """
    if _is_wsl():
        import subprocess

        for cmd in (["wslview", url], ["cmd.exe", "/c", "start", "", url]):
            try:
                subprocess.run(cmd, check=True, capture_output=True, timeout=10)
                return
            except (OSError, subprocess.SubprocessError):
                continue
        print("（无法自动打开浏览器，请手动复制上面的网址）")
        return

    import webbrowser

    if not webbrowser.open(url):
        print("（无法自动打开浏览器，请手动复制上面的网址）")


def serve(host: str = "0.0.0.0", port: int = 8000, open_browser: bool = True) -> int:  # noqa: S104
    """启动本地服务。

    默认绑 0.0.0.0 而不是 127.0.0.1：WSL 里服务跑在 Linux 侧、浏览器在 Windows 侧，
    绑回环地址时宿主不一定能访问到。要限制只允许本机访问就显式传 --host 127.0.0.1。
    """
    import socket

    import uvicorn

    shown = "localhost" if host in ("0.0.0.0", "127.0.0.1") else host  # noqa: S104
    url = f"http://{shown}:{port}"
    print(f"chordsheet 界面已启动: {url}")

    if host == "0.0.0.0":  # noqa: S104
        try:
            lan = socket.gethostbyname(socket.gethostname())
            print(f"打不开就试: http://{lan}:{port}")
        except OSError:
            pass

    print("音频只在本机处理，不会上传到任何地方。Ctrl+C 停止。")
    if open_browser:
        import threading

        threading.Timer(1.0, lambda: _open_browser(url)).start()
    uvicorn.run(app, host=host, port=port, log_level="warning")
    return 0
