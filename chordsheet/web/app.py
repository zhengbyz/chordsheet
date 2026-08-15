"""本地网页服务：上传音频 → 分析 → 返回可视化用的 JSON。

分析一首 4 分钟的歌要 1-2 分钟 CPU，所以放后台线程跑，前端轮询进度。
干转圈等两分钟体验很差，而且看不出是卡住了还是在正常工作。
"""

from __future__ import annotations

import shutil
import tempfile
import threading
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, HTTPException, UploadFile
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
            is_diatonic,
            recognize_chords,
            rephase_by_chord_changes,
        )
        from chordsheet.key import detect_key, mean_chroma

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
            },
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


@app.get("/api/audio/{job_id}")
async def audio(job_id: str) -> FileResponse:
    job = JOBS.get(job_id)
    if job is None or not job.audio_path.exists():
        raise HTTPException(404, "音频不存在")
    return FileResponse(job.audio_path)


app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="static")


def serve(host: str = "127.0.0.1", port: int = 8000, open_browser: bool = True) -> int:
    """启动本地服务。"""
    import uvicorn

    url = f"http://{host}:{port}"
    print(f"chordsheet 界面已启动: {url}")
    print("音频只在本机处理，不会上传到任何地方。Ctrl+C 停止。")
    if open_browser:
        import threading
        import webbrowser

        threading.Timer(1.0, lambda: webbrowser.open(url)).start()
    uvicorn.run(app, host=host, port=port, log_level="warning")
    return 0
