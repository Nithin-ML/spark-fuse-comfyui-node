"""Browser-playable previews for the queue video viewer (Halo Hues addition).

Browsers only play a few video formats: H.264 8-bit 4:2:0, VP8/VP9 and AV1.
ProRes .mov, H.265, 10-bit or 4:4:4 H.264 and similar studio formats show an
empty player at 0:00. When the panel's player can't decode a video, it asks
this module for a preview: a small H.264 copy made with ffmpeg and saved next
to the original in a _preview/ folder. The original file is never changed,
and the panel's Open / save links keep pointing at it.

A preview is made once and reused while it is newer than its source.
"""
from __future__ import annotations

import shutil
import subprocess
import threading
from pathlib import Path

PREVIEW_DIR = "_preview"
_LOCKS: dict[str, threading.Lock] = {}
_LOCKS_GUARD = threading.Lock()


def _lock_for(key: str) -> threading.Lock:
    with _LOCKS_GUARD:
        return _LOCKS.setdefault(key, threading.Lock())


def _ffmpeg() -> str:
    exe = shutil.which("ffmpeg")
    if exe:
        return exe
    try:  # VideoHelperSuite ships ffmpeg through imageio-ffmpeg
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError("ffmpeg was not found on this machine, so a browser "
                           "preview can't be made. Use Open to play the file in "
                           "your video player.") from exc


def _resolve(name: str, subfolder: str) -> tuple[Path, Path]:
    """The source file inside ComfyUI's output folder, refusing anything that
    points outside it."""
    import folder_paths  # ComfyUI-provided; available at runtime
    root = Path(folder_paths.get_output_directory()).resolve()
    src = (root / subfolder / name).resolve()
    if root not in src.parents or not src.is_file():
        raise FileNotFoundError("Video not found in ComfyUI's output folder.")
    return root, src


def make_preview(name: str, subfolder: str) -> dict:
    """Return {name, subfolder} of an H.264 preview of the given output video,
    creating it if needed. Blocking; run it in an executor."""
    root, src = _resolve(name, subfolder)
    dest = src.parent / PREVIEW_DIR / (src.stem + ".mp4")
    result = {"name": dest.name,
              "subfolder": dest.parent.relative_to(root).as_posix()}
    with _lock_for(str(dest)):
        if dest.is_file() and dest.stat().st_mtime >= src.stat().st_mtime:
            return result
        dest.parent.mkdir(parents=True, exist_ok=True)
        tmp = dest.with_name(dest.stem + ".part.mp4")
        cmd = [
            _ffmpeg(), "-hide_banner", "-loglevel", "error", "-y", "-i", str(src),
            "-map", "0:v:0", "-map", "0:a:0?",
            # Even dimensions, at most 1920 wide; 8-bit 4:2:0 H.264 plays everywhere.
            "-vf", "scale='min(1920,iw)':-2",
            "-c:v", "libx264", "-preset", "veryfast", "-crf", "20", "-pix_fmt", "yuv420p",
            "-c:a", "aac", "-b:a", "192k",
            "-movflags", "+faststart", str(tmp),
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
        if proc.returncode != 0 or not tmp.is_file():
            tmp.unlink(missing_ok=True)
            detail = (proc.stderr or "").strip().splitlines()[-1:] or ["unknown error"]
            raise RuntimeError(f"ffmpeg could not convert this video: {detail[0]}")
        tmp.replace(dest)
    return result
