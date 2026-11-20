"""Per-workflow outputs for the render queue (Halo Hues addition).

The runner moves each manifest entry's files into /output/wf_NN/ as soon as
that workflow finishes. This module fetches those files from the job's
ShareSync output folder so the panel can show each finished video while the
rest of the queue is still rendering:

* fetch_live() starts as soon as the runner logs "workflow N/T succeeded".
  It polls wf_NN/ on ShareSync and downloads the videos once their sizes stop
  changing (the output folder is uploaded while the job runs, so a file can
  be listed before its upload has finished). If Spark only publishes outputs
  when the whole job ends, it simply finds nothing and the final pass below
  delivers the videos at the end of the batch instead.
* collect_final() runs after the batch's job has finished. It downloads any
  video the live pass missed or got only partly (size mismatch), plus the
  images the panel preview has always shown, in one walk of wf_NN/.

Videos are saved to ComfyUI/output/spark_fuse/queue_<queue id>/NN_<label>/,
keeping the workflow's own subfolders. Images keep the old behaviour: they
go to the ComfyUI output root under fresh sequential names.
"""
from __future__ import annotations

import re
import shutil
import tempfile
import threading
import time
import traceback
from pathlib import Path
from urllib.parse import unquote, urlparse

from spark_fuse.sharesync import download_file, file_url_from_entry

VIDEO_EXTS = (".mp4", ".webm", ".mov", ".mkv", ".m4v", ".avi", ".gif")
IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".webp")
OUTPUT_SUBFOLDER = "spark_fuse"
_MAX_DEPTH = 16
_SAFE_PART = re.compile(r"[^A-Za-z0-9._ -]")

# Live fetch: how often to look at wf_NN/ and for how long before leaving it to
# the end-of-batch pass. A file counts as complete once two listings this far
# apart report the same non-zero size.
_LIVE_POLL_SECONDS = 20
_LIVE_GIVE_UP_SECONDS = 15 * 60

# One lock per (queue, item) so the live and final passes never write the same
# file at the same time.
_ITEM_LOCKS: dict[tuple[str, int], threading.Lock] = {}
_LOCKS_GUARD = threading.Lock()


def _item_lock(qid: str, index: int) -> threading.Lock:
    with _LOCKS_GUARD:
        return _ITEM_LOCKS.setdefault((qid, index), threading.Lock())


def _safe_part(name: str) -> str:
    name = unquote(name).strip().strip(".")
    return _SAFE_PART.sub("_", name) or "_"


def _walk(client, folder_url: str) -> list[tuple[list[str], str, int | None]]:
    """Every file under folder_url: [(relative parts, url, size)]. Missing
    folder (the workflow saved nothing yet) -> []."""
    found: list[tuple[list[str], str, int | None]] = []

    def walk(url: str, parts: list[str], depth: int) -> None:
        own = urlparse(url).path.rstrip("/")
        for entry in client.list_outputs(url):
            if entry.href.rstrip("/") == own:
                continue
            child = file_url_from_entry(url, entry)
            name = _safe_part(entry.name)
            if entry.is_collection:
                if depth < _MAX_DEPTH:
                    walk(child.rstrip("/") + "/", parts + [name], depth + 1)
            else:
                found.append((parts + [name], child, entry.content_length))

    try:
        walk(folder_url.rstrip("/") + "/", [], 0)
    except Exception:  # noqa: BLE001 - 404 before the folder exists, or a blip
        return []
    return found


def _wf_url(base_url: str, local_idx: int) -> str:
    return f"{base_url.rstrip('/')}/wf_{local_idx:02d}/"


def _dest_folder(qid: str, global_idx: int, label: str) -> tuple[Path, str]:
    import folder_paths  # ComfyUI-provided; available at runtime
    rel = f"{OUTPUT_SUBFOLDER}/queue_{qid}/{global_idx + 1:02d}_{_safe_part(label)[:40]}"
    return Path(folder_paths.get_output_directory()) / rel, rel


def _download_videos(client, qid: str, global_idx: int, label: str,
                     files: list[tuple[list[str], str, int | None]]) -> list[dict]:
    """Download the video files in `files` that are not already saved with
    the same size. Returns every video now on disk for this item."""
    dest, rel_root = _dest_folder(qid, global_idx, label)
    token = client._ensure_token()  # messenger exposes no public token getter
    saved = []
    for parts, url, size in files:
        if not parts[-1].lower().endswith(VIDEO_EXTS):
            continue
        target = dest.joinpath(*parts)
        if not (target.is_file() and size and target.stat().st_size == size):
            tmp_dir = Path(tempfile.mkdtemp(prefix="spark-fuse-qv-"))
            try:
                tmp = tmp_dir / parts[-1]
                download_file(url, token, tmp, client=client._http)
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(tmp), str(target))
            finally:
                shutil.rmtree(tmp_dir, ignore_errors=True)
        saved.append({"name": target.name,
                      "subfolder": "/".join([rel_root, *parts[:-1]]),
                      "size": target.stat().st_size})
    return saved


def _resolve_base_url(client, job_id: str) -> str | None:
    try:
        job = client.get_job(job_id)
    except Exception:  # noqa: BLE001
        return None
    return job.output.share_sync_base_url if job.output else None


def fetch_live(client, qid: str, job_id: str, local_idx: int, global_idx: int,
               label: str, on_videos, log, still_wanted) -> None:
    """Start a background fetch of one finished workflow's videos.

    on_videos(list) is called with the saved videos; still_wanted() returning
    False (queue gone, item already complete) stops the loop early.
    """
    def run() -> None:
        started = time.time()
        last_sizes: dict[str, int | None] | None = None
        try:
            while time.time() - started < _LIVE_GIVE_UP_SECONDS and still_wanted():
                base_url = _resolve_base_url(client, job_id)
                files = _walk(client, _wf_url(base_url, local_idx)) if base_url else []
                videos = [f for f in files if f[0][-1].lower().endswith(VIDEO_EXTS)]
                sizes = {f[1]: f[2] for f in videos}
                if videos and sizes == last_sizes and all(sizes.values()):
                    with _item_lock(qid, global_idx):
                        saved = _download_videos(client, qid, global_idx, label, videos)
                    if saved:
                        on_videos(saved)
                        log(f"[queue] workflow {global_idx + 1}: {len(saved)} video(s) ready "
                            "in the panel")
                    return
                last_sizes = sizes if videos else None
                time.sleep(_LIVE_POLL_SECONDS)
        except Exception as exc:  # noqa: BLE001 - the final pass still runs
            log(f"[queue] workflow {global_idx + 1}: live video fetch stopped ({exc}); "
                "it will download when the batch finishes")
            traceback.print_exc()

    threading.Thread(target=run, daemon=True).start()


def collect_final(client, qid: str, base_url: str, local_idx: int, global_idx: int,
                  label: str) -> tuple[list[str], list[dict]]:
    """After the batch's job has finished: download this workflow's images
    (into the ComfyUI output root, as before) and any videos the live pass
    missed. Returns (image names, videos)."""
    from . import jobs  # local import: jobs imports this package's config

    files = _walk(client, _wf_url(base_url, local_idx))
    if not files:
        return [], []
    with _item_lock(qid, global_idx):
        videos = _download_videos(client, qid, global_idx, label, files)

    import folder_paths  # ComfyUI-provided; available at runtime
    out_dir = Path(folder_paths.get_output_directory())
    out_dir.mkdir(parents=True, exist_ok=True)
    token = client._ensure_token()
    images = []
    for parts, url, _size in files:
        if not parts[-1].lower().endswith(IMAGE_EXTS):
            continue
        tmp_dir = Path(tempfile.mkdtemp(prefix="spark-fuse-qi-"))
        try:
            tmp = tmp_dir / parts[-1]
            download_file(url, token, tmp, client=client._http)
            name = jobs._unique_output_name(out_dir, parts[-1])
            shutil.move(str(tmp), str(out_dir / name))
            images.append(name)
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)
    return images, videos
