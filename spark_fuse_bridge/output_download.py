"""On-demand download of a job's full output (videos, images, everything).

The automatic post-render download in jobs.py keeps images only, for the panel
preview. This module backs the panel's "Download outputs" button: it fetches
every file a job wrote to its ShareSync output folder into ComfyUI's output
directory under spark_fuse/<job id>/, keeping the original file names.

It works for any job id the logged-in account can read, including jobs from an
earlier ComfyUI session, render-queue batches, and failed jobs that still wrote
some files before stopping. Downloads run in a daemon thread; the HTTP routes
poll get_download_state(), the same pattern as jobs._JOBS.
"""
from __future__ import annotations

import re
import shutil
import tempfile
import threading
import time
import traceback
from pathlib import Path

from spark_fuse.sharesync import download_file, file_url_from_entry

from .config import load_settings, make_client

# job_id -> {status, files[], folder, error, started, finished}
_DOWNLOADS: dict[str, dict] = {}
_LOCK = threading.Lock()

# Spark Fuse job ids are UUIDs; anything else is rejected before it can reach a
# filesystem path.
_JOB_ID_RE = re.compile(r"^[0-9a-fA-F-]{8,64}$")

OUTPUT_SUBFOLDER = "spark_fuse"
# Workflows often save into deep folder trees (FJ/Day_to_night/sc_67/sht_2130/
# latest/). The messenger's download_outputs() stops at 3 levels, so this
# module walks the tree itself, far deeper, and keeps the folder layout.
_MAX_DEPTH = 16
_SAFE_PART = re.compile(r"[^A-Za-z0-9._ -]")


def valid_job_id(job_id: str) -> bool:
    return bool(_JOB_ID_RE.fullmatch(job_id or ""))


def _set(job_id: str, **fields) -> None:
    with _LOCK:
        _DOWNLOADS.setdefault(job_id, {}).update(fields)


def get_download_state(job_id: str) -> dict | None:
    with _LOCK:
        state = _DOWNLOADS.get(job_id)
        return None if state is None else {**state, "files": list(state.get("files", []))}


def start_download(job_id: str) -> dict:
    """Start (or report) the download of one job's outputs. Idempotent while a
    download for the same job is running."""
    if not valid_job_id(job_id):
        raise ValueError("Not a valid Spark Fuse job id.")
    with _LOCK:
        current = _DOWNLOADS.get(job_id)
        if current and current.get("status") == "downloading":
            return dict(current)
        _DOWNLOADS[job_id] = {"status": "downloading", "files": [], "folder": None,
                              "error": None, "started": time.time(), "finished": None}
    threading.Thread(target=_run, args=(job_id,), daemon=True).start()
    return get_download_state(job_id)


def recent_jobs(limit: int = 20) -> list[dict]:
    """Newest jobs visible to this account (the whole org), for the panel's
    job picker. List responses carry no output URL; that is resolved per job
    when a download starts."""
    settings = load_settings()
    client = make_client(settings)
    try:
        client.login()
        jobs = client.list_jobs()
    finally:
        close = getattr(client, "close", None)
        if callable(close):
            try:
                close()
            except Exception:  # noqa: BLE001
                pass
    jobs.sort(key=lambda j: j.created_at or "", reverse=True)
    return [{"id": j.id, "status": j.status, "created_at": j.created_at,
             "instance_type": j.instance_type_name}
            for j in jobs[:limit]]


def _resolve_output_url(client, job_id: str) -> str | None:
    """The output URL can lag a terminal status by a few seconds; retry briefly."""
    for attempt in range(6):
        job = client.get_job(job_id)
        base_url = job.output.share_sync_base_url if job.output else None
        if base_url:
            return base_url
        if not job.is_terminal:
            raise RuntimeError(f"Job is still {job.status}; download once it has finished.")
        if attempt < 5:
            time.sleep(2)
    return None


def _safe_part(name: str) -> str:
    """One path component, made safe for Windows and Linux filesystems."""
    from urllib.parse import unquote
    name = unquote(name).strip().strip(".")
    name = _SAFE_PART.sub("_", name)
    return name or "_"


def _walk_outputs(client, base_url: str) -> list[tuple[list[str], str]]:
    """List every file under the job's output folder, at any depth.

    Returns [(relative path parts, download URL)], depth-first. Uses the
    messenger's PROPFIND (client.list_outputs); entries are skipped when they
    are the collection itself.
    """
    from urllib.parse import urlparse
    found: list[tuple[list[str], str]] = []

    def walk(url: str, parts: list[str], depth: int) -> None:
        own_path = urlparse(url).path.rstrip("/")
        for entry in client.list_outputs(url):
            if entry.href.rstrip("/") == own_path:
                continue
            child_url = file_url_from_entry(url, entry)
            name = _safe_part(entry.name)
            if entry.is_collection:
                if depth < _MAX_DEPTH:
                    walk(child_url.rstrip("/") + "/", parts + [name], depth + 1)
            else:
                found.append((parts + [name], child_url))

    walk(base_url.rstrip("/") + "/", [], 0)
    return found


def _unique_path(folder: Path, name: str) -> Path:
    target = folder / name
    stem, suffix, i = target.stem, target.suffix, 1
    while target.exists():
        target = folder / f"{stem}_{i}{suffix}"
        i += 1
    return target


def _run(job_id: str) -> None:
    client = None
    dl_dir = None
    try:
        settings = load_settings()
        client = make_client(settings)
        client.login()
        base_url = _resolve_output_url(client, job_id)
        if not base_url:
            raise RuntimeError("Spark Fuse returned no output folder for this job.")

        import folder_paths  # ComfyUI-provided; available at runtime
        out_root = Path(folder_paths.get_output_directory())
        dest = out_root / OUTPUT_SUBFOLDER / job_id[:8]
        dest.mkdir(parents=True, exist_ok=True)
        subfolder = f"{OUTPUT_SUBFOLDER}/{job_id[:8]}"
        _set(job_id, folder=str(dest))

        # Download into a temp dir first, then move into place, so a failed
        # download never leaves half-written files in ComfyUI's output folder.
        dl_dir = Path(tempfile.mkdtemp(prefix="spark-fuse-dl-"))
        remote = _walk_outputs(client, base_url)
        _set(job_id, total=len(remote))

        files = []
        token = client._ensure_token()  # messenger exposes no public token getter
        for i, (rel_parts, url) in enumerate(remote, 1):
            _set(job_id, progress=f"{i}/{len(remote)} {'/'.join(rel_parts)}")
            tmp_path = dl_dir.joinpath(*rel_parts)
            download_file(url, token, tmp_path, client=client._http)
            target_dir = dest.joinpath(*rel_parts[:-1])
            target_dir.mkdir(parents=True, exist_ok=True)
            target = _unique_path(target_dir, rel_parts[-1])
            shutil.move(str(tmp_path), str(target))
            rel_dir = "/".join([subfolder, *rel_parts[:-1]])
            files.append({"name": target.name, "subfolder": rel_dir,
                          "size": target.stat().st_size})
        # Media first, the job log last, so the panel lists what matters on top.
        files.sort(key=lambda f: (f["name"].lower().endswith(".log"),
                                  f["subfolder"], f["name"]))
        _set(job_id, status="done", files=files, finished=time.time(), progress=None,
             error=None if files else "The job's output folder is empty.")
    except Exception as exc:  # noqa: BLE001 - surface every failure in the panel
        traceback.print_exc()
        _set(job_id, status="failed", error=str(exc), finished=time.time())
    finally:
        if dl_dir is not None:
            shutil.rmtree(dl_dir, ignore_errors=True)
        close = getattr(client, "close", None)
        if callable(close):
            try:
                close()
            except Exception:  # noqa: BLE001
                pass
