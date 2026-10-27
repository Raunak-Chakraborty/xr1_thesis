"""Read indexed, lossless XR1 Doosan PNG-in-ZIP camera caches (v1).

Opt-in only; does not replace the existing Decord or PNG loader. Existing V2
episode JSON paths remain literal identity keys, even after moving to CLAIX.
Caching is strictly process-local so DataLoader workers never share ZIP handles.
"""
from __future__ import annotations

import json
import os
import zipfile
from collections import OrderedDict
from functools import lru_cache
from pathlib import Path

from PIL import Image

from mibot.utils.io import get_value

_SCHEMA = "doosan-xr1-rgb-png-zip-v1"
_CAMERAS = frozenset(("external_camera_2", "tcp_camera"))
_MAX_OPEN = 12
_pid = None
_handles = OrderedDict()


@lru_cache(maxsize=2)
def _manifest(root_string: str):
    path = Path(root_string) / "media_manifest.jsonl"
    if not path.is_file():
        raise FileNotFoundError(f"Missing Doosan ZIP cache manifest: {path}")
    by_source = {}
    with path.open(encoding="utf-8") as stream:
        for line_no, line in enumerate(stream, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            source = row["original_source_path"]
            archive = Path(row["archive"])
            if (not source or source in by_source or archive.is_absolute()
                    or ".." in archive.parts):
                raise ValueError(f"Invalid/duplicate ZIP manifest record {line_no}")
            if row["camera"] not in _CAMERAS or int(row["frames"]) < 1:
                raise ValueError(f"Invalid camera or frame count at manifest line {line_no}")
            by_source[source] = row
    if len(by_source) != 646:
        raise RuntimeError(f"Expected 646 video records, got {len(by_source)}")
    return by_source


def _get_archive(root: Path, row):
    """Open a ZIP once per worker/process, verify its header and sidecar."""
    global _pid, _handles
    pid = os.getpid()
    if pid != _pid:
        for handle in _handles.values():
            handle.close()
        _handles = OrderedDict()
        _pid = pid
    archive_path = root / row["archive"]
    key = str(archive_path)
    if key in _handles:
        _handles.move_to_end(key)
        return _handles[key]
    if not archive_path.is_file():
        raise FileNotFoundError(f"Missing offline ZIP archive, no MP4 fallback: {archive_path}")
    sidecar = archive_path.with_suffix(".metadata.json")
    if not sidecar.is_file():
        raise FileNotFoundError(f"Missing ZIP provenance sidecar: {sidecar}")
    meta = json.loads(sidecar.read_text(encoding="utf-8"))
    if (meta.get("schema") != _SCHEMA
            or meta.get("original_source_path") != row["original_source_path"]
            or meta.get("frames") != int(row["frames"])
            or meta.get("crop_bbox") != row.get("crop_bbox")
            or meta.get("archive_bytes") != archive_path.stat().st_size):
        raise RuntimeError(f"ZIP metadata mismatch: {archive_path}")
    archive = zipfile.ZipFile(archive_path, mode="r", allowZip64=True)
    try:
        if len(archive.infolist()) != int(row["frames"]):
            raise RuntimeError(f"ZIP frame count mismatch: {archive_path}")
        archive.getinfo("frame_000000000.png")
        archive.getinfo(f'frame_{int(row["frames"])-1:09d}.png')
    except BaseException:
        archive.close()
        raise
    _handles[key] = archive
    if len(_handles) > _MAX_OPEN:
        _, evicted = _handles.popitem(last=False)
        evicted.close()
    return archive


def decode_images(traj, keys, frame):
    """Same callable shape as the original Doosan video and PNG decoders.

    Images have *already* been cropped/resized with the exact deterministic
    XR1 pipeline. Caller may subsequently run the original _augment (its
    resize-to-same-size is pixel-preserving; randomized jitter remains online).
    """
    root_value = os.environ.get("XR1_DOOSAN_ZIP_CACHE_ROOT", "")
    if not root_value:
        raise RuntimeError("XR1_DOOSAN_ZIP_CACHE_ROOT must be set")
    root = Path(root_value).resolve()
    manifest = _manifest(str(root))
    index = max(int(frame), 0)
    images = []
    for key in keys:
        camera = str(key).split(".")[-1]
        if camera not in _CAMERAS:
            raise ValueError(f"Unsupported Doosan camera token: {key}")
        infos = get_value(traj, key)
        if infos is None:
            raise ValueError(f"Missing camera metadata: {key}")
        for info in infos:
            source = os.fspath(info["path"])
            row = manifest.get(source)
            if row is None:
                raise FileNotFoundError(f"Camera source not in ZIP manifest: {source}")
            if row["camera"] != camera or row.get("crop_bbox") != info.get("crop_bbox"):
                raise RuntimeError(f"Camera/crop provenance mismatch: {source}")
            idx = index + int(info.get("start", 0))
            if idx < 0 or idx >= int(row["frames"]):
                raise IndexError(f"Frame {idx} out of range for {source}")
            zipped = _get_archive(root, row)
            with zipped.open(f"frame_{idx:09d}.png", "r") as payload:
                with Image.open(payload) as image:
                    images.append(image.convert("RGB"))
    return images
