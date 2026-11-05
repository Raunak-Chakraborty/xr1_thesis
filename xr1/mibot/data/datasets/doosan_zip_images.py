"""Read indexed, lossless XR1 Doosan PNG-in-ZIP camera caches (v1).

Opt-in only; does not replace the existing Decord or PNG loader. Existing V2
episode JSON paths remain literal identity keys, even after moving to CLAIX.
Caching is strictly process-local so DataLoader workers never share ZIP handles.
"""
from __future__ import annotations

import atexit
import json
import os
import zipfile
import resource
import sys
from collections import OrderedDict
from functools import lru_cache
from pathlib import Path

from PIL import Image

from mibot.utils.doosan_io_cache import CacheMetrics, object_bytes
from mibot.utils.loader_trace import stage, traced, TracedPayload

_SCHEMA = "doosan-xr1-rgb-png-zip-v1"
_CAMERAS = frozenset(("external_camera_2", "tcp_camera"))


@lru_cache(maxsize=2)
@traced('zip_manifest_load')
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


def _signature(stat):
    # Size alone misses same-length rewrites and atomic replacements.
    return (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)


def _row_identity(row):
    return (row['original_source_path'], row['camera'], int(row['frames']),
            tuple(row['crop_bbox']) if row.get('crop_bbox') is not None else None)


def _directory_bytes(archive):
    # ZipInfo uses slots, not __dict__. Include all retained fields; count
    # aliases repeatedly for a conservative estimate, plus fixed handle costs.
    return (4096 + sys.getsizeof(archive.filelist) + sys.getsizeof(archive.NameToInfo)
            + sum(sys.getsizeof(info) + sum(object_bytes(getattr(info, slot, None))
                  for slot in zipfile.ZipInfo.__slots__) for info in archive.filelist))


class ZipArchiveCache:
    """Single-threaded worker-owned LRU with an explicit integrity policy.

    strict (default): two path stats before every reuse. immutable: no filesystem
    identity checks on live-handle hits; requires an externally frozen dataset
    (manifest, ZIPs, sidecars and episode JSONs) for the entire worker lifetime.
    Both policies validate identities on every miss/reopen. Neither stat identity
    nor member CRC is a cryptographic guarantee or a transactional read lock.
    """
    def __init__(self, max_open=32, max_bytes=128 * 1024**2,
                 metadata_entries=1024, metadata_bytes=8 * 1024**2, report_every=0,
                 integrity_policy='strict'):
        if max_open < 1 or min(max_bytes, metadata_entries, metadata_bytes, report_every) < 0:
            raise ValueError('Invalid ZIP cache bounds')
        if integrity_policy not in ('strict', 'immutable'):
            raise ValueError(f'Invalid ZIP integrity policy: {integrity_policy!r}')
        self.policy = (max_open, max_bytes, metadata_entries, metadata_bytes, report_every,
                       integrity_policy)
        self.handles = OrderedDict()
        self.metadata = OrderedDict()
        self.pid = None
        self._local()

    def _local(self):
        if self.pid == os.getpid():
            return
        self.close()
        self.pid = os.getpid()
        soft, _ = resource.getrlimit(resource.RLIMIT_NOFILE)
        # Leave at least 64 descriptors, and at least half the limit, for
        # tensors, worker queues, diagnostics and other libraries.
        self.max_open = self.policy[0] if soft == resource.RLIM_INFINITY else min(
            self.policy[0], max(0, min(soft // 2, soft - 64)))
        if self.max_open < 1:
            raise RuntimeError('RLIMIT_NOFILE leaves no safe ZIP cache capacity')
        self.metrics = CacheMetrics(('hits', 'misses', 'opens', 'reopens', 'sidecar_reads',
                                     'metadata_hits', 'evictions', 'metadata_evictions',
                                     'invalidations', 'oversized'), self.policy[4])
        # Always record the actual worker policy, independently of diagnostics.
        # One startup record per cache/PID, never per-frame logging.
        print('XR1_ZIP_CACHE_POLICY ' + json.dumps(dict(
            pid=self.pid, integrity_policy=self.policy[5], max_open=self.max_open,
            max_bytes=self.policy[1], metadata_entries=self.policy[2],
            metadata_bytes=self.policy[3],
            hit_path_stats=2 if self.policy[5] == 'strict' else 0,
            immutable_dataset_required=self.policy[5] == 'immutable',
        ), sort_keys=True), flush=True)

    def __getstate__(self):
        return {'policy': self.policy}

    def __setstate__(self, state):
        self.__init__(*state['policy'])

    def close(self):
        for archive, _, _ in self.handles.values():
            archive.close()
        self.handles.clear()
        self.metadata.clear()
        self.bytes = self.meta_bytes = 0

    def _evict(self, key):
        archive, _, size = self.handles.pop(key)
        self.bytes -= size
        archive.close()
        self.metrics.add('evictions')

    def _forget_metadata(self, key):
        cached = self.metadata.pop(key, None)
        if cached is not None:
            self.meta_bytes -= cached[1]
            self.metrics.add('metadata_evictions')

    def snapshot(self):
        self._local()
        return dict(self.metrics.counts, pid=self.pid, handles=len(self.handles),
                    metadata_entries=len(self.metadata), estimated_bytes=self.bytes,
                    metadata_estimated_bytes=self.meta_bytes, max_open=self.max_open,
                    integrity_policy=self.policy[5])

    def release(self, archive):
        # Oversized directories are usable for this read but never retained.
        if not any(entry[0] is archive for entry in self.handles.values()):
            archive.close()

    def get(self, root, row):
        self._local()
        archive_path = root / row['archive']
        sidecar = archive_path.with_suffix('.metadata.json')
        key = os.fspath(archive_path)
        identity = _row_identity(row)
        held = self.handles.get(key)
        # Explicit immutable opt-in only. Row provenance still compares in RAM;
        # replacement/mutation of paths is NOT detected here. Reopens below stat
        # both paths even under immutable policy and invalidate changed records.
        if (held is not None and self.policy[5] == 'immutable'
                and held[1][2] == identity):
            return self._hit(key, held)
        try:
            signature = (_signature(archive_path.stat()), _signature(sidecar.stat()), identity)
        except OSError:
            if key in self.handles:
                self._evict(key)
            self._forget_metadata(key)
            raise
        if held is not None and held[1] == signature:
            return self._hit(key, held)
        self.metrics.add('misses')
        if held is not None:
            self._evict(key)
        previous = self.metadata.get(key)
        if previous is not None and previous[0] != signature:
            self.metrics.add('invalidations')
            self._forget_metadata(key)
            previous = None
        if previous is None:
            self.metrics.add('sidecar_reads')
            with stage('zip_sidecar_load'):
                with sidecar.open('rb') as stream:
                    # Provenance files are small; never parse an unbounded file.
                    raw = stream.read(65537)
                    if len(raw) > 65536 or _signature(os.fstat(stream.fileno())) != signature[1]:
                        raise RuntimeError(f'ZIP sidecar changed/oversized: {sidecar}')
                meta = json.loads(raw)
            if (not isinstance(meta, dict) or meta.get('schema') != _SCHEMA
                    or meta.get('original_source_path') != row['original_source_path']
                    or meta.get('frames') != int(row['frames'])
                    or meta.get('crop_bbox') != row.get('crop_bbox')
                    or meta.get('archive_bytes') != signature[0][2]):
                raise RuntimeError(f'ZIP metadata mismatch: {archive_path}')
        else:
            self.metrics.add('metadata_hits')
            self.metadata.move_to_end(key)
        # Evict BEFORE opening to respect the descriptor cap even on misses.
        while len(self.handles) >= self.max_open:
            self._evict(next(iter(self.handles)))
        with stage('zip_archive_open'):
            archive = zipfile.ZipFile(archive_path, mode='r', allowZip64=True)
        self.metrics.add('opens')
        if previous is not None:
            # Reopens counted within the bounded validated-metadata history.
            self.metrics.add('reopens')
        try:
            if _signature(os.fstat(archive.fp.fileno())) != signature[0]:
                raise RuntimeError(f'ZIP changed while opening: {archive_path}')
            count = int(row['frames'])
            if count < 1 or len(archive.infolist()) != count or (previous is None and any(
                    f'frame_{i:09d}.png' not in archive.NameToInfo for i in range(count))):
                raise RuntimeError(f'ZIP frame directory mismatch: {archive_path}')
            # Reject replacement during validation; no stale successful record.
            if previous is None and (
                    _signature(archive_path.stat()), _signature(sidecar.stat())) != signature[:2]:
                raise RuntimeError(f'ZIP/sidecar changed during validation: {archive_path}')
            size = previous[2] if previous is not None else _directory_bytes(archive)
            if previous is None:
                meta_size = object_bytes(key) + object_bytes(signature)
                if self.policy[2] and meta_size <= self.policy[3]:
                    while self.metadata and (len(self.metadata) >= self.policy[2]
                                             or self.meta_bytes + meta_size > self.policy[3]):
                        self._forget_metadata(next(iter(self.metadata)))
                    self.metadata[key] = (signature, meta_size, size)
                    self.meta_bytes += meta_size
            if size <= self.policy[1]:
                while self.handles and self.bytes + size > self.policy[1]:
                    self._evict(next(iter(self.handles)))
                self.handles[key] = (archive, signature, size)
                self.bytes += size
            else:
                self.metrics.add('oversized')
            self.metrics.report('zip', handles=len(self.handles), estimated_bytes=self.bytes)
            return archive
        except BaseException:
            archive.close()
            self._forget_metadata(key)
            raise

    def _hit(self, key, held):
        self.metrics.add('hits')
        self.handles.move_to_end(key)
        self.metrics.report('zip', handles=len(self.handles), estimated_bytes=self.bytes)
        return held[0]


_cache = None


def configure_cache(**options):
    global _cache
    policy = dict(max_open=32, max_bytes=128 * 1024**2, metadata_entries=1024,
                  metadata_bytes=8 * 1024**2, report_every=0, integrity_policy='strict')
    policy.update(options)
    if _cache is None or _cache.policy != tuple(policy.values()):
        replacement = ZipArchiveCache(**policy)
        close_cache()
        _cache = replacement
    return _cache


def close_cache():
    if _cache is not None:
        _cache.close()


def cache_info():
    return configure_cache().snapshot() if _cache is None else _cache.snapshot()


# Close inherited descriptors immediately after fork, before any child reads.
# PID checks also protect explicitly constructed caches; spawn pickles policy only.
if hasattr(os, 'register_at_fork'):
    os.register_at_fork(after_in_child=close_cache)
atexit.register(close_cache)


@traced('zip_archive_lookup')
def _get_archive(root: Path, row):
    cache = configure_cache() if _cache is None else _cache
    return cache.get(root, row)


def decode_images(traj, keys, frame):
    """Same callable shape as the original Doosan video and PNG decoders.

    Images have *already* been cropped/resized with the exact deterministic
    XR1 pipeline. Caller may subsequently run the original _augment (its
    resize-to-same-size is pixel-preserving; randomized jitter remains online).
    """
    from mibot.utils.io import get_value

    root_value = os.environ.get("XR1_DOOSAN_ZIP_CACHE_ROOT", "")
    if not root_value:
        raise RuntimeError("XR1_DOOSAN_ZIP_CACHE_ROOT must be set")
    root = Path(os.path.abspath(root_value))
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
            with stage('zip_camera', archive=str(root / row['archive']), camera=camera,
                       source=source, member=f'frame_{idx:09d}.png', camera_frame=idx):
                zipped = _get_archive(root, row)
                try:
                    with stage('zip_member_open'):
                        payload = zipped.open(f"frame_{idx:09d}.png", "r")
                    with payload:
                        with stage('png_decode'):
                            with Image.open(TracedPayload(payload)) as image:
                                images.append(image.convert("RGB"))
                            # PIL may stop before the ZIP member's end. Drain its
                            # remainder so ZipExtFile always checks the full CRC.
                            while payload.read(65536):
                                pass
                finally:
                    _cache.release(zipped)
    return images
