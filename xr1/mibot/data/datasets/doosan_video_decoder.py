"""XR1 Doosan-only single-frame video decoder; native reader reuse is per process.

Mitigates Decord get_batch([one_frame]) hangs and repeated decoder thread creation.
Keeps XR1 camera ordering/cropping and sample contents unchanged.
"""
import os
from collections import OrderedDict

from decord import VideoReader, cpu
from PIL import Image
from mibot.utils.io import get_value
from mibot.data.datasets.json_dataset import JsonDataset

_MAX_OPEN = 12
_pid = None
_readers = OrderedDict()
_counters = {'opens': 0, 'hits': 0, 'evictions': 0, 'frames': 0}


def _reader(path):
    global _pid, _readers, _counters
    pid = os.getpid()
    if _pid != pid:
        # Do not share decoder readers across worker processes.
        _pid = pid
        _readers = OrderedDict()
        _counters = {'opens': 0, 'hits': 0, 'evictions': 0, 'frames': 0}
    path = os.fspath(path)
    if path in _readers:
        _readers.move_to_end(path)
        _counters['hits'] += 1
        return _readers[path]
    vr = VideoReader(path, ctx=cpu(0), num_threads=1)
    _readers[path] = vr
    _counters['opens'] += 1
    if len(_readers) > _MAX_OPEN:
        _readers.popitem(last=False)
        _counters['evictions'] += 1
    return vr


def decode_images(traj, keys, frame):
    images = []
    frame = max(int(frame), 0)
    for key in keys:
        infos = get_value(traj, key)
        if infos is None:
            raise ValueError(f"{key} missing in trajectory {traj.get('time', '<unknown>')}")
        for info in infos:
            vr = _reader(info['path'])
            idx = frame + int(info.get('start', 0))
            if idx < 0 or idx >= len(vr):
                raise IndexError(f"Frame {idx} outside [0,{len(vr)}) in {info['path']}")
            # Doosan requests exactly ONE frame. Avoid Decord's GetBatch C++ path.
            rgb = vr[idx].asnumpy()
            _counters['frames'] += 1
            cropped = JsonDataset._crop_bbox(rgb[None, ...], info.get('crop_bbox'))
            images.extend(Image.fromarray(im) for im in cropped)
    return images


def stats():
    return dict(_counters, pid=_pid, cached_readers=len(_readers), max_open=_MAX_OPEN)
