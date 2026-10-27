"""Fail-closed offline RGB frame loader for the Doosan XR1 dataset.

This module is deliberately NOT wired into DoosanJsonDataset yet. Only enable
it after verifying FFmpeg frame-index / pixel parity with the existing loader.
No Decord imports, no MP4 reads, no online fallback.
"""
from __future__ import annotations

import hashlib
import os
from pathlib import Path

import numpy as np
from PIL import Image

from mibot.data.datasets.json_dataset import JsonDataset
from mibot.utils.io import get_value


def decode_images(traj, keys, frame):
    root_value = os.environ.get('XR1_DOOSAN_OFFLINE_FRAME_ROOT', '')
    if not root_value:
        raise RuntimeError('XR1_DOOSAN_OFFLINE_FRAME_ROOT is required for offline image loading')
    root = Path(root_value)
    images = []
    frame = max(int(frame), 0)
    for key in keys:
        infos = get_value(traj, key)
        if infos is None:
            raise ValueError(f'Missing camera in Doosan trajectory: {key}')
        for info in infos:
            source = os.fspath(info['path'])
            idx = frame + int(info.get('start', 0))
            key_hash = hashlib.sha256(source.encode('utf-8')).hexdigest()[:24]
            cached = root / 'videos' / key_hash / f'frame_{idx:09d}.png'
            if not cached.is_file():
                raise FileNotFoundError(f'Offline Doosan frame cache miss (no Decord fallback): {cached}')
            with Image.open(cached) as pil:
                rgb = np.asarray(pil.convert('RGB'))
            cropped = JsonDataset._crop_bbox(rgb[None, ...], info.get('crop_bbox'))
            images.extend(Image.fromarray(item) for item in cropped)
    return images
