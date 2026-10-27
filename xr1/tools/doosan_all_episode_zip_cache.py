#!/usr/bin/env python3
"""Offline CPU-only XR1 Doosan all-episode image cache: resumable ZIP per camera/video.

Input episode JSONs remain untouched. Each ZIP stores lossless, deterministic
XR1-pre-resized PNG images, with efficient random access by filename.
This program does not install an XR1 loader; that must be validated separately.
"""
import argparse
import hashlib
import io
import json
import math
import os
import subprocess
import sys
import tempfile
import time
import zipfile
from pathlib import Path

from PIL import Image, ImageChops

CAMERAS = ('external_camera_2', 'tcp_camera')
SCHEMA = 'doosan-xr1-rgb-png-zip-v1'

def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for buf in iter(lambda: f.read(4 * 1024 * 1024), b''):
            h.update(buf)
    return h.hexdigest()

def resized_like_xr1(image):
    width, height = image.size
    if max(width, height) / min(width, height) > 200:
        raise ValueError('Unsupported aspect ratio')
    factor, min_pixels, max_pixels = 32, 4 * 28 * 28, 160000
    new_h = max(factor, round(height / factor) * factor)
    new_w = max(factor, round(width / factor) * factor)
    if new_h * new_w > max_pixels:
        scale = math.sqrt(height * width / max_pixels)
        new_h = math.floor(height / scale / factor) * factor
        new_w = math.floor(width / scale / factor) * factor
    elif new_h * new_w < min_pixels:
        scale = math.sqrt(min_pixels / (height * width))
        new_h = math.ceil(height * scale / factor) * factor
        new_w = math.ceil(width * scale / factor) * factor
    return image.resize((new_w, new_h))  # same Pillow default as XR1

def source_on_this_host(path, old, new):
    p = Path(path)
    if old:
        original = Path(old)
        try:
            suffix = p.relative_to(original)
        except ValueError as e:
            raise ValueError(f'Video outside old source prefix: {p}') from e
        p = Path(new) / suffix
    if not p.is_file():
        raise FileNotFoundError(p)
    return p

def ffprobe_dimensions(path):
    cmd = ['ffprobe', '-v', 'error', '-select_streams', 'v:0',
           '-show_entries', 'stream=width,height', '-of', 'json', str(path)]
    obj = json.loads(subprocess.check_output(cmd, text=True))
    stream = obj['streams'][0]
    return int(stream['width']), int(stream['height'])

def manifest_entries(dataset, remap_old, remap_new):
    output = []
    counts = {}
    unique = set()
    for split in ('train', 'val', 'test'):
        folder = dataset / 'splits' / split
        if not folder.is_dir() and split == 'val':
            folder = dataset / 'splits' / 'validation'
        if not folder.is_dir():
            raise RuntimeError(f'Missing split: {folder}')
        episodes = sorted(folder.rglob('*.json'))
        counts[split] = len(episodes)
        for episode in episodes:
            obj = json.loads(episode.read_text())
            num_frames = int(obj['num_frames'])
            if num_frames <= 0:
                raise ValueError(f'Nonpositive num_frames: {episode}')
            for camera in CAMERAS:
                infos = obj['videos'][camera]
                if isinstance(infos, dict):
                    infos = [infos]
                if len(infos) != 1:
                    raise ValueError(f'Expected one video segment: {episode} {camera}')
                info = infos[0]
                if int(info.get('start', 0)) != 0:
                    raise ValueError(f'Nonzero video start not yet supported: {episode} {camera}')
                literal_source = str(info['path'])
                if literal_source in unique:
                    raise ValueError(f'Duplicate source video across episode-camera map: {literal_source}')
                unique.add(literal_source)
                # The literal source path remains the lookup key even after relocation.
                source = source_on_this_host(literal_source, remap_old, remap_new)
                rel_ep = episode.relative_to(dataset).with_suffix('')
                output.append({
                    'split': split, 'episode': str(episode.relative_to(dataset)),
                    'camera': camera, 'original_source_path': literal_source,
                    'source_host_path': str(source), 'frames': num_frames,
                    'crop_bbox': info.get('crop_bbox'),
                    'archive': str(Path('archives') / rel_ep / (camera + '.zip')),
                })
    if counts != {'train': 255, 'val': 34, 'test': 34}:
        raise ValueError(f'Frozen split counts changed: {counts}')
    if len(output) != 646 or sum(x['frames'] for x in output) != 723800:
        raise ValueError('Frozen media population changed')
    return output

def encode_one(entry, output_root, ffmpeg_threads, minimum_free):
    archive = output_root / entry['archive']
    sidecar = archive.with_suffix('.metadata.json')
    if archive.exists() and sidecar.exists():
        meta = json.loads(sidecar.read_text())
        if (meta.get('schema') == SCHEMA
            and meta.get('frames') == entry['frames']
            and meta.get('original_source_path') == entry['original_source_path']
            and meta.get('archive_bytes') == archive.stat().st_size):
            return 'SKIP_VERIFIED_METADATA', meta.get('archive_bytes', 0), 0
        raise RuntimeError(f'Existing archive/metadata mismatch; not overwriting: {archive}')
    if archive.exists() or sidecar.exists():
        raise RuntimeError(f'Existing partial completion; not overwriting: {archive}')
    free = os.statvfs(output_root)
    if (free.f_bavail * free.f_frsize) < minimum_free * (1024 ** 3):
        raise RuntimeError(f'Output filesystem has <{minimum_free} GiB available')
    archive.parent.mkdir(parents=True, exist_ok=True)
    source = Path(entry['source_host_path'])
    w, h = ffprobe_dimensions(source)
    frame_bytes = 3 * w * h
    bbox = entry['crop_bbox']
    cmd = ['ffmpeg', '-nostdin', '-hide_banner', '-loglevel', 'error',
           '-threads', str(ffmpeg_threads), '-i', str(source),
           '-map', '0:v:0', '-an', '-sn', '-dn', '-vsync', '0',
           '-f', 'rawvideo', '-pix_fmt', 'rgb24', 'pipe:1']
    tmp = archive.with_name(archive.name + f'.partial.{os.getpid()}')
    start = time.monotonic()
    proc = None
    try:
        with tempfile.TemporaryFile(mode='w+b') as errors:
            proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=errors)
            with zipfile.ZipFile(tmp, mode='x', compression=zipfile.ZIP_STORED,
                                 allowZip64=True) as zipped:
                for idx in range(entry['frames']):
                    rgb = proc.stdout.read(frame_bytes)
                    if len(rgb) != frame_bytes:
                        raise RuntimeError(f'Decoded only {idx}/{entry["frames"]} frames: {source}')
                    image = Image.frombytes('RGB', (w, h), rgb)
                    if bbox is not None:
                        top, left, crop_h, crop_w = (int(v) for v in bbox)
                        image = image.crop((left, top, left + crop_w, top + crop_h))
                    image = resized_like_xr1(image)
                    stream = io.BytesIO()
                    image.save(stream, 'PNG')
                    zipped.writestr(f'frame_{idx:09d}.png', stream.getvalue())
                extra = proc.stdout.read(1)
                if extra:
                    raise RuntimeError(f'More decoded frames than JSON declared: {source}')
            status = proc.wait()
            if status:
                errors.seek(0)
                raise RuntimeError(f'FFmpeg failed code={status}: {errors.read()[-2500:]}')
        # CRC-verify the actual archive on disk before atomic publish.
        with zipfile.ZipFile(tmp) as zipped:
            if len(zipped.namelist()) != entry['frames'] or zipped.testzip() is not None:
                raise RuntimeError(f'Archive integrity failed: {tmp}')
        metadata = {
            'schema': SCHEMA, 'frames': entry['frames'],
            'original_source_path': entry['original_source_path'],
            'source_host_path': entry['source_host_path'],
            'source_bytes': source.stat().st_size,
            'archive_bytes': tmp.stat().st_size,
            'archive_sha256': sha256(tmp), 'crop_bbox': bbox,
            'width': w, 'height': h,
            'ffmpeg_threads': ffmpeg_threads,
            'processing': 'ffmpeg rgb24 -> XR1 deterministic PIL crop/resize -> PNG in ZIP_STORED',
        }
        os.replace(tmp, archive)
        temp_meta = sidecar.with_name(sidecar.name + '.partial')
        temp_meta.write_text(json.dumps(metadata, indent=2) + '\n')
        os.replace(temp_meta, sidecar)
        return 'CREATED_AND_CRC_VERIFIED', archive.stat().st_size, time.monotonic()-start
    except BaseException:
        if proc is not None and proc.poll() is None:
            proc.kill()
            proc.wait()
        tmp.unlink(missing_ok=True)
        raise

def compare_to_existing_ffmpeg_cache(entry, output_root, reference_root):
    """Compare decoded/resized ZIP images to previously accepted FFmpeg PNGs."""
    key = hashlib.sha256(entry['original_source_path'].encode('utf-8')).hexdigest()[:24]
    old_folder = reference_root / 'videos' / key
    available = sorted(old_folder.glob('frame_*.png')) if old_folder.is_dir() else []
    if not available:
        raise RuntimeError(f'No reference PNGs for selected source: {old_folder}')
    chosen = available[:4] + available[-4:]
    unique = list(dict.fromkeys(chosen))
    with zipfile.ZipFile(output_root / entry['archive']) as archive:
        for path in unique:
            with Image.open(path) as im:
                expected = im.convert('RGB')
                if entry['crop_bbox'] is not None:
                    top, left, ch, cw = map(int, entry['crop_bbox'])
                    expected = expected.crop((left, top, left + cw, top + ch))
                expected = resized_like_xr1(expected)
            with archive.open(path.name) as f:
                with Image.open(f) as img:
                    actual = img.convert('RGB')
            if expected.size != actual.size or ImageChops.difference(expected, actual).getbbox() is not None:
                raise RuntimeError(f'PIXEL_PARITY_FAIL: {entry["archive"]}:{path.name}')
    return len(unique)

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset', type=Path, required=True)
    p.add_argument('--output-root', type=Path, required=True)
    p.add_argument('--policy', type=Path, required=True)
    p.add_argument('--pilot-videos', type=int, default=2, help='0 for all 646 videos')
    p.add_argument('--ffmpeg-threads', type=int, default=2)
    p.add_argument('--minimum-free-gib', type=int, default=20)
    p.add_argument('--reference-cache-root', type=Path, help='Existing accepted full-resolution FFmpeg PNG cache for pixel-parity checking')
    p.add_argument('--remap-source-prefix-old')
    p.add_argument('--remap-source-prefix-new')
    args = p.parse_args()
    if (args.remap_source_prefix_old is None) != (args.remap_source_prefix_new is None):
        p.error('Use both remap flags, or neither')
    if args.pilot_videos < 0 or args.pilot_videos > 646:
        p.error('pilot-videos must be in [0,646]')
    if args.ffmpeg_threads < 1 or args.ffmpeg_threads > 4:
        p.error('ffmpeg-threads must be 1..4')
    if sha256(args.policy) != '21d8ffbdbfec2f9833339faaf0c5818c85cd5a135dcf474fe2d3f76cda1cf34d':
        raise RuntimeError('Frozen Language V2 checksum mismatch')
    root = args.output_root.resolve()
    dataset = args.dataset.resolve()
    if root == dataset or dataset in root.parents or root in dataset.parents:
        raise RuntimeError('Output must be outside the immutable V2 dataset')
    root.mkdir(parents=True, exist_ok=True)
    entries = manifest_entries(dataset, args.remap_source_prefix_old, args.remap_source_prefix_new)
    # Stable manifest deliberately excludes CLAIX/workstation host-specific paths.
    portable = [{k:v for k,v in e.items() if k != 'source_host_path'} for e in entries]
    payload = ''.join(json.dumps(e, sort_keys=True) + '\n' for e in portable)
    manifest = root / 'media_manifest.jsonl'
    if manifest.exists() and manifest.read_text() != payload:
        raise RuntimeError('Existing manifest mismatch. Use a new output root.')
    if not manifest.exists():
        manifest.write_text(payload)
    selected = entries if args.pilot_videos == 0 else entries[:args.pilot_videos]
    print(f'SCHEMA={SCHEMA} TOTAL_VIDEOS={len(entries)} SELECTED_VIDEOS={len(selected)}', flush=True)
    for i, entry in enumerate(selected, 1):
        status, size, elapsed = encode_one(entry, root, args.ffmpeg_threads, args.minimum_free_gib)
        parity = 'NOT_REQUESTED'
        if args.reference_cache_root is not None:
            matched = compare_to_existing_ffmpeg_cache(entry, root, args.reference_cache_root)
            parity = f'PASS_{matched}_FRAMES'
        print(f'VIDEO={i}/{len(selected)} STATUS={status} CAMERA={entry["camera"]} '
              f'FRAMES={entry["frames"]} GiB={size/2**30:.3f} SEC={elapsed:.1f} PIXEL_PARITY={parity}', flush=True)
    print('OFFLINE_ZIP_BUILD=PASS', flush=True)
    if args.pilot_videos:
        print('DATASET_STATUS=PILOT_ONLY_NOT_READY_FOR_TRAINING', flush=True)

if __name__ == '__main__':
    main()
