#!/usr/bin/env python3
"""Build a versioned, opt-in offline RGB PNG cache for sampled XR1 Doosan frames.

This mirrors the Oct-2026 DoosanJsonDataset._samples() selection policy:
  max_samples = max_steps * batch_size * WORLD_SIZE
  rng = random.Random(sampler_seed)
  shuffle(range(total_population))
  take samples from that shuffled population (without replacement per cycle).

It does not edit JSON, videos, model files, or training configurations.
Run --pilot-videos 4 before any full cache generation. Always review parity
of FFmpeg frame indices with the current loader before enabling cached reads.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import random
import shutil
import subprocess
import sys
import tarfile
import tempfile
import time
import traceback

from PIL import Image

EXPECTED_ROOT = Path('/home/ktt_rc/robotics_thesis/xr1_lab/datasets/doosan_xr1_force_free_language_v2')
EXPECTED_POLICY_SHA = '21d8ffbdbfec2f9833339faaf0c5818c85cd5a135dcf474fe2d3f76cda1cf34d'
POLICY = Path('/home/ktt_rc/robotics_thesis/xr1_lab/configs/doosan_assembly_language_v2.yaml')
LOG_ROOT = Path('/home/ktt_rc/robotics_thesis/xr1_lab/logs')
CAMERAS = ('external_camera_2', 'tcp_camera')


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def video_key(path: Path) -> str:
    return hashlib.sha256(os.fspath(path).encode('utf-8')).hexdigest()[:24]


def selected_requests(dataset: Path, steps: int, batch: int, world: int, seed: int):
    paths = sorted((dataset / 'splits/train').rglob('*.json'))
    if len(paths) != 255:
        raise RuntimeError(f'Expected 255 training JSON files; found {len(paths)}')
    population = []
    details = {}
    for p in paths:
        obj = json.loads(p.read_text(encoding='utf-8'))
        n = int(obj['num_frames'])
        if n < 1:
            raise RuntimeError(f'Invalid frame count {n}: {p}')
        population.extend((p, i) for i in range(n))
        details[p] = obj
    k = steps * batch * world
    if k < 1:
        raise RuntimeError('At least one training sample is required')
    rng = random.Random(seed)
    chosen = []
    remaining = k
    while remaining:
        indices = list(range(len(population)))
        rng.shuffle(indices)
        take = min(remaining, len(population))
        chosen.extend(population[i] for i in indices[:take])
        remaining -= take
    targets = defaultdict(set)
    counts = defaultdict(int)
    for episode, frame in chosen:
        obj = details[episode]
        general = obj['instruction']['general']
        if len(general) != 1:
            raise RuntimeError(f'Unexpected instruction.general length: {episode}')
        images = general[0]['images']
        if images != [f'videos.{CAMERAS[0]}', f'videos.{CAMERAS[1]}']:
            raise RuntimeError(f'Unexpected camera order: {episode}: {images}')
        for camera in CAMERAS:
            for info in obj['videos'][camera]:
                src = Path(info['path'])
                if not src.is_file():
                    raise FileNotFoundError(src)
                target_idx = frame + int(info.get('start', 0))
                if target_idx < 0:
                    raise RuntimeError(f'Negative frame index: {src} {target_idx}')
                targets[src].add(target_idx)
                counts[camera] += 1
    return targets, len(population), len(chosen), dict(counts)


def extract_one(path: Path, indices: list[int], cache: Path, timeout: int):
    """Decode selected RGB frame indices sequentially using ONE ffmpeg process."""
    key = video_key(path)
    folder = cache / 'videos' / key
    folder.mkdir(parents=True, exist_ok=True)
    st = path.stat()
    existing = folder / 'metadata.json'
    if existing.is_file():
        old = json.loads(existing.read_text())
        if old.get('source_path') != os.fspath(path) or old.get('source_size') != st.st_size or old.get('source_mtime_ns') != st.st_mtime_ns:
            raise RuntimeError(f'Cache provenance mismatch; use new cache root: {folder}')
    missing = [idx for idx in indices if not (folder / f'frame_{idx:09d}.png').is_file()]
    if not missing:
        return {'source': os.fspath(path), 'key': key, 'wanted': len(indices), 'new': 0, 'bytes_new': 0, 'seconds': 0}
    t = time.monotonic()
    expression = '+'.join(f'eq(n\\,{idx})' for idx in missing)
    with tempfile.TemporaryDirectory(prefix='xr1_ffmpeg_', dir=cache) as tmp:
        tmp = Path(tmp)
        pattern = tmp / 'selected_%06d.png'
        cmd = [
            'ffmpeg', '-hide_banner', '-loglevel', 'error', '-nostdin',
            '-threads', '2', '-i', os.fspath(path), '-map', '0:v:0', '-an', '-sn', '-dn',
            '-vf', f'select={expression}', '-fps_mode', 'passthrough',
            '-pix_fmt', 'rgb24', '-start_number', '0', '-f', 'image2', os.fspath(pattern),
        ]
        done = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, check=False)
        if done.returncode:
            raise RuntimeError(f'ffmpeg exit={done.returncode} source={path}\n{done.stderr[-1500:]}')
        outputs = sorted(tmp.glob('selected_*.png'))
        if len(outputs) != len(missing):
            raise RuntimeError(f'Frame count mismatch: expected={len(missing)} extracted={len(outputs)} source={path}')
        bytes_new = 0
        for idx, candidate in zip(missing, outputs):
            with Image.open(candidate) as img:
                if img.mode != 'RGB' or img.size[0] < 1 or img.size[1] < 1:
                    raise RuntimeError(f'Not RGB frame idx={idx} path={path} actual={img.mode} {img.size}')
                img.verify()
            bytes_new += candidate.stat().st_size
            os.replace(candidate, folder / f'frame_{idx:09d}.png')
        new_all = sorted(set(indices) | set(int(p.stem.split('_')[-1]) for p in folder.glob('frame_*.png')))
        meta = {
            'source_path': os.fspath(path), 'source_size': st.st_size,
            'source_mtime_ns': st.st_mtime_ns, 'video_key': key,
            'format': 'PNG', 'indices': new_all,
            'generated_utc': datetime.utcnow().isoformat() + 'Z',
            'conversion': 'ffmpeg -vf select=eq(n\\,index) -fps_mode passthrough -pix_fmt rgb24',
        }
        tmpmeta = folder / 'metadata.json.tmp'
        tmpmeta.write_text(json.dumps(meta, indent=2), encoding='utf-8')
        os.replace(tmpmeta, folder / 'metadata.json')
    return {'source': os.fspath(path), 'key': key, 'wanted': len(indices), 'new': len(missing), 'bytes_new': bytes_new, 'seconds': round(time.monotonic() - t, 1)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, default=EXPECTED_ROOT)
    parser.add_argument('--cache-root', type=Path, default=Path('/mnt/robotics_ssd/ktt_rc_robotics_thesis/xr1_offline_frame_cache_v1'))
    parser.add_argument('--max-steps', type=int, default=2000)
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--world-size', type=int, default=2)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--pilot-videos', type=int, default=4, help='4 for safe pilot; 0 for all selected videos')
    parser.add_argument('--timeout-per-video', type=int, default=300)
    args = parser.parse_args()
    stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    out = LOG_ROOT / f'xr1_offline_cache_pilot_{stamp}_{os.getpid()}'
    out.mkdir(parents=True, exist_ok=False)
    report_lines = []
    detail = []
    error = None

    def log(s):
        print(s, flush=True)
        report_lines.append(s)
        (out / 'report.txt').write_text('\n'.join(report_lines) + '\n', encoding='utf-8')

    try:
        log('=== XR1 OFFLINE RGB CACHE — ADDITIVE PILOT ===')
        log(f'SCRIPT={Path(__file__).resolve()}')
        log(f'DATASET={args.dataset}')
        log(f'CACHE_ROOT={args.cache_root}')
        log(f'PILOT_VIDEO_LIMIT={args.pilot_videos}')
        if args.dataset.resolve() != EXPECTED_ROOT.resolve():
            raise RuntimeError('Dataset path differs from frozen V2 dataset; refusing')
        if sha(POLICY) != EXPECTED_POLICY_SHA:
            raise RuntimeError('V2 language policy checksum changed; refusing')
        if shutil.which('ffmpeg') is None or shutil.which('ffprobe') is None:
            raise RuntimeError('ffmpeg and ffprobe must already be installed')
        if args.pilot_videos < 0:
            raise RuntimeError('pilot-videos must be >=0')
        if args.cache_root.resolve() == args.dataset.resolve() or args.dataset.resolve() in args.cache_root.resolve().parents:
            raise RuntimeError('Cache path cannot be inside dataset')
        if '/mnt/robotics_ssd/' not in os.fspath(args.cache_root.resolve()) + '/':
            raise RuntimeError('Cache must be placed on Robotics SSD')
        free = shutil.disk_usage(args.cache_root.parent if args.cache_root.parent.exists() else Path('/mnt/robotics_ssd')).free
        if free < 10 * 1024**3:
            raise RuntimeError(f'Insufficient SSD headroom ({free/1024**3:.1f} GiB)')
        t = time.monotonic()
        targets, population, sample_count, counts = selected_requests(args.dataset, args.max_steps, args.batch_size, args.world_size, args.seed)
        log(f'TRAIN_POPULATION={population}')
        log(f'SELECTED_TRAINING_SAMPLES={sample_count}')
        log(f'CAMERA_REFERENCES={counts}')
        log(f'UNIQUE_VIDEO_FILES={len(targets)}')
        log(f'UNIQUE_SELECTED_VIDEO_FRAMES={sum(len(v) for v in targets.values())}')
        sorted_videos = sorted(targets, key=lambda p: (-len(targets[p]), str(p)))
        if args.pilot_videos:
            sorted_videos = sorted_videos[:args.pilot_videos]
        args.cache_root.mkdir(parents=True, exist_ok=True)
        manifest = {
            'schema_version': 1,
            'policy_sha256': EXPECTED_POLICY_SHA,
            'dataset_root': os.fspath(args.dataset.resolve()),
            'sampling': {'max_steps': args.max_steps, 'batch_size': args.batch_size, 'world_size': args.world_size, 'seed': args.seed},
            'population_frames': population,
            'selected_samples': sample_count,
            'unique_video_files': len(targets),
            'unique_selected_video_frames': sum(len(v) for v in targets.values()),
            'planned_pilot_videos': len(sorted_videos),
            'video_preview': [{'path': str(p), 'indices': len(targets[p])} for p in sorted_videos[:30]],
        }
        (out / 'selection_manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
        log(f'SELECTION_MANIFEST={out / "selection_manifest.json"}')
        for count, source in enumerate(sorted_videos, 1):
            items = sorted(targets[source])
            result = extract_one(source, items, args.cache_root, args.timeout_per_video)
            detail.append(result)
            log(f'VIDEO_PROGRESS={count}/{len(sorted_videos)} FRAMES={len(items)} NEW={result["new"]} TIME_S={result["seconds"]}')
            if shutil.disk_usage(args.cache_root).free < 8 * 1024**3:
                raise RuntimeError('SSD headroom below 8 GiB; stopping')
        new_bytes = sum(entry['bytes_new'] for entry in detail)
        log(f'OFFLINE_FRAMES_NEW={sum(x["new"] for x in detail)}')
        log(f'OFFLINE_BYTES_NEW={new_bytes}')
        log(f'ELAPSED_S={round(time.monotonic() - t, 1)}')
        log('OFFLINE_CACHE_PILOT=PASS')
        log('DATASETS_MODIFIED=NO')
        log('TRAINING_STARTED=NO')
        log('DECODER_SWITCHED=NO')
    except Exception:
        error = traceback.format_exc()
        (out / 'exception.txt').write_text(error, encoding='utf-8')
        log('OFFLINE_CACHE_PILOT=FAIL')
        log(error.splitlines()[-1])
    finally:
        (out / 'video_details.json').write_text(json.dumps(detail, indent=2), encoding='utf-8')
        archive = Path(str(out) + '_evidence.tar.gz')
        with tarfile.open(archive, 'w:gz') as tar:
            tar.add(out, arcname=out.name)
        log(f'UPLOAD_FILE={archive}')
    return 1 if error else 0


if __name__ == '__main__':
    sys.exit(main())
