import copy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
import zipfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import digest, sha, source_check, write_new, module
from recover import copy_exact, telemetry_summary
from stage_cache import seal, load_index, stage, verify, check_stage
from observe import observe


class ControlTests(unittest.TestCase):
    def test_source_fingerprint_is_authoritative(self):
        source_check()

    def test_source_fingerprint_detects_added_training_test_but_separates_tools(self):
        launch = module('claix_b24_launch')
        with tempfile.TemporaryDirectory() as folder:
            launch.ROOT = Path(folder)
            (launch.ROOT/'xr1/configs').mkdir(parents=True)
            (launch.ROOT/'scripts').mkdir()
            (launch.ROOT/'xr1/a.py').write_text('original')
            original = launch.tree_digest()
            (launch.ROOT/'reliability').mkdir()
            (launch.ROOT/'reliability/tool.py').write_text('separately audited')
            self.assertEqual(launch.tree_digest(),original)
            (launch.ROOT/'xr1/new_test.py').write_text('changes compatibility')
            self.assertNotEqual(launch.tree_digest(),original)

    def test_telemetry_tail_preserved_and_duplicate_rejected(self):
        rows = [dict(event='train_start', step=8750)]
        rows += [dict(event='step', step=s) for s in range(8751,8759)]
        rows += [dict(event='fatal', step=8758, error='synthetic failure')]
        raw = b''.join(json.dumps(row).encode()+b'\n' for row in rows)
        with tempfile.TemporaryDirectory() as folder:
            parent = Path(folder)/'telemetry.jsonl'
            parent.write_bytes(raw)
            copy_exact(parent, Path(folder)/'audit.jsonl')
            self.assertEqual(telemetry_summary(raw)['last_completed'],8758)
            self.assertEqual(parent.read_bytes(),raw)
            with self.assertRaises(FileExistsError):
                copy_exact(parent, Path(folder)/'audit.jsonl')
        for bad in (raw[:-1],raw+json.dumps(dict(event='step',step=8758)).encode()+b'\n'):
            with self.assertRaises(RuntimeError):
                telemetry_summary(bad)

    def test_passive_observer_pre_timeout_capture_and_pid_exit(self):
        with tempfile.TemporaryDirectory() as folder:
            child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(1.5)'])
            try:
                output = Path(folder)/'observe.json'
                report = observe(child.pid, output, interval=.1, duration=1., stall_seconds=.3, samples=4)
                self.assertLessEqual(len(report['snapshots']),4)
                self.assertTrue(Path(str(output)+'.inactivity.json').is_file())
                self.assertEqual(report['target_identity'][0],child.pid)
                self.assertIsNone(child.poll())  # observer did not signal or stop it
            finally:
                child.wait(timeout=3)


class StagingTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.cache = self.root/'source'
        (self.cache/'archives').mkdir(parents=True)
        rows = []
        for index in range(646):
            name = f'archives/{index}.zip'
            path = self.cache/name
            with zipfile.ZipFile(path,'w') as zipped:
                zipped.writestr('frame_000000000.png', b'fixture: hash integrity, not PNG decode')
            row = dict(archive=name, original_source_path=f'/original/{index}.mp4', frames=1,
                       camera='tcp_camera', crop_bbox=None)
            write_new(path.with_suffix('.metadata.json'),
                      dict(schema='doosan-xr1-rgb-png-zip-v1', archive_bytes=path.stat().st_size,
                           archive_sha256=sha(path), **{k:row[k] for k in ('original_source_path','frames','crop_bbox')}))
            rows.append(row)
        (self.cache/'media_manifest.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in rows))
        self.manifest_sha = sha(self.cache/'media_manifest.jsonl')
        self.index = seal(self.cache,self.manifest_sha)
        self.index_path = self.root/'index.json'
        write_new(self.index_path,self.index)

    def test_stage_roundtrip_identity_ready_and_no_clobber(self):
        self.assertEqual(load_index(self.index_path,sha(self.index_path)),self.index)
        target = self.root/'staged'
        stage(self.cache,target,self.index,reserve_bytes=0)
        check_stage(target,self.index)
        self.assertEqual((target/'media_manifest.jsonl').read_bytes(),(self.cache/'media_manifest.jsonl').read_bytes())
        with self.assertRaises(RuntimeError):
            stage(self.cache,target,self.index)
        (target/'archives/1.zip').chmod(0o644)
        (target/'archives/1.zip').write_bytes(b'corrupt')
        with self.assertRaises(RuntimeError):
            check_stage(target,self.index)

    def test_corruption_missing_files_and_sidecar_rejected(self):
        path = self.cache/'archives/2.zip'
        path.write_bytes(b'corrupt')
        with self.assertRaises(RuntimeError):
            seal(self.cache,self.manifest_sha)
        target = self.root/'partial'
        with self.assertRaises(RuntimeError):
            stage(self.cache,target,self.index,reserve_bytes=0)
        self.assertFalse((target/'STAGING_READY.json').exists())
        path.unlink()
        with self.assertRaises((OSError,RuntimeError)):
            verify(self.cache,self.index)

    def test_manifest_seal_and_space_fail_closed(self):
        with self.assertRaises(RuntimeError):
            seal(self.cache,'a'*64)
        with self.assertRaises(RuntimeError):
            load_index(self.index_path,'a'*64)
        bad = copy.deepcopy(self.index)
        bad['files']['../escape'] = bad['files'].pop('archives/1.zip')
        write_new(self.root/'bad.json',bad)
        with self.assertRaises(RuntimeError):
            load_index(self.root/'bad.json',sha(self.root/'bad.json'))
        with patch('stage_cache.shutil.disk_usage',return_value=type('Usage',(),{'free':1})()):
            with self.assertRaises(RuntimeError):
                stage(self.cache,self.root/'no-space',self.index)
        self.assertFalse((self.root/'no-space').exists())
        with self.assertRaises(FileNotFoundError):
            stage(self.cache,self.root/'report-failure',self.index,reserve_bytes=0,
                  report_path=self.root/'missing-parent/report.json')
        self.assertFalse((self.root/'report-failure/STAGING_READY.json').exists())

    def test_symlink_and_incomplete_ready_rejected(self):
        path = self.cache/'archives/0.zip'
        path.unlink()
        path.symlink_to(self.cache/'archives/1.zip')
        with self.assertRaises(RuntimeError):
            verify(self.cache,self.index)
        with self.assertRaises(OSError):
            check_stage(self.cache,self.index)


if __name__ == '__main__':
    unittest.main()
