import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from stage_cache import allocation_scratch


class ScratchPolicyTests(unittest.TestCase):
    def test_local_beeond_tmpfs_and_nested_mount(self):
        with tempfile.TemporaryDirectory() as root:
            def response(kind):
                return json.dumps({'filesystems':[dict(source='device', target=root, fstype=kind)]})
            for kind in ('ext4', 'xfs'):
                with patch('stage_cache.subprocess.check_output', return_value=response(kind)):
                    result = allocation_scratch(Path(root)/'staged', scratch_root=root)
                    self.assertEqual(result['storage_kind'], 'local_disk')
                    with self.assertRaisesRegex(RuntimeError, 'BeeOND'):
                        allocation_scratch(Path(root)/'staged', scratch_root=root, beeond=True)
            with patch('stage_cache.subprocess.check_output', return_value=response('beegfs')):
                with self.assertRaises(RuntimeError):
                    allocation_scratch(Path(root)/'staged', scratch_root=root)
                self.assertTrue(allocation_scratch(Path(root)/'staged', scratch_root=root, beeond=True)['beeond_explicit'])
            with patch('stage_cache.subprocess.check_output', return_value=response('tmpfs')):
                with self.assertRaises(RuntimeError):
                    allocation_scratch(Path(root)/'staged', scratch_root=root)
                self.assertEqual(allocation_scratch(Path(root)/'staged', scratch_root=root, allow_tmpfs=True)['storage_kind'], 'ram')
            with patch('stage_cache.subprocess.check_output', side_effect=[response('ext4'), response('lustre')]):
                with self.assertRaisesRegex(RuntimeError, 'interior'):
                    allocation_scratch(Path(root)/'staged', scratch_root=root)


if __name__ == '__main__':
    unittest.main()
