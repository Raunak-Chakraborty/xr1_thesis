import importlib.util
from pathlib import Path
import sys
import unittest
root=Path.cwd()
sys.path[:0]=[str(root/'xr1'),str(root/'xr1/tests')]
spec=importlib.util.spec_from_file_location('harness',root/'xr1/tests/run_claix_cpu_tests.py')
harness=importlib.util.module_from_spec(spec);spec.loader.exec_module(harness)
harness.bootstrap()
suite=unittest.TestSuite()
loader=unittest.TestLoader()
for path in sorted((root/'xr1/tests').glob('test_*.py')):
    if path.name=='test_claix_final_release.py' or not path.name.startswith(('test_doosan_', 'test_claix_')):
        continue
    suite.addTests(loader.loadTestsFromName(path.stem))
import test_training_logic
for name in sorted(dir(test_training_logic)):
    if name.startswith('test_'): suite.addTest(unittest.FunctionTestCase(getattr(test_training_logic,name)))
assert suite.countTestCases()==98,suite.countTestCases()
result=unittest.TextTestRunner(verbosity=2).run(suite)
raise SystemExit(not result.wasSuccessful())
