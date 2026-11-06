"""Read-only source/regression/syntax checks, runnable with system Python."""
import ast
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[3]
BASE = '1e528af3ef370f2aa2c46d4e418f185dcb48be32'
changed = {'scripts/claix_b24_launch.py', 'xr1/mibot/utils/claix_experiment.py',
           'xr1/tools/train_doosan_compact.py'}
head = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
assert head == BASE, head
actual = set(subprocess.check_output(['git', 'diff', '--name-only', BASE], cwd=ROOT, text=True).splitlines())
assert actual == changed, actual
tracked = subprocess.check_output(['git', 'ls-tree', '-r', '--name-only', BASE], cwd=ROOT, text=True).splitlines()
protected = [p for p in tracked if p not in changed and
             (p.startswith('xr1/configs/') or p.startswith('xr1/mibot/utils/') or
              p.startswith('xr1/tests/test_claix') or p.startswith('scripts/claix_b24') or
              p == 'docs/claix_b24/CHECKPOINT500_TASK04.md')]
for name in protected:
    assert (ROOT/name).read_bytes() == subprocess.check_output(['git', 'show', BASE+':'+name], cwd=ROOT), name
new_python = ['scripts/claix_b24_checkpoint_two_way.py', 'xr1/mibot/utils/claix_async_checkpoint.py',
              'xr1/tests/run_claix_async_cpu_tests.py', 'xr1/tests/test_claix_async_checkpoint.py',
              'xr1/tests/test_claix_async_framework.py', 'xr1/tests/test_claix_two_way.py']
for name in sorted(changed | set(new_python)):
    ast.parse((ROOT/name).read_text(), filename=name)
wrappers = sorted((ROOT/'scripts').glob('claix_b24*.sbatch')) + [ROOT/'scripts/claix_b24_environment.sh']
for path in wrappers:
    subprocess.run(['bash', '-n', str(path)], check=True)
subprocess.run(['git', 'diff', '--check'], cwd=ROOT, check=True)
print(f'HEAD={head}')
print(f'BASELINE_PROTECTION=PASS ({len(protected)} tracked files byte-identical)')
print(f'PYTHON_AST=PASS ({len(changed)+len(new_python)} changed/new source files)')
print(f'BASH_SYNTAX=PASS ({len(wrappers)} wrappers/environment scripts)')
print('GIT_DIFF_CHECK=PASS')
