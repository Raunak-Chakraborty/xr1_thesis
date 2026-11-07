"""Task04C reproducible local syntax/source-protection checks. No framework claim."""
import ast
import hashlib
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[3]
START = 'c3b2c6895fe2d3bfd19c3d301ef5b49a02841686'
allowed = {
    'scripts/claix_b24_launch.py', 'scripts/claix_b24_checkpoint_two_way.py',
    'scripts/claix_b24_checkpoint_two_way.sbatch', 'xr1/tools/train_doosan_compact.py',
    'xr1/mibot/utils/claix_experiment.py', 'xr1/mibot/utils/claix_async_checkpoint.py',
    'xr1/tests/test_claix_async_framework.py', 'xr1/tests/test_claix_two_way.py',
    'xr1/tests/run_claix_async_cpu_tests.py',
}

def git(*args):
    return subprocess.check_output(['git', *args], cwd=ROOT)

subprocess.run(['git', 'merge-base', '--is-ancestor', START, 'HEAD'], cwd=ROOT, check=True)
changed = set(git('diff', '--name-only', START).decode().splitlines())
assert changed <= allowed, changed-allowed
tracked = git('ls-tree', '-r', '--name-only', START).decode().splitlines()
protected = [p for p in tracked if p not in allowed and
             (p.startswith(('xr1/configs/', 'xr1/mibot/utils/', 'xr1/tests/test_claix', 'scripts/claix_b24',
                            'docs/claix_b24/task04_evidence/', 'docs/claix_b24/task04b_evidence/'))
              or p in ('docs/claix_b24/CHECKPOINT500_TASK04.md', 'docs/claix_b24/CHECKPOINT_ASYNC_TASK04B.md'))]
for name in protected:
    assert (ROOT/name).read_bytes() == git('show', START+':'+name), name
python_files = sorted([*ROOT.joinpath('xr1').rglob('*.py'), *ROOT.joinpath('scripts').glob('claix_b24*.py')])
for path in python_files:
    ast.parse(path.read_text(), filename=str(path))
wrappers = sorted(ROOT.joinpath('scripts').glob('claix_b24*.sbatch')) + [ROOT/'scripts/claix_b24_environment.sh']
for path in wrappers:
    subprocess.run(['bash', '-n', str(path)], check=True)
subprocess.run(['git', 'diff', '--check'], cwd=ROOT, check=True)
print('START=' + START)
print('HEAD=' + git('rev-parse', 'HEAD').decode().strip())
print('BRANCH=' + git('branch', '--show-current').decode().strip())
print(f'PROTECTED_BYTES=PASS files={len(protected)} (science, sync runtime/I/O, historical wrappers, retained evidence)')
print(f'PYTHON_AST=PASS files={len(python_files)}')
print(f'BASH_SYNTAX=PASS files={len(wrappers)}')
print('GIT_DIFF_CHECK=PASS')
print('REAL_CLAIX_ACCEPTANCE=PENDING')
print('GPU_JOBS_SUBMITTED=0')
