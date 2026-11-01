import ast
import hashlib
import json
from pathlib import Path
import subprocess
root=Path.cwd()
files=[*root.joinpath('xr1').rglob('*.py'),*root.joinpath('scripts').glob('claix_b24*.py')]
for path in files: ast.parse(path.read_text(),filename=str(path))
for path in [*root.joinpath('scripts').glob('claix_b24*.sbatch'),root/'scripts/claix_b24_environment.sh']:
    subprocess.run(['bash','-n',str(path)],check=True)
base='4339a0fcd7a9c18c81356d371360bbf629656c0c'
# All scientific/model/data/scheduler files are unchanged, including every-step logging.
protected=['xr1/configs','xr1/mibot/models','xr1/mibot/data','xr1/mibot/utils/cosine_warmup.py',
           'xr1/mibot/utils/doosan_replay.py','xr1/mibot/utils/doosan_compact_fp32.py',
           'xr1/mibot/utils/claix_contract.py','xr1/tools/train_doosan_compact.py']
assert not subprocess.check_output(['git','diff',base,'--',*protected],text=True)
print(json.dumps(dict(python_syntax_files=len(files),bash_syntax='PASS',scientific_source_unchanged=True),indent=2))
