#!/usr/bin/env python3
"""Read-only persistent storage protection; quota output is optional diagnostics."""
import argparse
from datetime import datetime, timezone
import json
import os
import subprocess

from claix_b24_filesystem import inspect_directory

GIB = 1024**3
MIN_FREE_BYTES = 30*GIB
MIN_FREE_INODES = 10000


def query(cmd):
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
        return dict(command=cmd, returncode=result.returncode, stdout=result.stdout, stderr=result.stderr)
    except (OSError, subprocess.TimeoutExpired) as error:
        return dict(command=cmd, error=str(error))


def evaluate(run_root, filesystem_source, project_id=None, *, stage='submission', quota_diagnostics=False):
    if stage not in ('submission', 'allocation'):
        raise ValueError('Unknown storage preflight stage')
    if not filesystem_source:
        raise ValueError('Approved persistent filesystem source is required')
    location = inspect_directory(run_root, allow_missing=True, expected_source=filesystem_source)
    root, parent, fs = location['resolved_path'], location['existing_parent'], location['filesystem']
    stat = os.statvfs(parent)
    free = stat.f_bavail * stat.f_frsize
    budget = dict(compact_bytes=2*GIB, retained_payloads=9, new_publications=1,
                  logs_and_temporary_reserve_bytes=4*GIB, required_free_bytes=MIN_FREE_BYTES,
                  required_free_inodes=MIN_FREE_INODES)
    checks = dict(persistent_filesystem=True, filesystem_headroom=free >= budget['required_free_bytes'],
                  inodes=stat.f_favail >= budget['required_free_inodes'])
    quotas = []
    if quota_diagnostics:
        # Preserve site output verbatim, including CLAIX bquota/blimit and pid
        # labels. Diagnostics never determine storage approval.
        quotas.append(query(['lfs', 'quota', '-u', str(os.getuid()), fs['target']]))
        if project_id is not None:
            quotas.append(query(['lfs', 'quota', '-p', str(project_id), fs['target']]))
        quotas.append(query(['r_quota']))
    reasons = [key+' failed' for key, passed in checks.items() if not passed]
    return dict(schema='claix-b24-storage-v1', status='PASS' if all(checks.values()) else 'FAIL',
                stage=stage, timestamp=datetime.now(timezone.utc).isoformat(), run_root=root,
                existing_parent=parent, filesystem=fs, available_bytes=free,
                available_inodes=stat.f_favail, budget=budget, checks=checks,
                project_id=project_id, uid=os.getuid(), quota_queries=quotas, abort_reasons=reasons)


def allocation_recheck(record):
    """Called by the production launcher after release validation, before model loading."""
    from claix_b24_release import validate_storage_binding
    validate_storage_binding(record)
    binding = record['storage_preflight']
    return evaluate(record['plan']['run_root'], binding['filesystem_source'],
                    binding.get('project_id'), stage='allocation')


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-root', required=True)
    p.add_argument('--filesystem-source', required=True, help='Operator-confirmed persistent findmnt SOURCE')
    p.add_argument('--project-id', type=int, help='Operator-confirmed Lustre project ID, if applicable')
    p.add_argument('--quota-diagnostics', action='store_true', help='Record optional lfs quota/r_quota output')
    p.add_argument('--stage', choices=('submission', 'allocation'), default='submission')
    args = p.parse_args(argv)
    try:
        result = evaluate(args.run_root, args.filesystem_source, args.project_id,
                          stage=args.stage, quota_diagnostics=args.quota_diagnostics)
    except Exception as error:
        result = dict(status='UNVERIFIED', stage=args.stage, reason=str(error), abort_reasons=[str(error)])
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result['status'] == 'PASS' else 1


if __name__ == '__main__':
    raise SystemExit(main())
