#!/usr/bin/env python3
"""Read-only persistent storage protection; quota output is optional diagnostics."""
import argparse
from datetime import datetime, timezone
import json
import os
import re
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


def quota_headroom(result, identity, number, mount):
    """Parse numeric Lustre default (KiB) output; unknown/error output fails closed.

    Both CLAIX bquota/blimit and upstream quota/limit column labels have the
    same eight-column ordering. Wrapped filesystem names are supported.
    Soft limits are conservative ceilings even during a grace period.
    """
    output = result.get('stdout', '')
    if (result.get('returncode') != 0 or result.get('error')
            or not re.search(r'\(' + identity + r'\s+' + str(number) + r'\)', output)):
        raise RuntimeError('Cannot verify Lustre quota identity/query: ' + identity)
    lines = output.splitlines()
    header = next((i for i, line in enumerate(lines)
                   if line.split()[:2] == ['Filesystem', 'kbytes']), None)
    if header is None:
        raise RuntimeError('Unknown Lustre quota format')
    columns = lines[header].split()
    if columns not in (['Filesystem', 'kbytes', 'bquota', 'blimit', 'bgrace', 'files', 'iquota', 'ilimit', 'igrace'],
                       ['Filesystem', 'kbytes', 'quota', 'limit', 'grace', 'files', 'quota', 'limit', 'grace']):
        raise RuntimeError('Unknown Lustre quota columns')
    fields = ' '.join(lines[header+1:]).split()
    if len(fields) != 9 or fields[0] != mount:
        raise RuntimeError('Ambiguous Lustre quota filesystem/measurements')
    try:
        used, soft, hard, files, isoft, ihard = (int(fields[i].rstrip('*')) for i in (1, 2, 3, 5, 6, 7))
    except ValueError as error:
        raise RuntimeError('Non-numeric Lustre quota measurements') from error
    if min(used, soft, hard, files, isoft, ihard) < 0:
        raise RuntimeError('Negative Lustre quota measurements')
    byte_limits = [value*1024 for value in (soft, hard) if value]
    inode_limits = [value for value in (isoft, ihard) if value]
    remaining_bytes = min(byte_limits)-used*1024 if byte_limits else None
    remaining_inodes = min(inode_limits)-files if inode_limits else None
    return dict(identity=identity, id=number, used_bytes=used*1024, used_inodes=files,
                remaining_bytes=remaining_bytes, remaining_inodes=remaining_inodes,
                sufficient=(remaining_bytes is None or remaining_bytes >= MIN_FREE_BYTES)
                and (remaining_inodes is None or remaining_inodes >= MIN_FREE_INODES))


def experimental_evaluate(run_root, filesystem_source, project_id, *, stage='allocation'):
    """Independent live storage gate; no scientific release record or approvals.

    Unlike the unchanged production policy, experimental quota queries are
    mandatory. No attested/fabricated quota success or unparsed output fallback.
    """
    if type(project_id) is not int or project_id <= 0:
        raise RuntimeError('Experimental storage requires the actual positive Lustre project ID')
    result = evaluate(run_root, filesystem_source, project_id, stage=stage)
    fs = result['filesystem']
    if fs['fstype'] != 'lustre':
        raise RuntimeError('Experimental ZIP training requires persistent Lustre HPCWORK')
    checks = result['checks']
    result['quota_measurements'] = []
    result['quota_errors'] = []
    # Verify inherited project accounting on the directory we will use/create in.
    project = query(['lfs', 'project', '-d', result['existing_parent']])
    result['quota_queries'].append(project)
    fields = project.get('stdout', '').split()
    checks['project_accounting'] = (project.get('returncode') == 0 and len(fields) >= 3
                                    and fields[0] == str(project_id) and fields[1] == 'P'
                                    and ' '.join(fields[2:]) == result['existing_parent'])
    for flag, identity, number in (('-u', 'uid', os.getuid()), ('-p', 'pid', project_id)):
        row = query(['lfs', 'quota', flag, str(number), fs['target']])
        result['quota_queries'].append(row)
        try:
            measurement = quota_headroom(row, identity, number, fs['target'])
            result['quota_measurements'].append(measurement)
            checks[identity + '_quota'] = measurement['sufficient']
        except RuntimeError as error:
            checks[identity + '_quota'] = False
            result['quota_errors'].append(str(error))
    result['abort_reasons'] = [key+' failed' for key, passed in checks.items() if not passed]
    result['status'] = 'PASS' if all(checks.values()) else 'FAIL'
    result['experimental_unqualified'] = True
    return result


def serious_evaluate(run_root, filesystem_source, project_id, *, stage='allocation'):
    """Serious-only capacity; retain the historical 30 GiB policy unchanged."""
    from pathlib import Path
    from mibot.utils.claix_last2_serious import NAME, storage_budget, check_milestones
    from mibot.utils.claix_checkpoint import _read_receipt, SERIOUS_LAST2_MILESTONES
    target = Path(run_root) / 'project_xr1_doosan' / NAME
    maximum = 0
    pinned_bytes = 0
    if target.exists():
        latest = _read_receipt(target / 'latest.json')
        check_milestones(target, latest['global_step'], expected_config=latest['config_sha256'],
                         expected_experiment=latest['experiment'])
        for path in target.glob('step-*.verified.json'):
            row = _read_receipt(path)
            if (row.get('experiment') != latest['experiment'] or type(row.get('bytes')) is not int
                    or row['bytes'] <= 0):
                raise RuntimeError('Unrecognized serious receipt in storage budget')
            maximum = max(maximum, row['bytes'])
            if row['global_step'] in (*SERIOUS_LAST2_MILESTONES, 48284):
                pinned_bytes += row['bytes']
    budget = storage_budget(maximum)
    budget['measured_receipt_max_bytes'] = maximum
    budget['existing_pinned_bytes'] = pinned_bytes
    budget['required_free_bytes'] -= pinned_bytes
    result = experimental_evaluate(run_root, filesystem_source, project_id, stage=stage)
    result['budget'] = budget
    checks = result['checks']
    checks['filesystem_headroom'] = result['available_bytes'] >= budget['required_free_bytes']
    checks['inodes'] = result['available_inodes'] >= budget['required_free_inodes']
    for row in result['quota_measurements']:
        row['sufficient'] = ((row['remaining_bytes'] is None or row['remaining_bytes'] >= budget['required_free_bytes'])
                            and (row['remaining_inodes'] is None or row['remaining_inodes'] >= budget['required_free_inodes']))
        checks[row['identity'] + '_quota'] = row['sufficient']
    result['abort_reasons'] = [key + ' failed' for key, passed in checks.items() if not passed]
    result['status'] = 'PASS' if all(checks.values()) else 'FAIL'
    return result


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-root', required=True)
    p.add_argument('--filesystem-source', required=True, help='Operator-confirmed persistent findmnt SOURCE')
    p.add_argument('--project-id', type=int, help='Operator-confirmed Lustre project ID, if applicable')
    p.add_argument('--quota-diagnostics', action='store_true', help='Record optional lfs quota/r_quota output')
    p.add_argument('--experimental', action='store_true', help='Independent experimental storage and mandatory quotas')
    p.add_argument('--stage', choices=('submission', 'allocation'), default='submission')
    args = p.parse_args(argv)
    try:
        if args.experimental:
            result = experimental_evaluate(args.run_root, args.filesystem_source, args.project_id, stage=args.stage)
        else:
            result = evaluate(args.run_root, args.filesystem_source, args.project_id,
                              stage=args.stage, quota_diagnostics=args.quota_diagnostics)
    except Exception as error:
        result = dict(status='UNVERIFIED', stage=args.stage, reason=str(error), abort_reasons=[str(error)])
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result['status'] == 'PASS' else 1


if __name__ == '__main__':
    raise SystemExit(main())
