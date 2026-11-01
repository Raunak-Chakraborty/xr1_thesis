#!/usr/bin/env python3
"""Read-only storage/quota report; no data probes, deletions or inferred quota PASS."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess

from claix_b24_filesystem import inspect_directory

GIB = 1024**3
MAX_REVIEW_SECONDS = 3600
MAX_QUEUE_POLICY_SECONDS = 7*24*3600
REFRESH_POLICY = 'identical_successful_lustre_queries_v1'


def query(cmd):
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
        return dict(command=cmd, returncode=result.returncode, stdout=result.stdout, stderr=result.stderr)
    except (OSError, subprocess.TimeoutExpired) as error:
        return dict(command=cmd, error=str(error))


def complete_lustre_snapshot(result, identity, identifier, mount):
    """Recognize only full, unrounded, non-grace Lustre rows for unattended reuse.

    Capacity interpretation remains the operator's signed review. Byte-identical
    fresh output proves the reviewed usage AND limits still apply. Unknown site
    formats, truncation, warnings, exceeded quotas and unsupported tools fail closed.
    """
    if result.get('returncode') != 0 or result.get('stderr', '').strip():
        return False
    output = result.get('stdout', '')
    if not re.search(r'\(' + identity + r'\s+' + str(identifier) + r'\):', output):
        return False
    lines = [line.split() for line in output.splitlines() if line.strip()]
    header = ['Filesystem', 'kbytes', 'quota', 'limit', 'grace', 'files', 'quota', 'limit', 'grace']
    if header not in lines:
        return False
    data = [word for line in lines[lines.index(header)+1:] for word in line]
    if len(data) != 9 or data[4] != '-' or data[8] != '-':
        return False
    try:
        return (Path(data[0]).resolve(strict=True) == Path(mount).resolve(strict=True)
                and all(re.fullmatch(r'\d+', data[i]) for i in (1, 2, 3, 5, 6, 7)))
    except (OSError, RuntimeError):
        return False


def evaluate(run_root, filesystem_source, quota_review=None, project_id=None, *, stage='submission'):
    if stage not in ('submission', 'allocation'):
        raise ValueError('Unknown storage preflight stage')
    location = inspect_directory(run_root, allow_missing=True, expected_source=filesystem_source)
    root, parent, fs = location['resolved_path'], location['existing_parent'], location['filesystem']
    stat = os.statvfs(parent)
    free = stat.f_bavail * stat.f_frsize
    budget = dict(compact_bytes=2*GIB, retained_payloads=9, new_publications=1,
                  logs_and_temporary_reserve_bytes=4*GIB, required_free_bytes=30*GIB,
                  required_free_inodes=10000)
    # Stable mount target, so creating the run root between checks does not alter
    # the quota command binding. GID is never assumed to be the project ID.
    mount = fs['target']
    quotas = [query(['lfs', 'quota', '-u', str(os.getuid()), mount]),
              (query(['lfs', 'quota', '-p', str(project_id), mount]) if project_id is not None
               else dict(status='UNVERIFIED', reason='Project ID not supplied; GID is not a project ID')),
              query(['quota', '-s', '-u', str(os.getuid())])]
    raw_hash = hashlib.sha256(json.dumps(quotas, sort_keys=True).encode()).hexdigest()
    checks = dict(persistent_filesystem=True, filesystem_headroom=free >= budget['required_free_bytes'],
                  inodes=stat.f_favail >= budget['required_free_inodes'])
    now = datetime.now(timezone.utc)
    quota, method, reason, review_hash = 'UNVERIFIED', 'UNVERIFIED', 'No operator quota review supplied', None
    if quota_review:
        raw = Path(quota_review).read_bytes()
        review_hash = hashlib.sha256(raw).hexdigest()
        review = json.loads(raw)
        reviewed_at = datetime.fromisoformat(review['timestamp'])
        if reviewed_at.tzinfo is None:
            raise ValueError('Quota review timestamp must have a timezone')
        age = (now-reviewed_at).total_seconds()
        expires = datetime.fromisoformat(review['allocation_valid_until']) if review.get('allocation_valid_until') else None
        expiry_valid = (expires is None or (expires.tzinfo is not None and now <= expires
                        and 0 < (expires-reviewed_at).total_seconds() <= MAX_QUEUE_POLICY_SECONDS))
        valid = (review.get('reviewed_by') and review.get('run_root') == root
                 and review.get('filesystem_source') == fs['source'] and review.get('uid') == os.getuid()
                 and review.get('project_id') == project_id and review.get('user_and_project_reviewed') is True
                 and 0 <= age and expiry_valid and math.isfinite(review.get('usable_bytes', 0))
                 and math.isfinite(review.get('usable_inodes', 0))
                 and review.get('usable_bytes', 0) >= budget['required_free_bytes']
                 and review.get('usable_inodes', 0) >= budget['required_free_inodes']
                 and review.get('quota_query_sha256') == raw_hash)
        reason = ('Quota review expiry invalid or expired' if not expiry_valid else
                  'Quota review identity, capacity or fresh query snapshot does not match')
        if valid and age <= MAX_REVIEW_SECONDS:
            quota, method, reason = 'PASS', 'explicit_operator_review', 'Fresh queries match recent operator review'
        elif valid:
            reason = 'Quota review stale; no valid unattended refresh policy'
            policy_valid = (stage == 'allocation' and review.get('allocation_refresh_policy') == REFRESH_POLICY
                            and expires is not None and expires.tzinfo is not None
                            and now <= expires and 0 < (expires-reviewed_at).total_seconds() <= MAX_QUEUE_POLICY_SECONDS)
            if (policy_valid and fs['fstype'] == 'lustre' and project_id is not None
                    and complete_lustre_snapshot(quotas[0], 'uid', os.getuid(), mount)
                    and complete_lustre_snapshot(quotas[1], 'projid', project_id, mount)):
                quota, method, reason = ('PASS', REFRESH_POLICY,
                    'Fresh successful user/project quota snapshots exactly match operator-reviewed usage and limits')
            elif stage == 'submission':
                reason = 'Submission requires operator quota review within one hour'
    reasons = [key+' failed' for key, passed in checks.items() if not passed]
    if quota != 'PASS':
        reasons.append(reason)
    status = 'FAIL' if not all(checks.values()) else ('PASS' if quota == 'PASS' else 'UNVERIFIED')
    return dict(status=status, stage=stage, timestamp=now.isoformat(), run_root=root,
                existing_parent=parent, filesystem=fs, available_bytes=free,
                available_inodes=stat.f_favail, budget=budget, checks=checks, quota_status=quota,
                project_id=project_id, uid=os.getuid(), quota_queries=quotas, quota_query_sha256=raw_hash,
                quota_review_sha256=review_hash, quota_verification=method, reason=reason, abort_reasons=reasons,
                operator_review='Review actual user AND project quota, grace/soft/hard limits, existing retention and orphan usage; GID is not a project ID.')


def allocation_recheck(record):
    """Called by the production launcher after release validation, before model loading."""
    binding = record['storage_preflight']
    review = Path(binding['quota_review'])
    if hashlib.sha256(review.read_bytes()).hexdigest() != binding['quota_review_sha256']:
        raise RuntimeError('Quota review changed after release approval')
    return evaluate(record['plan']['run_root'], binding['filesystem_source'], review,
                    binding.get('project_id'), stage='allocation')


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-root', required=True)
    p.add_argument('--filesystem-source', required=True, help='Operator-confirmed persistent findmnt SOURCE')
    p.add_argument('--project-id', type=int, help='Operator-confirmed Lustre project ID, if applicable')
    p.add_argument('--quota-review', help='Explicit operator attestation; see FINAL_CORRECTIONS.md')
    p.add_argument('--stage', choices=('submission', 'allocation'), default='submission')
    args = p.parse_args(argv)
    try:
        result = evaluate(args.run_root, args.filesystem_source, args.quota_review, args.project_id, stage=args.stage)
    except Exception as error:
        result = dict(status='UNVERIFIED', stage=args.stage, reason=str(error), abort_reasons=[str(error)])
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result['status'] == 'PASS' else 1


if __name__ == '__main__':
    raise SystemExit(main())
