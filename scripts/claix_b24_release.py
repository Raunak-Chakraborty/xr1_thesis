#!/usr/bin/env python3
"""Prepare an UNVERIFIED draft, or check an explicitly reviewed release. Never submits."""
import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path

GATES = ('frozen_splits', 'tokens_24_examples_48_views', 'workers_0_4_8',
         'h100_b24_memory', 'end_to_end_throughput', 'real_compact_resume',
         'safe_stop_and_storage', 'held_out_validation', 'lr_and_horizon_approval')

SCIENCE = dict(max_lr=2e-5, min_lr=5e-6, warmup_lr_start=5e-7, warmup_steps=500,
               horizon=48284, batch_size=24, workers=4, last_n=1, cameras=2,
               optimizer='AdamW', compute='BF16', trainable_weights='FP32', checkpoint_interval=250)
CURRENT_H100_GATES = {'h100_b24_memory', 'end_to_end_throughput', 'real_compact_resume',
                      'safe_stop_and_storage'}


def file_digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8*1024**2), b''):
            h.update(block)
    return h.hexdigest()


def validate_review(record):
    if set(record.get('gates', {})) != set(GATES):
        raise RuntimeError('Release requires exactly nine gates')
    if (record.get('schema') != 'claix-b24-release-v2' or not record.get('reviewed_by')
            or not record.get('reviewed_at') or record.get('scientific_parameters') != SCIENCE
            or record.get('scientific_approved_by') in (None, '')):
        raise RuntimeError('Missing explicit release/scientific approval')
    plan = record['plan']
    for key, expected in dict(batch_size=24, max_length=12288, data_horizon=48284,
                              scheduler_steps=48284, warmup_steps=500, workers=4, world_size=1).items():
        if plan.get(key) != expected:
            raise RuntimeError('Unapproved production plan: ' + key)
    if not Path(plan.get('run_root', '')).is_absolute():
        raise RuntimeError('Production run root must be absolute')
    for gate in GATES:
        row = record['gates'][gate]
        if not Path(row['evidence']).is_absolute():
            raise RuntimeError('Use absolute evidence paths')
        envelope = json.loads(Path(row['evidence']).read_text())
        # Envelope is reviewable provenance for the actual artifact(s), including
        # historical evidence. No inferred source commit/job ID or hardware result.
        if (envelope.get('gate') != gate or not envelope.get('source_commit')
                or len(envelope.get('source_tree_sha256', '')) != 64
                or envelope.get('kind') not in ('historical_h100', 'current_h100', 'cpu', 'operator')
                or not row.get('applicability_reviewed_by') or not row.get('applicability_rationale')):
            raise RuntimeError('Incomplete evidence provenance/applicability: ' + gate)
        if envelope['kind'] in ('historical_h100', 'current_h100') and not envelope.get('job_id'):
            raise RuntimeError('Missing original H100 job ID')
        if gate in CURRENT_H100_GATES and (envelope['kind'] != 'current_h100'
                or envelope['source_tree_sha256'] != record['source_tree_sha256']):
            raise RuntimeError('Final-source H100 evidence required: ' + gate)
        if envelope['kind'] == 'current_h100' and envelope['source_tree_sha256'] != record['source_tree_sha256']:
            raise RuntimeError('Current H100 evidence source mismatch')
        artifacts = envelope.get('artifacts', [])
        if not artifacts:
            raise RuntimeError('Missing real evidence artifacts: ' + gate)
        for artifact in artifacts:
            if not Path(artifact['path']).is_absolute():
                raise RuntimeError('Use absolute artifact paths')
            if file_digest(artifact['path']) != artifact.get('sha256'):
                raise RuntimeError('Evidence artifact checksum mismatch: ' + gate)
        artifact_paths = {str(Path(artifact['path']).resolve()) for artifact in artifacts}
        fields = ['source_commit', 'source_tree_sha256']
        if envelope['kind'] in ('historical_h100', 'current_h100'):
            fields.append('job_id')
        for field in fields:
            binding = envelope.get('provenance_bindings', {}).get(field, {})
            provenance = Path(binding.get('path', ''))
            if str(provenance.resolve()) not in artifact_paths:
                raise RuntimeError('Missing hashed original provenance: ' + field)
            value = json.loads(provenance.read_text())
            for key in binding.get('key', '').split('.'):
                value = value[key]
            if str(value) != str(envelope[field]):
                raise RuntimeError('Original evidence provenance mismatch: ' + field)
    validate_storage_binding(record)


def validate_storage_binding(record):
    """Bind a real pre-submission review; its age is re-evaluated at allocation."""
    binding = record.get('storage_preflight', {})
    for key in ('report', 'quota_review'):
        path = Path(binding.get(key, ''))
        if not path.is_absolute() or file_digest(path) != binding.get(key+'_sha256'):
            raise RuntimeError('Missing or changed submission storage evidence: ' + key)
    report = json.loads(Path(binding['report']).read_text())
    review = json.loads(Path(binding['quota_review']).read_text())
    checked = datetime.fromisoformat(report['timestamp'])
    reviewed = datetime.fromisoformat(review['timestamp'])
    if (checked.tzinfo is None or reviewed.tzinfo is None
            or not 0 <= (checked-reviewed).total_seconds() <= 3600
            or report.get('status') != 'PASS' or report.get('stage') != 'submission'
            or report.get('quota_verification') != 'explicit_operator_review'
            or report.get('run_root') != record['plan']['run_root']
            or review.get('run_root') != record['plan']['run_root']
            or report.get('quota_review_sha256') != binding['quota_review_sha256']
            or report.get('quota_query_sha256') != review.get('quota_query_sha256')
            or not binding.get('filesystem_source')
            or report.get('filesystem', {}).get('source') != binding['filesystem_source']
            or review.get('filesystem_source') != binding['filesystem_source']
            or report.get('project_id') != binding.get('project_id')
            or review.get('project_id') != binding.get('project_id')
            or not review.get('reviewed_by') or review.get('user_and_project_reviewed') is not True):
        raise RuntimeError('Invalid pre-submission storage/quota review')



def main(argv=None):
    from claix_b24_launch import checkout_identity, tree_digest
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--plan', required=True, help='Exact production_plan object saved from dry-run')
    p.add_argument('--output', help='Exclusive-create UNVERIFIED draft')
    p.add_argument('--check')
    args = p.parse_args(argv)
    plan = json.loads(Path(args.plan).read_text())
    if args.check:
        from claix_b24_launch import check_release
        check_release(args.check, plan, require_immutable=True)
        print('Reviewed release checks passed; no submission performed')
    elif args.output:
        record = dict(schema='claix-b24-release-v2', source_tree_sha256=tree_digest(),
                      checkout=checkout_identity(), plan=plan, scientific_parameters=SCIENCE,
                      storage_preflight=dict(report='', report_sha256='', quota_review='',
                          quota_review_sha256='', filesystem_source='', project_id=None),
                      scientific_approved_by='', reviewed_by='', reviewed_at='',
                      gates={gate: dict(status='UNVERIFIED', evidence='', sha256='',
                              applicability_reviewed_by='', applicability_rationale='') for gate in GATES})
        with Path(args.output).open('x') as stream:
            json.dump(record, stream, indent=2, sort_keys=True)
            stream.write('\n')
    else:
        p.error('Choose --output or --check')


if __name__ == '__main__':
    main()
