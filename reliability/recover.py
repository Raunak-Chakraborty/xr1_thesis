#!/usr/bin/env python3
"""Auditable CPU-only recovery preparation. This program never starts training."""
from __future__ import annotations

import argparse
import copy
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shlex
import shutil
import sys
import tarfile

from common import (ROOT, COMMIT, SOURCE, CHECKPOINT, CONFIG, INVENTORY, INCIDENT,
                    CPU_EVIDENCE, TARGET, beneath, digest, module, persistent,
                    read_json, regular, require, sha, source_check, tools_identity, write_new)
from stage_cache import check_stage, load_index, verify


def archive_files(path, expected):
    require(sha(path) == expected, 'Wrong or altered evidence archive')
    files = {}
    with tarfile.open(path, 'r:gz') as archive:
        for member in archive:
            if member.isdir():
                continue
            name = member.name.removeprefix('./')
            require(member.isfile() and not name.startswith('/') and '..' not in Path(name).parts
                    and name not in files and member.size <= 128 * 1024**2, 'Unsafe evidence member')
            files[name] = archive.extractfile(member).read()
    return files


def telemetry_summary(raw):
    """Validate chronology without rewriting even one original byte."""
    cursor = None
    checkpoints, fatal = [], []
    for line in raw.splitlines(keepends=True):
        require(line.endswith(b'\n') and len(line) <= 65536, 'Incomplete/oversized original telemetry')
        row = json.loads(line)
        step = row['step']
        require(type(step) is int, 'Invalid telemetry step')
        if row['event'] == 'train_start':
            require(cursor is None or cursor == step, 'Telemetry resume cursor mismatch')
            cursor = step
        elif row['event'] == 'step':
            require(cursor is not None and step == cursor + 1, 'Missing/repeated optimizer step')
            cursor = step
        else:
            require(cursor == step, 'Telemetry event cursor mismatch')
        if row['event'] == 'checkpoint':
            require(row.get('verified') is True, 'Unverified checkpoint event')
            checkpoints.append(step)
        if row['event'] == 'fatal':
            fatal.append(row)
    return dict(last_completed=cursor, last_checkpoint=max(checkpoints, default=0), fatal=fatal)


def incident(incident_path, cpu_path):
    files = archive_files(incident_path, INCIDENT)
    cpu = archive_files(cpu_path, CPU_EVIDENCE)
    target = 'production_run/' + TARGET.as_posix() + '/'
    provenance_names = [k for k in files if k.startswith('production_run/launch-') and k.endswith('.provenance.json')]
    require(len(provenance_names) == 1, 'Ambiguous original provenance')
    provenance = json.loads(files[provenance_names[0]])
    require(provenance['git_commit'] == COMMIT and provenance['source_tree_sha256'] == SOURCE
            and provenance['slurm']['SLURM_JOB_ID'] == '5363645', 'Wrong job/source provenance')
    receipt = json.loads(files[target + 'step-008750.verified.json'])
    require(receipt == json.loads(files[target + 'latest.json'])
            and receipt['sha256'] == CHECKPOINT and receipt['config_sha256'] == CONFIG
            and receipt['global_step'] == 8750, 'Incident checkpoint binding mismatch')
    summary = telemetry_summary(files[target + 'telemetry.jsonl'])
    require(summary['last_completed'] == 8758 and summary['last_checkpoint'] == 8750
            and len(summary['fatal']) == 1
            and summary['fatal'][0]['error'] == 'DataLoader timed out after 180 seconds', 'Wrong failure telemetry')
    resolved = [k for k in files if k.startswith(target + 'resolved-') and k.endswith('.json')]
    require(len(resolved) == 1, 'Ambiguous original resolved configuration')
    inventory = json.loads(files[target + 'training_inventory.json'])
    require(inventory['sha256'] == INVENTORY and digest(inventory['rows']) == INVENTORY,
            'Original dataset inventory mismatch')
    report = json.loads(cpu['forensic_report.json'])
    require(len(report['results']) == 9 and all(r['status'] == 'PASS' for r in report['results']),
            'Unexpected forensic replay report')
    return dict(files=files, provenance=provenance, receipt=receipt, telemetry=summary,
                config=json.loads(files[resolved[0]]), inventory=inventory,
                contract=json.loads(files[target + 'resume_contract.json']), cpu_report=report,
                manifest_sha256=hashlib.sha256(files['cache/media_manifest.jsonl']).hexdigest())


def original_matches(run, evidence):
    run = Path(run).resolve(strict=True)
    require(str(run) == evidence['provenance']['plan']['run_root'], 'Wrong original run root')
    hashes = {}
    for name, raw in evidence['files'].items():
        if name.startswith('production_run/'):
            relative = name[len('production_run/'):]
            expected = hashlib.sha256(raw).hexdigest()
            require(sha(beneath(run, relative)) == expected, 'Original evidence changed: ' + relative)
            hashes[relative] = expected
    return hashes


def recompose(evidence, run_root, resume=None):
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf
    overrides = evidence['provenance']['plan']['overrides'] + [
        'trainer.default_root_dir=' + json.dumps(str(Path(run_root).resolve())),
        'data.params.train_datasets.num_workers=4']
    if resume:
        overrides.append('trainer.ckpt_path=' + json.dumps(str(Path(resume).resolve())))
    with initialize_config_dir(version_base=None, config_dir=str(ROOT / 'xr1/configs')):
        actual = OmegaConf.to_container(compose(config_name='claix_b24_production', overrides=overrides), resolve=True)
    expected = copy.deepcopy(evidence['config'])
    expected['trainer']['default_root_dir'] = str(Path(run_root).resolve())
    expected['trainer']['ckpt_path'] = str(Path(resume).resolve()) if resume else None
    require(actual == expected, 'Resolved scientific/configuration drift')
    return actual


def validate_payload(path, evidence, conf):
    sys.path.insert(0, str(ROOT / 'xr1'))
    from mibot.utils.claix_contract import resume_fingerprint, split_inventory
    from mibot.utils.claix_checkpoint import load_verified
    require(sha(path) == CHECKPOINT, 'Foreign/corrupt production checkpoint')
    require(read_json(Path(path).with_suffix('.verified.json')) == evidence['receipt'], 'Wrong checkpoint receipt')
    actual = split_inventory(conf['data']['params']['train_datasets']['paths'][0])
    require(actual == evidence['inventory'], 'Dataset differs from original frozen split')
    fingerprint = resume_fingerprint(conf, actual, 12288)
    require(fingerprint == CONFIG, 'Incompatible source/configuration fingerprint')
    for name, hash_name in (('pretrained', 'pretrained_sha256'), ('language_policy_path', 'language_sha256')):
        require(sha(conf['model']['params'][name]) == conf['model']['params'][hash_name], 'Immutable model/language mismatch')
    ck = load_verified(path, evidence['contract']['replay'], expected_config=CONFIG)
    require(ck['global_step'] == 8750 and ck['xr1_replay']['rank_sample_offset'] == 210000,
            'Wrong recovery cursor')
    meta = ck['xr1_compact_fp32']
    params = conf['model']['params']
    require(all(meta[k] == params[k] for k in ('pretrained_sha256', 'language_sha256', 'source_commit', 'last_n')),
            'Compact model provenance mismatch')
    require(len(ck['xr1_production']['rng_by_rank'][0]['cuda']) == 1, 'Wrong original CUDA RNG topology')
    # CPU deserialization validates saved CUDA RNG byte tensors without restoring a CUDA context.
    return dict(global_step=8750, rank_sample_offset=210000, config_sha256=fingerprint,
                checkpoint_sha256=CHECKPOINT, trainable_parameter_count=meta['trainable_parameter_count'])


def copy_exact(source, destination):
    with regular(source).open('rb') as reader, Path(destination).open('xb') as writer:
        shutil.copyfileobj(reader, writer, 8 * 1024**2)
        writer.flush()
        os.fsync(writer.fileno())
    require(sha(source) == sha(destination), 'Copy verification failed')
    Path(destination).chmod(0o444)


def prepare(args):
    launch = source_check()
    evidence = incident(args.incident_archive, args.cpu_archive)
    original = Path(args.original_run).resolve(strict=True)
    new = Path(args.new_run).resolve()
    require(not new.exists() and not new.is_relative_to(original)
            and not original.is_relative_to(new), 'Recovery requires a new disjoint root')
    for path in (original, new, Path(args.cache).resolve()):
        persistent(path, missing=path == new)
    index = load_index(args.cache_index, args.cache_index_sha256)
    require(index['manifest_sha256'] == evidence['manifest_sha256'], 'Wrong cache manifest')
    verify(args.cache, index)
    # Read-only lock acquisition refuses a still-running writer; never create or alter its lock.
    with regular(original / '.production.lock').open('rb') as lock:
        fcntl.flock(lock, fcntl.LOCK_SH | fcntl.LOCK_NB)
        original_hashes = original_matches(original, evidence)
        checkpoint = original / TARGET / 'step-008750.ckpt'
        conf = recompose(evidence, new, checkpoint)
        payload = validate_payload(checkpoint, evidence, conf)
        plan = copy.deepcopy(evidence['provenance']['plan'])
        plan['run_root'] = str(new)
        new.mkdir(mode=0o700, parents=False)
        audit = new / '_recovery'
        audit.mkdir()
        seed = audit / 'seed'
        seed.mkdir()
        for src, dst in ((args.incident_archive, audit/'incident.tar.gz'),
                         (args.cpu_archive, audit/'cpu.tar.gz'),
                         (args.cache_index, audit/'cache-index.json'),
                         (checkpoint, seed/checkpoint.name),
                         (checkpoint.with_suffix('.verified.json'), seed/checkpoint.with_suffix('.verified.json').name)):
            copy_exact(src, dst)
        require(sha(seed/checkpoint.name) == CHECKPOINT, 'Recovery seed checksum mismatch')
        write_new(audit/'overrides.json', plan['overrides'])
        write_new(audit/'plan.json', plan)
        original_matches(original, evidence)
        record = dict(schema='xr1-5363645-recovery-v1', status='CPU_PREPARED_NOT_CERTIFIED',
                      source_commit=COMMIT, engineering_source_sha256=SOURCE,
                      tools=tools_identity(), original_run=str(original), new_run=str(new),
                      original_files=original_hashes, original_checkpoint=str(checkpoint),
                      checkpoint_receipt=evidence['receipt'], restored=payload,
                      observed_tail=[8751,8758], first_recomputed_step=8751,
                      committed_prefix_end=8750, first_previously_unobserved_step=8759,
                      seed=str(seed/checkpoint.name), cache=str(Path(args.cache).resolve()),
                      cache_index_sha256=args.cache_index_sha256, plan=plan,
                      checkout=launch.checkout_identity())
        # This is the only completion marker. Any prior failure leaves an unusable partial root.
        write_new(audit/'recovery.json', record)
        print('RECOVERY_LINEAGE_SHA256=' + sha(audit/'recovery.json'))
        print('TOOLS_SHA256=' + record['tools']['sha256'])


def check(args):
    launch = source_check()
    root = Path(args.new_run).resolve(strict=True)
    persistent(root)
    audit = root / '_recovery'
    require(sha(audit/'recovery.json') == args.lineage_sha256, 'Recovery lineage changed since review')
    record = read_json(audit/'recovery.json')
    require(record['schema'] == 'xr1-5363645-recovery-v1' and record['new_run'] == str(root), 'Wrong recovery root')
    require(record.get('checkout') == launch.checkout_identity(),
            'Reliability checkout differs from the prepared and reviewed recovery')
    require(record['tools'] == tools_identity() and record['tools']['sha256'] == args.tools_sha256,
            'Recovery tools changed since review')
    require(not (root / TARGET).exists(), 'First recovery launch requires absent target; use a new reviewed continuation procedure')
    require(read_json(audit/'plan.json') == record['plan']
            and read_json(audit/'overrides.json') == record['plan']['overrides'], 'Prepared plan/overrides changed')
    evidence = incident(audit/'incident.tar.gz', audit/'cpu.tar.gz')
    original_matches(record['original_run'], evidence)
    require(record['checkpoint_receipt'] == evidence['receipt'] and record['committed_prefix_end'] == 8750
            and record['seed'] == str(audit/'seed/step-008750.ckpt'), 'Wrong recovery checkpoint binding')
    conf = recompose(evidence, root, record['seed'])
    validate_payload(record['seed'], evidence, conf)
    index = load_index(audit/'cache-index.json', record['cache_index_sha256'])
    require(index['manifest_sha256'] == evidence['manifest_sha256'], 'Wrong cache manifest')
    cache = Path(args.staged_cache).resolve() if args.staged_cache else Path(record['cache'])
    if args.staged_cache:
        check_stage(cache, index, persistent_report=True)
    else:
        verify(cache, index)
    review = read_json(args.recovery_review)
    require(review.get('schema') == 'xr1-recovery-review-v1' and review.get('status') == 'REVIEWED'
            and review.get('reviewed_by') and review.get('reviewed_at')
            and review.get('lineage_sha256') == args.lineage_sha256
            and review.get('tools_sha256') == args.tools_sha256
            and review.get('cache_index_sha256') == record['cache_index_sha256']
            and review.get('release_sha256') == sha(args.release_record), 'Missing explicit bound recovery review')
    require(review.get('artifacts'), 'Missing real recovery validation evidence')
    for artifact in review['artifacts']:
        require(sha(artifact['path']) == artifact['sha256'], 'Recovery validation evidence changed')
    launch.check_release(args.release_record, record['plan'], require_immutable=True)
    require(not Path(args.recovery_review).stat().st_mode & 0o222, 'Recovery review must be read-only')
    command = [sys.executable, str(ROOT/'scripts/claix_b24_launch.py'), '--mode', 'production',
               '--run-root', str(root), '--resume', record['seed'], '--overrides', str(audit/'overrides.json'),
               '--release-record', str(Path(args.release_record).resolve())]
    print('REVIEWED_INPUTS_VALID=YES; TRAINING_NOT_STARTED')
    print('export XR1_DOOSAN_ZIP_CACHE_ROOT=' + shlex.quote(str(cache)))
    print('DRY_RUN_COMMAND=' + shlex.join(command))
    print('Execution remains a separate operator action through the unchanged launcher and allocation gates.')


def main():
    # Hard CPU boundary for this utility, even when called inside a GPU allocation.
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='action', required=True)
    a = sub.add_parser('audit-incident')
    a.add_argument('--incident-archive', required=True)
    a.add_argument('--cpu-archive', required=True)
    a = sub.add_parser('prepare')
    for arg in ('original-run', 'new-run', 'incident-archive', 'cpu-archive', 'cache', 'cache-index', 'cache-index-sha256'):
        a.add_argument('--' + arg, required=True)
    a = sub.add_parser('check')
    for arg in ('new-run', 'lineage-sha256', 'tools-sha256', 'recovery-review', 'release-record'):
        a.add_argument('--' + arg, required=True)
    a.add_argument('--staged-cache')
    sub.add_parser('identity')
    args = p.parse_args()
    if args.action == 'prepare':
        prepare(args)
    elif args.action == 'check':
        check(args)
    elif args.action == 'identity':
        source_check()
        print(json.dumps(tools_identity(), sort_keys=True, indent=2))
    else:
        source_check()
        evidence = incident(args.incident_archive, args.cpu_archive)
        print(json.dumps({k:evidence[k] for k in ('telemetry','receipt','manifest_sha256')}, sort_keys=True, indent=2))


if __name__ == '__main__':
    main()
