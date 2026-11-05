# Task 03 — CLAIX compatibility corrections

Implemented on `thesis/xr1-lustre-freshstart-v1`, starting at
`a0b52a6b8a31f3d2c934ab23f5fb4d99f3b4c51f`. The scoped corrections are complete.
The laptop CPU tests that executed passed. Full CLAIX framework acceptance remains
required; the laptop full harness could not finish collection.

## Evidence inspected

Read the supplied archive in place:

`/home/horus/robotics_thesis/xr1_claix/evidence/freshstart/claix_cpu_failures/cpu_failure_evidence_20261105T183235Z.tar.gz`

Inspected its summary, experimental CPU, I/O, full-suite and real Hydra/dataset
logs. CPU acceptance job **5504490** reported experimental tests passing, I/O
38 tests with one failure, and full-suite 189 tests with three failures and
17 errors. The real Hydra/dataset check reported PASS, 255 training episodes,
289703 frames, timeout 600, workers four and prefetch factor two. These are prior
CLAIX results, not new checks performed from this laptop.

## Failures, causes and corrections

| Failure | Root cause | Correction |
| --- | --- | --- |
| Experimental allocation request rejected | CLAIX advertises the `hopper` GRES type; the supplied real test-only submission evidence rejects `h100` | Change only the experimental wrapper's GRES directive; update wrapper and synthetic allocation expectations |
| Sixteen missing-reference errors | Legacy tests implicitly read ignored `.claix_b24_reference/configs/hydra_overrides.json`, which intentionally does not exist in this worktree | Tests explicitly pass temporary synthetic overrides, including evaluator integration and allocation/monitor tests |
| ZIP sidecar rewrite assertion | Rapid rewrite and restored mtime can leave the same observable ctime on Lustre; the I/O log shows this for immutable and the full log for strict policy | Explicitly change mtime by two seconds and verify the observed mtime differs before expecting invalidation |
| Legacy production wall-time assertion | Test still expected 23:45:00 although the committed production wrapper is 20:00:00 | Expect 20:00:00 in the test; production wrapper unchanged |
| Legacy Hydra qualification command | Test combined resume and an explicit stop step; launcher correctly exits with status two | Exercise supported fresh qualification with stop five and separate resume qualification deriving stop 21506 from the required step-21500 receipt |
| Monitor start expected, zero calls | Missing overrides abort before monitor construction. The success subcase errors; the failure subcase's broad `OSError` assertion accidentally accepts `FileNotFoundError` and then fails the monitor assertion | Supply explicit overrides, require the exact synthetic child-start error, assert child construction and monitor start/close once, and verify signal restoration |

The seventeenth full-suite error is the intentionally rejected qualification CLI
combination. The extra monitor assertion failure is another consequence of the
missing fixture, not evidence of a launcher cleanup regression. No launcher,
monitor, signal-handler or CLI-validation implementation was changed.

The exact Slurm correction is:

```diff
-#SBATCH --gres=gpu:h100:1
+#SBATCH --gres=gpu:hopper:1
```

Every other byte in the experimental wrapper is unchanged, including account
`thes2452`, partition `c23g`, one node, one task, 12 CPUs/task, 96G memory,
20:00:00 and `B:USR1@1800`. The actual allocation still requires exactly one
Slurm-visible **NVIDIA H100**. Tests accept synthetic `gres/gpu:hopper=1` allocation
metadata while continuing to reject A100, multiple devices and invalid resources.
No other batch wrapper was edited.

## Test fixture ownership and integrity policy

`claix_test_fixtures.py` supplies exactly four visibly synthetic path overrides
under `/hpcwork/SYNTHETIC/`. It contains no real CLAIX dataset locations. Each
affected unittest case creates its own temporary directory, registers its cleanup
before writing, and explicitly passes `--overrides`. The evaluator writes into
its existing temporary-directory context. Fixture files use exclusive creation.
No test creates, reads, replaces or removes the ignored reference configuration.
Production launcher defaults and its four-path validation are unchanged; the
negative override-validation test still exercises the real validation.

This works for discovery, focused harnesses and individual test cases. The old
focused-harness-only monkeypatch of `launch.main` was removed. Framework-dependent
imports in the metrics test module now occur inside the methods that need them,
so monitor tests can execute without importing a Lightning training fixture.
The focused harness includes all monitor-release and monitor-qualification cases.

The ZIP regression continues to cover archive and sidecar replacement, same-size
rewrite with observable metadata change, and deletion under both policies.
Strict mode invalidates/revalidates when its observed signature changes. Immutable
mode deliberately reuses a held handle until a miss/eviction forces validation.
The test no longer depends on elapsed time or filesystem ctime resolution. Neither
policy promises detection of an in-place write whose observable metadata stays
unchanged. Strict stat checks are not a transactional lock or content checksum.
Archive member CRC/draining, initial validation, bounds, provenance and both policy
implementations are unchanged. No production sleep was introduced.

Resume tests use synthetic receipts in temporary directories for dry command
composition only, without a checkpoint payload. A valid identity produces the
derived stop and full horizon; missing receipts and incorrect schema, basename,
step or checksum remain rejected. Resume plus explicit stop is separately checked
for exit status two and the specific validation message. Historical checkpoints
and their identity checks were not changed.

## Laptop validation

Used existing Python
`/home/horus/anaconda3/envs/forcevla_thesis_env/bin/python`, torch 2.7.1+cu128 on
CPU with CUDA hidden. No packages, datasets or model weights were downloaded.

```bash
CUDA_VISIBLE_DEVICES='' /home/horus/anaconda3/envs/forcevla_thesis_env/bin/python -B \
  xr1/tests/run_claix_experimental_cpu_tests.py
CUDA_VISIBLE_DEVICES='' /home/horus/anaconda3/envs/forcevla_thesis_env/bin/python -B \
  xr1/tests/run_doosan_io_cpu_tests.py --allow-framework-stubs
CUDA_VISIBLE_DEVICES='' /home/horus/anaconda3/envs/forcevla_thesis_env/bin/python -B \
  xr1/tests/run_claix_cpu_tests.py
```

| Check | Result |
| --- | --- |
| Expanded experimental/focused legacy CPU harness | **84 passed**, zero failed/errors/skipped, 8.082 s |
| I/O CPU harness | **38 passed**, zero failed/errors/skipped, 0.648 s |
| Full CLAIX CPU harness attempt | **Unavailable locally**: collection stops with `ModuleNotFoundError: No module named 'lightning'`; no test execution/result from this harness |
| Real Hydra tests | Hydra is also absent. Two Hydra composition methods are deliberately excluded from the focused harness; evaluator integration requires the CLAIX framework environment |
| Python syntax | AST parsing passed for all nine changed/new Python files |
| Bash syntax | `bash -n` passed for all six CLAIX batch wrappers plus the environment script |
| Task 01/01B/02 protection comparison | 21 key runtime/configuration/storage/release files byte-identical to starting commit |
| Experimental wrapper comparison | Exact one-directive replacement verified |
| Whitespace | `git diff --check` passed |

The experimental harness explicitly stubs unavailable Lightning Callback and
CheckpointIO bases; it does not execute a real Trainer. The I/O harness uses its
existing explicit mmengine/backbone and LightningDataModule-base stubs. Actual
PyTorch CPU optimizer/scheduler operations, checkpoint I/O, dataset augmentation,
ordered DataLoader, cache and storage/launch guards execute as covered by those
harnesses. The monitor tests use mocked queries or a temporary fake executable;
no real GPU query is performed. Qualification/resume tests generate commands or
mock the child process; they do not start training or contact Slurm.

The full harness is **not reported as passed**. No executed test failed; its
dependency/collection failure is retained in the evidence. Evidence files:

- `task03_evidence/experimental_cpu.log`
- `task03_evidence/io_cpu.log`
- `task03_evidence/full_claix_cpu_attempt.log`

## CLAIX validation still required

Rerun all three CPU harnesses in the existing CLAIX environment, including the
full discovery harness with real Hydra, Lightning and mmengine. Specifically
verify both corrected Hydra command-composition cases and evaluator integration
with temporary synthetic paths and no reference file. Rerun the ZIP regression
on Lustre to confirm the explicit observed mtime change, both policy behaviors,
replacement/deletion coverage and unchanged CRC checks.

The supplied successful `gpu:hopper:1` test-only submission evidence supports the
directive correction. Any later operator-run allocation must still satisfy the
unchanged actual-H100 and exact-resource checks. This task does not establish
GPU training acceptance or production qualification.

## Scientific and operational protections

All model/training/data configurations and Python production implementations are
unchanged. Original pretrained Xiaomi XR-1 identity, architecture, losses,
optimizer/scheduler, batch 24, four workers, prefetch two, persistent workers,
ordered deterministic loading, timeout 600 seconds, frozen dataset/sampler,
immutable ZIP policy, horizon 48284, checkpoint interval 250, safe-stop margin
1800 seconds, experimental optimizer-step-zero initialization and same-run
verified recovery are preserved. Strict production release requirements remain
unchanged. No node-local SSD staging was added.

Task 01/01B cache implementations, DataLoader and replay protections, and Task 02
experimental/checkpoint/runtime protections remain intact. The focused suites
exercise them; byte comparisons also covered both CLAIX recipes, model/trainer/
data recipes, caches, dataset, datamodule, replay, checkpoint, contract,
experimental/runtime helpers, training entrypoint, launcher, strict wrapper,
release validator, filesystem/storage preflight and environment bootstrap.

No GPU tests or Slurm commands were run, and no GPU/Slurm jobs were submitted.
No datasets or weights were downloaded; no historical checkpoints, private
reference configuration or other worktrees were modified. No commit or push was
performed.

## Exact changed files

1. `scripts/claix_b24_experimental.sbatch`
2. `xr1/tests/claix_test_fixtures.py` (new)
3. `xr1/tests/run_claix_experimental_cpu_tests.py`
4. `xr1/tests/test_claix_evaluation.py`
5. `xr1/tests/test_claix_experimental.py`
6. `xr1/tests/test_claix_final_corrections.py`
7. `xr1/tests/test_claix_final_release.py`
8. `xr1/tests/test_claix_launch.py`
9. `xr1/tests/test_claix_metrics_release.py`
10. `xr1/tests/test_doosan_io_cache.py`
11. `docs/claix_b24/LUSTRE_FRESHSTART_TASK03.md` (new)
12. `docs/claix_b24/task03_evidence/experimental_cpu.log` (new)
13. `docs/claix_b24/task03_evidence/io_cpu.log` (new)
14. `docs/claix_b24/task03_evidence/full_claix_cpu_attempt.log` (new)
