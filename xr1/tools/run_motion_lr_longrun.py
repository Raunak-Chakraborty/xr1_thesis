#!/usr/bin/env python3
"""Run verified natural Last-1 long-run trainer with peak AdamW LR=2e-5.

This is a combined motion-sampling plus LR experiment. The 48,284-step
trainer, checkpoint/replay, model/loss/optimizer implementations are unmodified.
"""
from tools import run_natural_longrun as original

NAME = original.MODEL
assert NAME == 'natural_adamw_l1'
profile = original.base.PROFILE_NAMES[NAME]
assert profile['kind'] == 'adamw_control'
assert profile['last_n'] == 1
assert profile['sampler'] == 'natural'
assert profile['adamw_lr'] == 1e-5
profile['adamw_lr'] = 2e-5

if __name__ == '__main__':
    print('COMBINED_MOTION_LR2E5_WRAPPER=PASS adamw_lr=2e-5', flush=True)
    original.main()
