"""Execute actual XR-1 methods/DiT on CPU without importing the 5B backbone.

AST loading only removes imports and the registry decorator. The actual forward,
loss, generation, attention and DiT definitions run with PyTorch. A small VLM
test double supplies a cache; it does not verify Qwen/checkpoint compatibility.
"""

import ast
import importlib.util
import math
from pathlib import Path
import random
from types import SimpleNamespace

import torch
from torch import nn
import torch.nn.functional as F
from torch.distributions import Beta

from mibot.utils.force_contract import ForceEncoderConfig
from mibot.utils.model_utils import auto_cast

ROOT = Path(__file__).resolve().parents[2]


def load_file(name, path):
    spec = importlib.util.spec_from_file_location(name, ROOT / path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


force = load_file("isolated_force_encoder", "mibot/models/force_encoder.py")


class RMSNorm(nn.Module):
    # Qwen-compatible formula for isolated DiT tests, without transformers.
    def __init__(self, size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(size))
        self.eps = eps

    def forward(self, value):
        return self.weight * (value.float() * torch.rsqrt(value.float().square().mean(-1, keepdim=True) + self.eps)).to(value.dtype)


def rotate_half(x):
    left, right = x.chunk(2, dim=-1)
    return torch.cat((-right, left), dim=-1)


def xr1_definitions():
    tree = ast.parse((ROOT / "mibot/models/VLA/XR1.py").read_text())
    definitions = []
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
            if isinstance(node, ast.ClassDef) and node.name == "xr1":
                node.decorator_list = []
            definitions.append(node)
    namespace = dict(torch=torch, nn=nn, F=F, math=math, random=random, Beta=Beta,
                     Qwen2RMSNorm=RMSNorm, rotate_half=rotate_half, auto_cast=auto_cast,
                     ACT2FN={"silu": F.silu}, ForceEncoderConfig=ForceEncoderConfig,
                     ForceConditioner=force.ForceConditioner,
                     split_force_state=force.split_force_state,
                     apply_force_train_policy=force.apply_force_train_policy,
                     ACTION_START_ID=100, ACTION_END_ID=160, SCORE_ID=99)
    exec(compile(ast.Module(body=definitions, type_ignores=[]), "XR1.py", "exec"), namespace)
    return namespace


class VLMStub(nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = nn.Parameter(torch.ones(1))
        self.seen_state = None

    def forward(self, input_ids, state_embeds=None, **unused):
        self.seen_state = state_embeds
        batch, length = input_ids.shape
        hidden = self.anchor * torch.ones(batch, length, 8, dtype=self.anchor.dtype)
        cache = self.anchor * torch.ones(batch, 8, length, 128, dtype=self.anchor.dtype) * 0.01
        return SimpleNamespace(hidden_states=hidden, past_key_values=[(cache, cache)],
                               position_ids=torch.arange(length).reshape(1, 1, -1).expand(3, batch, -1),
                               attention_mask=torch.ones(batch, length, dtype=torch.bool))


class RotaryStub(nn.Module):
    def forward(self, action, ids):
        shape = (ids.shape[1], ids.shape[2], 128)
        return torch.ones(shape, dtype=action.dtype), torch.zeros(shape, dtype=action.dtype)


def tiny_model(enabled=True, **overrides):
    ns = xr1_definitions()

    class TinyXR1(ns["xr1"]):
        def _build_model(self):
            self.vlm = VLMStub()
            self.state_projector_choice = nn.Linear(60, 8)
            self.action_projector_choice = nn.Linear(8, 60 * self.n_choices)
            self.score_projector_choice = nn.Linear(8, self.n_choices)
            # Real attention, MLP, modulation and flow path, one DiT layer.
            self.dit = ns["DiT"](layer_num=1)
            self.state_projector = nn.Linear(60, 1024)
            self.action_projector = nn.Linear(60, 1024)
            self.action_output_layer = nn.Linear(1024, 60)
            self.t_embedder = ns["TimestepEmbedder"](1024)
            self.t_projector = ns["Projector"](1024, 6 * 1024, bias=True)
            self.sink = nn.Embedding(1, 1024)
            self.rotary_emb = RotaryStub()
            self.to(torch.bfloat16)

    model = TinyXR1(async_train=False, force_encoder={"enabled": enabled, **overrides})
    model.training_repeat = 2
    model.num_steps = 2
    return model


def batch(training=True, batch_size=1):
    state = torch.zeros(batch_size, 1, 60)
    state[..., 20:26] = torch.tensor([-.8, -.4, -.1, .2, .5, .9])
    mask = torch.zeros(batch_size, 30, 60, dtype=torch.int32)
    mask[:, :3, :7] = 1
    result = dict(state=state, action=torch.zeros(batch_size, 30, 60), action_mask=mask,
                  force_normalization=torch.ones(batch_size, dtype=torch.int64),
                  input_ids=torch.tensor([[100, 101, 102, 99]]).expand(batch_size, -1))
    if training:
        result.update(vlm_action_actual_length=torch.tensor([3] * batch_size),
                      vlm_action_target=torch.zeros(3 * batch_size, 60),
                      vlm_action_mask=mask[:, :3].flatten(0, 1))
    return result
