"""Actual PyTorch CPU tests; explicit skips when PyTorch is unavailable.

No checkpoints, downloads, video decoding, networking, or GPU work.
The full backbone is verified separately by tools/verify_force_encoder.py.
"""

import copy
from dataclasses import replace
import importlib.util
from io import BytesIO
import unittest

TORCH = importlib.util.find_spec("torch") is not None
NUMPY = importlib.util.find_spec("numpy") is not None
if TORCH:
    import torch
    from torch import nn
    from encoder_support.harness import batch, force, tiny_model, xr1_definitions
    from mibot.utils.force_checkpoint import load_force_checkpoint
    from mibot.utils.force_normalization import normalize_force_state


@unittest.skipUnless(TORCH, "PyTorch unavailable; numerical tests not executed")
class ForceTensorTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        self.state = torch.randn(2, 1, 60).clamp(-1, 1)
        self.state[..., 20:26] = torch.tensor([-.9, -.6, -.2, .1, .4, .8])
        self.marker = torch.ones(2, dtype=torch.int64)

    def test_exact_order_extraction_zeroing_and_no_mutation(self):
        original = self.state.clone()
        clean, wrench = force.split_force_state(self.state, self.marker)
        torch.testing.assert_close(wrench, original[..., 20:26], rtol=0, atol=0)
        self.assertEqual(wrench.shape, (2, 1, 6))
        self.assertEqual(torch.count_nonzero(clean[..., 20:26]).item(), 0)
        torch.testing.assert_close(clean[..., :20], original[..., :20], rtol=0, atol=0)
        torch.testing.assert_close(clean[..., 26:], original[..., 26:], rtol=0, atol=0)
        torch.testing.assert_close(self.state, original, rtol=0, atol=0)

    def test_dual_path_is_explicit(self):
        clean, _ = force.split_force_state(self.state, self.marker, dual_path=True)
        self.assertIs(clean, self.state)

    def test_shapes_dtype_and_device(self):
        for dtype in (torch.float32, torch.bfloat16):
            module = force.ForceConditioner({"hidden_dims": [11], "output_dim": 9}, hidden_size=16).to(dtype)
            result = module(self.state[..., 20:26])
            self.assertEqual(result.shape, (2, 6, 16))
            self.assertEqual(result.dtype, dtype)
            self.assertEqual(result.device, self.state.device)
            self.assertTrue(torch.isfinite(result).all())
        module = force.ForceConditioner(hidden_size=16).to("meta")
        with self.assertRaisesRegex(ValueError, "same device"):
            module(self.state[..., 20:26])

    def test_malformed_nonfinite_and_unnormalized_inputs_rejected(self):
        module = force.ForceConditioner(hidden_size=16)
        for bad in (torch.zeros(2, 6), torch.zeros(2, 2, 6), torch.zeros(2, 1, 5),
                    torch.zeros(2, 1, 6, dtype=torch.int64),
                    torch.full((2, 1, 6), float("nan")), torch.full((2, 1, 6), float("inf")),
                    torch.full((2, 1, 6), 1.1)):
            with self.subTest(shape=bad.shape), self.assertRaises(ValueError):
                module(bad)
        for bad in (torch.zeros(2, 60), torch.zeros(2, 2, 60), torch.zeros(2, 1, 25),
                    torch.zeros(2, 1, 60, dtype=torch.int64),
                    torch.full((2, 1, 60), float("nan"))):
            with self.assertRaises(ValueError):
                force.split_force_state(bad, self.marker)

    def test_normalization_marker_required(self):
        for marker in (None, torch.ones(1, dtype=torch.int64), torch.zeros(2, dtype=torch.int64), torch.ones(2)):
            with self.assertRaises(ValueError):
                force.split_force_state(self.state, marker)

    def test_normalizes_once_and_clips_large_finite_raw_forces(self):
        low, high = torch.full((1, 60), -10.), torch.full((1, 60), 30.)
        raw = torch.zeros(2, 1, 60)
        raw[0, 0, 20:26] = torch.tensor([0., 1., 5., 10., 20., 25.])
        raw[1, 0, 20:26] = torch.tensor([1e12, -1e12, 1e10, -1e10, 1e8, -1e8])
        normalized = normalize_force_state(raw, low, high)
        _, wrench = force.split_force_state(normalized, self.marker)
        expected = (2 * ((raw[..., 20:26] + 10) / ((30 + torch.tensor(1e-6)) + 10)) - 1).clamp(-1, 1)
        torch.testing.assert_close(wrench, expected)
        module = force.ForceConditioner(hidden_size=16)
        for values in (wrench, torch.zeros_like(wrench)):
            self.assertTrue(torch.isfinite(module(values)).all())
        raw[0, 0, 20] = float("inf")
        with self.assertRaises(ValueError):
            normalize_force_state(raw, low, high)

    def test_each_of_six_channels_reaches_encoder_and_changes_features(self):
        module = force.ForceConditioner(hidden_size=16)
        inputs = []
        handle = module.encoder[0].register_forward_pre_hook(lambda _, args: inputs.append(args[0].clone()))
        zero = torch.zeros(1, 1, 6)
        base = module(zero)
        for axis in range(6):
            changed = zero.clone()
            changed[..., axis] = .75
            output = module(changed)
            torch.testing.assert_close(inputs[-1], changed, rtol=0, atol=0)
            self.assertGreater((output - base).abs().max().item(), 0)
        handle.remove()

    def test_first_step_nonzero_finite_encoder_and_fusion_gradients(self):
        module = force.ForceConditioner(hidden_size=16)
        result = module(self.state[..., 20:26])
        result.square().mean().backward()
        for name, parameter in module.named_parameters():
            with self.subTest(name=name):
                self.assertIsNotNone(parameter.grad)
                self.assertTrue(torch.isfinite(parameter.grad).all())
                self.assertGreater(parameter.grad.abs().max().item(), 0)

    def test_reproducible_initialization_preserves_rng_and_bias_policy(self):
        before = torch.get_rng_state().clone()
        first = force.ForceConditioner(hidden_size=16)
        self.assertTrue(torch.equal(before, torch.get_rng_state()))
        torch.rand(99)
        second = force.ForceConditioner(hidden_size=16)
        for name, value in first.state_dict().items():
            torch.testing.assert_close(value, second.state_dict()[name], rtol=0, atol=0)
            if name.endswith("bias"):
                self.assertEqual(torch.count_nonzero(value).item(), 0)
        self.assertGreater(first.fusion.weight.abs().max().item(), 0)

    def test_zero_init_is_identity_and_learns_after_fusion_update(self):
        module = force.ForceConditioner({"fusion_init": "zero"}, hidden_size=16)
        opt = torch.optim.SGD(module.parameters(), lr=.1)
        first = module(self.state[..., 20:26])
        self.assertEqual(torch.count_nonzero(first).item(), 0)
        (first - 1).square().mean().backward()
        self.assertGreater(module.fusion.weight.grad.abs().max().item(), 0)
        opt.step()
        opt.zero_grad()
        (module(self.state[..., 20:26]) - 1).square().mean().backward()
        self.assertGreater(module.encoder[0].weight.grad.abs().max().item(), 0)

    def test_real_checkpoint_and_optimizer_round_trip(self):
        class Extension(nn.Module):
            def __init__(self):
                super().__init__()
                self.original = nn.Linear(3, 4)
                self.force_conditioner = force.ForceConditioner(hidden_size=16)

        module = Extension()
        baseline = {k: v.clone() for k, v in module.state_dict().items() if k.startswith("original.")}
        load_force_checkpoint(module, baseline, allow_base_checkpoint=True)
        bad = dict(baseline, **{"original.weight": torch.zeros(3, 4)})
        with self.assertRaises(RuntimeError):
            load_force_checkpoint(module, bad, allow_base_checkpoint=True)
        opt = torch.optim.AdamW(module.parameters(), lr=.001)
        module.force_conditioner(self.state[..., 20:26]).square().mean().backward()
        opt.step()
        saved = BytesIO()
        torch.save({"state_dict": module.state_dict(), "optimizer": opt.state_dict()}, saved)
        saved.seek(0)
        restored = torch.load(saved, weights_only=True)
        clone = Extension()
        load_force_checkpoint(clone, restored["state_dict"])
        other_opt = torch.optim.AdamW(clone.parameters(), lr=.001)
        other_opt.load_state_dict(restored["optimizer"])
        torch.testing.assert_close(module.force_conditioner(self.state[..., 20:26]),
                                   clone.force_conditioner(self.state[..., 20:26]), rtol=0, atol=0)
        for left, right in zip(opt.state.values(), other_opt.state.values()):
            for key in left:
                torch.testing.assert_close(left[key], right[key], rtol=0, atol=0)
        # One more identical step checks restored momentum as well as serialization.
        for net, optimizer in ((module, opt), (clone, other_opt)):
            optimizer.zero_grad()
            net.force_conditioner(self.state[..., 20:26]).square().mean().backward()
            optimizer.step()
        for name, value in module.state_dict().items():
            torch.testing.assert_close(value, clone.state_dict()[name], rtol=0, atol=0)

    def test_all_freeze_policies_and_new_module_switches(self):
        from types import SimpleNamespace
        # Use registered stand-in modules to test selection, no VLM allocation.
        for policy in ("original", "frozen", "dit", "last_vlm_block"):
            model = nn.Module()
            for name in ("dit", "state_projector", "action_projector", "action_output_layer",
                         "t_embedder", "t_projector", "sink", "state_projector_choice"):
                setattr(model, name, nn.Linear(2, 2))
            model.vlm = nn.Module()
            model.vlm.model = nn.Module()
            model.vlm.model.language_model = nn.Module()
            model.vlm.model.language_model.layers = nn.ModuleList([nn.Linear(2, 2), nn.Linear(2, 2)])
            model.force_conditioner = force.ForceConditioner(hidden_size=16)
            config = force.ForceEncoderConfig.from_mapping({"enabled": True, "pretrained_policy": policy,
                                                            "train_encoder": False, "train_fusion": True})
            force.apply_force_train_policy(model, config)
            for name, parameter in model.named_parameters():
                if name.startswith("force_conditioner.encoder."):
                    expected = False
                elif name.startswith("force_conditioner.fusion."):
                    expected = True
                elif policy == "original":
                    expected = True
                elif policy == "frozen":
                    expected = False
                elif policy == "dit":
                    expected = name.split(".")[0] in {"dit", "state_projector", "action_projector", "action_output_layer",
                                                         "t_embedder", "t_projector", "sink"}
                else:
                    expected = name.startswith("vlm.model.language_model.layers.1.")
                with self.subTest(policy=policy, name=name):
                    self.assertEqual(parameter.requires_grad, expected)


@unittest.skipUnless(TORCH, "PyTorch unavailable; XR-1 integration numerics not executed")
class ForceXR1Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.model = tiny_model(pretrained_policy="frozen")

    def test_train_forward_action_loss_gradients_and_optimizer_inclusion(self):
        model = self.model.train()
        model.zero_grad(set_to_none=True)
        inputs = batch()
        seen = []
        hooks = [p.register_forward_pre_hook(lambda _, args: seen.append(args[0].detach().clone()))
                 for p in (model.state_projector, model.state_projector_choice)]
        losses = model(copy.deepcopy(inputs), return_loss=True)
        losses["loss"].backward()
        for handle in hooks:
            handle.remove()
        self.assertEqual(len(seen), 2)
        for state in seen:
            self.assertEqual(torch.count_nonzero(state[..., 20:26]).item(), 0)
        for name, parameter in model.named_parameters():
            if name.startswith("force_conditioner."):
                self.assertTrue(parameter.requires_grad)
                self.assertIsNotNone(parameter.grad)
                self.assertTrue(torch.isfinite(parameter.grad).all())
                self.assertGreater(parameter.grad.abs().max().item(), 0)
            else:
                self.assertFalse(parameter.requires_grad)
                self.assertIsNone(parameter.grad)
        # Run the actual runner optimizer builder, without importing Lightning.
        import ast
        from importlib import import_module
        from typing import Any, Dict, Iterator, Tuple
        from torch.optim import Optimizer
        from encoder_support.harness import ROOT
        tree = ast.parse((ROOT / "mibot/models/runner/base_runner.py").read_text())
        runner = next(n for n in tree.body if isinstance(n, ast.ClassDef))
        method = next(n for n in runner.body if isinstance(n, ast.FunctionDef) and n.name == "build_optimizer")
        method.decorator_list = []
        ns = dict(torch=torch, Any=Any, Dict=Dict, Iterator=Iterator, Tuple=Tuple,
                  Optimizer=Optimizer, import_module=import_module)
        exec(compile(ast.Module(body=[method], type_ignores=[]), "base_runner.py", "exec"), ns)
        opt = ns["build_optimizer"]({"type": "torch.optim.AdamW", "params": {"lr": .001}}, model.named_parameters())
        included = {id(p) for group in opt.param_groups for p in group["params"]}
        self.assertEqual(included, {id(p) for p in model.parameters() if p.requires_grad})

    def test_generation_and_training_share_force_representation_and_repeat_order(self):
        model = self.model
        inputs = batch(batch_size=2)
        inputs["state"][1, :, 20:26] *= -1
        observed = []
        handle = model.dit.register_forward_pre_hook(lambda _, args: observed.append(args[-1].detach().clone()))
        residuals = []
        original = model.dit_forward

        def record(*args, **kwargs):
            residuals.append(kwargs["force_condition"].detach().clone())
            return original(*args, **kwargs)

        model.dit_forward = record
        try:
            model.train()
            model(copy.deepcopy(inputs), return_loss=True)
            expected = residuals[0]
            self.assertEqual(expected.shape, (4, 6, 1024))
            model.eval()
            result = model.generate(batch(training=False, batch_size=2) | {"state": inputs["state"]})
            self.assertEqual(result.shape, (2, 30, 60))
            for residual in residuals[1:]:
                torch.testing.assert_close(expected, residual.repeat_interleave(2, dim=0), rtol=0, atol=0)
            self.assertTrue(torch.isfinite(result).all())
        finally:
            model.dit_forward = original
            handle.remove()

    def test_disabled_identity_generic_state_and_zero_fusion_equivalence(self):
        model = self.model.eval()
        config = model.force_encoder_config
        fusion = model.force_conditioner.fusion.weight.detach().clone()
        try:
            model.force_encoder_config = replace(config, enabled=False)
            generic = torch.randn(2, 3, 60)  # disabled path imposes no Doosan shape restriction
            clean, condition = model.prepare_force_conditioning(generic)
            self.assertIs(clean, generic)
            self.assertIsNone(condition)
            inputs = batch(training=False)
            inputs["state"][..., 20:26] = 0
            del inputs["force_normalization"]
            torch.manual_seed(12)
            baseline = model.generate(copy.deepcopy(inputs))
            model.force_encoder_config = config
            with torch.no_grad():
                model.force_conditioner.fusion.weight.zero_()
            torch.manual_seed(12)
            enabled = model.generate(batch(training=False))
            torch.testing.assert_close(enabled, baseline, rtol=0, atol=0)
        finally:
            model.force_encoder_config = config
            with torch.no_grad():
                model.force_conditioner.fusion.weight.copy_(fusion)

    def test_flow_loss_excludes_temporal_padding_and_inactive_channels(self):
        model = self.model
        pred = torch.randn(2, 30, 60, requires_grad=True)
        target = torch.randn_like(pred)
        mask = batch(batch_size=2)["action_mask"]
        mse, freq = model.compute_flow_loss(pred, target, mask, torch.ones_like(pred))
        self.assertEqual(freq.item(), 0)
        changed = pred.detach().clone()
        changed[~mask.bool()] += 1000
        mse2, freq2 = model.compute_flow_loss(changed, target, mask, torch.ones_like(pred))
        torch.testing.assert_close(mse, mse2, rtol=0, atol=0)
        self.assertEqual(freq2.item(), 0)
        (mse + freq).backward()
        self.assertEqual(torch.count_nonzero(pred.grad[~mask.bool()]).item(), 0)

    def test_no_new_keys_when_disabled_and_original_shapes_preserved(self):
        # Existing state keys are compared against a disabled model using the same
        # actual constructor. This is a synthetic test, not a real checkpoint load.
        disabled = tiny_model(enabled=False)
        current = self.model.state_dict()
        self.assertFalse(any(k.startswith("force_conditioner.") for k in disabled.state_dict()))
        self.assertEqual(set(current) - set(disabled.state_dict()),
                         {k for k in current if k.startswith("force_conditioner.")})
        for name, value in disabled.state_dict().items():
            self.assertEqual(value.shape, current[name].shape)


if __name__ == "__main__":
    unittest.main()
