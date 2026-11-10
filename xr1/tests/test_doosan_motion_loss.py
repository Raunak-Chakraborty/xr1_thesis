"""CPU-only loss-weighting regression checks."""
import unittest

import torch

from mibot.utils.doosan_motion_loss import (
    masked_weighted_mean,
    motion_multipliers,
    validate_motion_profile,
)


class TestMotionLoss(unittest.TestCase):

    def setUp(self):
        self.profile = validate_motion_profile({
            'first_5': 2.0,
            'next_5': 1.25,
            'remaining_20': 0.75,
        })

    def test_dimensions_and_window_bands(self):
        factor = motion_multipliers(
            torch.ones((2, 30, 60)),
            prefix_length=0,
            profile=self.profile
        )
        self.assertEqual(tuple(factor.shape), (2, 30, 60))
        self.assertTrue(torch.all(factor[:, :5, :6] == 2.0))
        self.assertTrue(torch.all(factor[:, 5:10, :6] == 1.25))
        self.assertTrue(torch.all(factor[:, 10:, :6] == 0.75))
        self.assertTrue(torch.all(factor[:, :, 6:] == 1.0))

    def test_prefix_retains_original_positions(self):
        factor = motion_multipliers(
            torch.ones((3, 26, 60)),
            prefix_length=4,
            profile=self.profile
        )
        self.assertEqual(factor[0, 0, 0].item(), 2.0)
        self.assertEqual(factor[0, 1, 0].item(), 1.25)
        self.assertEqual(factor[0, 6, 0].item(), 0.75)
        self.assertEqual(factor[0, 0, 6].item(), 1.0)

    def test_opt_out_equals_original_masked_mean(self):
        x = torch.arange(60.0).reshape(2, 5, 6)
        mask = (
            torch.arange(60).reshape(2, 5, 6) % 3 == 0
        )
        self.assertTrue(torch.equal(
            masked_weighted_mean(x, mask),
            x[mask].mean()
        ))

    def test_invalid_elements_do_not_contribute_gradients(self):
        pred = torch.tensor(
            [[[2., 5.], [8., 13.]]],
            requires_grad=True
        )
        mask = torch.tensor(
            [[[1, 1], [0, 0]]],
            dtype=torch.bool
        )
        focus = torch.tensor(
            [[[2., 1.], [9., 9.]]]
        )

        loss = masked_weighted_mean(
            pred.square(), mask, focus
        )
        self.assertAlmostEqual(
            loss.item(), (2.0 * 4 + 25) / 3
        )
        loss.backward()

        self.assertTrue(
            torch.all(pred.grad[:, 1] == 0)
        )

    def test_bad_config_fails_closed(self):
        for bad in (
            {'first_5': 2},
            {
                'first_5': 2.,
                'next_5': 1.,
                'remaining_20': -1.
            },
        ):
            with self.assertRaises(ValueError):
                validate_motion_profile(bad)


if __name__ == '__main__':
    unittest.main()
