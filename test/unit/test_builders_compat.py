import inspect
import unittest
from unittest.mock import patch

import torch
from torch.optim.lr_scheduler import ExponentialLR, ReduceLROnPlateau, StepLR

from joeynmt import builders
from joeynmt.builders import build_scheduler


class _PlateauWithoutVerbose:
    """
    Stand-in for torch >= 2.2's ReduceLROnPlateau, which no longer accepts
    `verbose`. Local torch 2.1.2 still does, so this is the only way to exercise
    the path that Colab's torch 2.11 takes.
    """

    def __init__(
        self,
        optimizer,
        mode="min",
        factor=0.1,
        patience=10,
        threshold=1e-4,
        threshold_mode="rel",
        cooldown=0,
        min_lr=0,
        eps=1e-8
    ):
        self.optimizer = optimizer
        self.mode = mode
        self.factor = factor
        self.patience = patience

    def step(self, *args, **kwargs):
        pass


class TestBuildersTorchCompat(unittest.TestCase):

    def setUp(self):
        self.parameter = torch.nn.Parameter(torch.zeros(2, 2))
        self.optimizer = torch.optim.SGD([self.parameter], lr=0.1)
        self.cfg = {"scheduling": "plateau", "decrease_factor": 0.5, "patience": 3}

    def test_plateau_builds_on_the_installed_torch(self):
        scheduler, step_at = build_scheduler(
            cfg=self.cfg, optimizer=self.optimizer, scheduler_mode="max"
        )
        self.assertIsInstance(scheduler, ReduceLROnPlateau)
        self.assertEqual(step_at, "validation")
        self.assertEqual(scheduler.factor, 0.5)
        self.assertEqual(scheduler.patience, 3)

    def test_plateau_drops_verbose_when_torch_no_longer_accepts_it(self):
        """The torch >= 2.2 path: `verbose` must not be forwarded."""
        # teeth: the stub really does reject `verbose`, so without the shim in
        # build_scheduler this test would fail with TypeError
        with self.assertRaises(TypeError):
            _PlateauWithoutVerbose(self.optimizer, verbose=False)

        with patch.object(builders, "ReduceLROnPlateau", _PlateauWithoutVerbose):
            scheduler, step_at = build_scheduler(
                cfg=self.cfg, optimizer=self.optimizer, scheduler_mode="max"
            )
        self.assertIsInstance(scheduler, _PlateauWithoutVerbose)
        self.assertEqual(step_at, "validation")
        self.assertEqual(scheduler.factor, 0.5)
        self.assertEqual(scheduler.patience, 3)
        self.assertEqual(scheduler.mode, "max")

    def test_verbose_still_passed_where_supported(self):
        """
        On a torch that accepts `verbose`, the shim must not strip it, i.e.
        behaviour on the local torch 2.1.2 is unchanged. The recorder copies the
        real `__signature__`, so the shim still sees a torch that accepts it.
        """
        if "verbose" not in inspect.signature(ReduceLROnPlateau).parameters:
            self.skipTest("this torch no longer accepts `verbose`")
        captured = {}

        class _Recording(ReduceLROnPlateau):
            __signature__ = inspect.signature(ReduceLROnPlateau)

            def __init__(self, *args, **kwargs):
                captured.update(kwargs)
                super().__init__(*args, **kwargs)

        with patch.object(builders, "ReduceLROnPlateau", _Recording):
            build_scheduler(
                cfg=self.cfg, optimizer=self.optimizer, scheduler_mode="max"
            )
        self.assertIn("verbose", captured)
        self.assertFalse(captured["verbose"])

    def test_other_schedulers_untouched(self):
        for name, cls, step_at in [
            ("decaying", StepLR, "epoch"),
            ("exponential", ExponentialLR, "epoch"),
        ]:
            optimizer = torch.optim.SGD([self.parameter], lr=0.1)
            scheduler, at = build_scheduler(
                cfg={"scheduling": name}, optimizer=optimizer, scheduler_mode="max"
            )
            self.assertIsInstance(scheduler, cls, msg=name)
            self.assertEqual(at, step_at, msg=name)


if __name__ == "__main__":
    unittest.main()
