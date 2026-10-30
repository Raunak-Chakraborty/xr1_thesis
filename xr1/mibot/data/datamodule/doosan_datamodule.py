"""Lightning DataModule for the Doosan XR-1 thesis dataset.

This keeps Xiaomi's original BaseDataModule unchanged while replacing
the humanoid-specific JsonDataset with DoosanJsonDataset.

In single-process use the DataLoader shuffles normally.

When torch.distributed is already initialized, the loader uses the
same DistributedSampler pattern as Xiaomi's BaseDataModule.
"""

from __future__ import annotations

from copy import deepcopy

import torch.distributed as dist
import inspect
import torch
from mmengine import Config, DATASETS
from torch.utils.data import (
    DataLoader,
    DistributedSampler,
)

from mibot.data.datamodule.base_datamodule import (
    BaseDataModule,
)
from mibot.data.datasets.doosan_json_dataset import (
    DoosanJsonDataset,
)
from mibot.utils.doosan_replay import RankOffsetSampler, seed_worker


class ReplayDataLoader(DataLoader):
    """Report the original epoch budget to Lightning's restored batch counters.

    Iteration yields only the remaining suffix. Lightning 2.5 restores absolute
    epoch progress; reporting suffix length would terminate the epoch too early
    on a late resume. Sampler length still exposes the remaining sample count.
    """
    def __len__(self):
        if isinstance(self.sampler, RankOffsetSampler):
            return len(self.sampler.base) // self.batch_size
        return super().__len__()


@DATASETS.register_module()
class DoosanDataModule(BaseDataModule):
    """Training data module for single-arm Doosan XR-1."""

    def __init__(
        self,
        params: Config,
    ) -> None:
        super().__init__(params)

        data = params.train_datasets

        self.num_workers = int(
            data.get(
                "num_workers",
                8,
            )
        )

        self.prefetch_factor = int(
            data.get(
                "prefetch_factor",
                4,
            )
        )

        self.pin_memory = bool(
            data.get(
                "pin_memory",
                True,
            )
        )

        self.persistent_workers = bool(
            data.get(
                "persistent_workers",
                self.num_workers > 0,
            )
        )

        self.deterministic_replay = bool(data.get("deterministic_replay", False))
        self.resume_step = int(data.get("resume_step", 0))
        if self.resume_step < 0:
            raise ValueError("Negative resume step")
        if self.deterministic_replay and self.num_workers not in (0, 4, 8):
            raise ValueError("Deterministic replay supports workers=0,4,8")
        if self.deterministic_replay and data.get("in_order", True) is not True:
            raise ValueError("Deterministic replay requires in_order=True")
        if self.deterministic_replay:
            self.collate_fn.strict_samples = True
            self.collate_fn.expected_views = 2
        self.sampler_seed = int(
            data.get(
                "sampler_seed",
                42,
            )
        )

        if self.num_workers < 0:
            raise ValueError(
                "num_workers cannot be negative"
            )

        if self.prefetch_factor <= 0:
            raise ValueError(
                "prefetch_factor must be positive"
            )

        if self.num_workers == 0:
            # PyTorch does not allow persistent worker
            # processes when no workers exist.
            self.persistent_workers = False

    def setup(
        self,
        stage=None,
    ) -> None:
        if (
            stage in (None, "fit")
            and self.train_set is None
        ):
            self.train_set = DoosanJsonDataset(
                deepcopy(
                    self.params
                )
            )

    def train_dataloader(
        self,
    ) -> DataLoader:
        if self.train_set is None:
            self.setup("fit")

        sampler = None
        shuffle = True

        if (
            dist.is_available()
            and dist.is_initialized()
        ):
            sampler = DistributedSampler(
                self.train_set,
                num_replicas=dist.get_world_size(),
                rank=dist.get_rank(),
                shuffle=True,
                seed=self.sampler_seed,
            )
            shuffle = False

        if self.deterministic_replay:
            if not (dist.is_available() and dist.is_initialized()):
                raise RuntimeError("Deterministic DDP sampler requires initialized distributed process group")
            if not isinstance(sampler, DistributedSampler):
                raise RuntimeError("Expected DistributedSampler before offset wrapper")
            horizon = int(self.params.max_steps)
            if len(self.train_set) != horizon * self.batch_size * dist.get_world_size():
                raise RuntimeError("Dataset sample length does not match frozen replay horizon")
            offset = self.resume_step * self.batch_size
            sampler = RankOffsetSampler(sampler, offset, batch_size=self.batch_size)
            shuffle = False
            print("XR1_PRODUCTION_SAMPLER=PASS rank=" + str(dist.get_rank()) +
                  " offset=" + str(offset) + " horizon=" + str(horizon), flush=True)
        kwargs = {
            "dataset": self.train_set,
            "batch_size": self.batch_size,
            "shuffle": shuffle,
            "sampler": sampler,
            "num_workers": self.num_workers,
            "collate_fn": self.collate_fn,
            "pin_memory": self.pin_memory,
            "drop_last": False,
            "worker_init_fn": seed_worker,
            # Iterator construction must not consume the model's RNG stream.
            "generator": torch.Generator().manual_seed(self.sampler_seed),
        }

        if "in_order" in inspect.signature(DataLoader).parameters:
            kwargs["in_order"] = True
        elif self.deterministic_replay and self.num_workers:
            raise RuntimeError("Worker replay requires DataLoader with explicit in_order support")

        if self.num_workers > 0:
            kwargs[
                "prefetch_factor"
            ] = self.prefetch_factor

            kwargs[
                "persistent_workers"
            ] = self.persistent_workers
            kwargs["timeout"] = int(self.params.train_datasets.get("worker_timeout", 180))

        return ReplayDataLoader(
            **kwargs
        )
