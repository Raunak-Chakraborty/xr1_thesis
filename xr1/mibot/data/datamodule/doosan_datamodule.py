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
                shuffle=True,
                seed=self.sampler_seed,
            )
            shuffle = False

        kwargs = {
            "dataset": self.train_set,
            "batch_size": self.batch_size,
            "shuffle": shuffle,
            "sampler": sampler,
            "num_workers": self.num_workers,
            "collate_fn": self.collate_fn,
            "pin_memory": self.pin_memory,
        }

        if self.num_workers > 0:
            kwargs[
                "prefetch_factor"
            ] = self.prefetch_factor

            kwargs[
                "persistent_workers"
            ] = self.persistent_workers

        return DataLoader(
            **kwargs
        )
