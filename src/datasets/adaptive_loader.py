from collections.abc import Mapping, Sequence
from typing import Union, List, Optional
import math

import torch
from torch_geometric.data import Batch, Dataset
from torch_geometric.data.data import BaseData
from torch.utils.data.dataloader import default_collate
import torch.utils.data
from torch_geometric.loader import DataLoader
try:
    from torch_geometric.data import LightningDataset
except ImportError:
    from torch_geometric.data.lightning import LightningDataset


def effective_batch_size(max_size, reference_batch_size, reference_size=20, sampling=False):
    x = reference_batch_size * (reference_size / max_size) ** 2
    return math.floor(1.8 * x) if sampling else math.floor(x)


class AdaptiveCollater:
    def __init__(self, follow_batch, exclude_keys, reference_batch_size):
        """ Copypaste from pyg.loader.Collater + small changes"""
        self.follow_batch = follow_batch
        self.exclude_keys = exclude_keys
        self.reference_bs = reference_batch_size

    def __call__(self, batch):
        # checks the number of node for basedata graphs and slots into appropriate buckets,
        # errors on other options
        elem = batch[0]
        if isinstance(elem, BaseData):
            to_keep = []
            graph_sizes = []

            for e in batch:
                e: BaseData
                graph_sizes.append(e.num_nodes)

            m = len(graph_sizes)
            graph_sizes = torch.Tensor(graph_sizes)
            srted, argsort = torch.sort(graph_sizes)
            random = torch.randint(0, m, size=(1, 1)).item()
            max_size = min(srted.max().item(), srted[random].item() + 5)
            max_size = max(max_size, 9)           # The batch sizes may be huge if the graphs happen to be tiny

            ebs = effective_batch_size(max_size, self.reference_bs)

            max_index = torch.nonzero(srted <= max_size).max().item()
            min_index = max(0, max_index - ebs)
            indices_to_keep = set(argsort[min_index: max_index + 1].tolist())
            if max_index < ebs:
                for index in range(max_index + 1, m):
                    # Check if we could add the graph to the list
                    size = srted[index].item()
                    potential_ebs = effective_batch_size(size, self.reference_bs)
                    if len(indices_to_keep) < potential_ebs:
                        indices_to_keep.add(argsort[index].item())

            for i, e in enumerate(batch):
                e: BaseData
                if i in indices_to_keep:
                    to_keep.append(e)

            new_batch = Batch.from_data_list(to_keep, self.follow_batch, self.exclude_keys)
            return new_batch

        elif True:
            # early exit
            raise NotImplementedError("Only supporting BaseData for now")
        elif isinstance(elem, torch.Tensor):
            return default_collate(batch)
        elif isinstance(elem, float):
            return torch.tensor(batch, dtype=torch.float)
        elif isinstance(elem, int):
            return torch.tensor(batch)
        elif isinstance(elem, str):
            return batch
        elif isinstance(elem, Mapping):
            return {key: self([data[key] for data in batch]) for key in elem}
        elif isinstance(elem, tuple) and hasattr(elem, '_fields'):
            return type(elem)(*(self(s) for s in zip(*batch)))
        elif isinstance(elem, Sequence) and not isinstance(elem, str):
            return [self(s) for s in zip(*batch)]

        raise TypeError(f'DataLoader found invalid type: {type(elem)}')

    def collate(self, batch):  # Deprecated...
        return self(batch)


class AdaptiveDataLoader(torch.utils.data.DataLoader):
    r"""A data loader which merges data objects from a
    :class:`torch_geometric.data.Dataset` into mini-batches, each minibatch being a bucket with num_nodes < some threshold,
    except the last which holds the overflow-graphs. Apart from the bucketing, identical to torch_geometric.loader.DataLoader
    Default bucket_thresholds is [30,50,90], yielding 4 buckets
    Data objects can be either of type :class:`~torch_geometric.data.Data` or
    :class:`~torch_geometric.data.HeteroData`.

    Args:
        dataset (Dataset): The dataset from which to load the data.
        batch_size (int, optional): How many samples per batch to load.
            (default: :obj:`1`)
        shuffle (bool, optional): If set to :obj:`True`, the data will be
            reshuffled at every epoch. (default: :obj:`False`)
        follow_batch (List[str], optional): Creates assignment batch
            vectors for each key in the list. (default: :obj:`None`)
        exclude_keys (List[str], optional): Will exclude each key in the
            list. (default: :obj:`None`)
        **kwargs (optional): Additional arguments of
            :class:`torch.utils.data.DataLoader`.
    """
    def __init__(
            self,
            dataset: Union[Dataset, List[BaseData]],
            batch_size: int = 1,
            reference_batch_size: int = 1,
            shuffle: bool = False,
            follow_batch: Optional[List[str]] = None,
            exclude_keys: Optional[List[str]] = None,
            **kwargs,
    ):

        if 'collate_fn' in kwargs:
            del kwargs['collate_fn']

        # Save for PyTorch Lightning:
        self.follow_batch = follow_batch
        self.exclude_keys = exclude_keys

        super().__init__(
            dataset,
            batch_size,
            shuffle,
            collate_fn=AdaptiveCollater(follow_batch, exclude_keys, reference_batch_size=reference_batch_size),
            **kwargs,
        )


class AdaptiveLightningDataset(LightningDataset):
    r"""Converts a set of :class:`~torch_geometric.data.Dataset` objects into a
    :class:`pytorch_lightning.LightningDataModule` variant, which can be
    automatically used as a :obj:`datamodule` for multi-GPU graph-level
    training via `PyTorch Lightning <https://www.pytorchlightning.ai>`__.
    :class:`LightningDataset` will take care of providing mini-batches via
    :class:`~torch_geometric.loader.DataLoader`.

    .. note::

        Currently only the
        :class:`pytorch_lightning.strategies.SingleDeviceStrategy` and
        :class:`pytorch_lightning.strategies.DDPSpawnStrategy` training
        strategies of `PyTorch Lightning
        <https://pytorch-lightning.readthedocs.io/en/latest/guides/
        speed.html>`__ are supported in order to correctly share data across
        all devices/processes:

        .. code-block::

            import pytorch_lightning as pl
            trainer = pl.Trainer(strategy="ddp_spawn", accelerator="gpu",
                                 devices=4)
            trainer.fit(model, datamodule)

    Args:
        train_dataset (Dataset): The training dataset.
        val_dataset (Dataset, optional): The validation dataset.
            (default: :obj:`None`)
        test_dataset (Dataset, optional): The test dataset.
            (default: :obj:`None`)
        batch_size (int, optional): How many samples per batch to load.
            (default: :obj:`1`)
        num_workers: How many subprocesses to use for data loading.
            :obj:`0` means that the data will be loaded in the main process.
            (default: :obj:`0`)
        **kwargs (optional): Additional arguments of
            :class:`torch_geometric.loader.DataLoader`.
    """
    def __init__(
        self,
        train_dataset: Dataset,
        val_dataset: Optional[Dataset] = None,
        test_dataset: Optional[Dataset] = None,
        batch_size: int = 1,
        reference_batch_size: int = 1,
        num_workers: int = 0,
        **kwargs,
    ):
        self.reference_batch_size = reference_batch_size
        super().__init__(
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            test_dataset=test_dataset,
            # has_val=val_dataset is not None,
            # has_test=test_dataset is not None,
            batch_size=batch_size,
            num_workers=num_workers,
            **kwargs,
        )

        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.test_dataset = test_dataset

    def dataloader(self, dataset: Dataset, shuffle: bool = False, **kwargs) -> AdaptiveDataLoader:
        return AdaptiveDataLoader(dataset, reference_batch_size=self.reference_batch_size,
                                  shuffle=shuffle, **self.kwargs)
