from torch.utils.data import Subset
from typing import Optional

import torch
from midi.datasets.adaptive_loader import AdaptiveLightningDataset
try:
    from torch_geometric.data import LightningDataset
except ImportError:
    from torch_geometric.data.lightning import LightningDataset
from midi.diffusion.distributions import DistributionNodes

def maybe_subset(ds,random_subset:Optional[float]=None,split=None)->torch.utils.data.Dataset:
    if random_subset is None or split in {"test", "val"}:
        return ds
    else:
        idx = torch.randperm(len(ds))[:int(random_subset * len(ds))]
        return Subset(ds, idx)



class Mixin:
    def __getitem__(self, idx):
        return self.dataloaders['train'][idx]

    def node_counts(self, max_nodes_possible=300):
        all_counts = torch.zeros(max_nodes_possible)
        for split in ['train', 'val', 'test']:
            for i, batch in enumerate(self.dataloaders[split]):
                for data in batch:
                    if data is None:
                        continue
                    unique, counts = torch.unique(data.batch, return_counts=True)
                    for count in counts:
                        all_counts[count] += 1
        max_index = max(all_counts.nonzero())
        all_counts = all_counts[:max_index + 1]
        all_counts = all_counts / all_counts.sum()
        return all_counts

    def node_types(self):
        num_classes = None
        for batch in self.dataloaders['train']:
            for data in batch:
                num_classes = data.x.shape[1]
                break
            break

        counts = torch.zeros(num_classes)

        for split in ['train', 'val', 'test']:
            for i, batch in enumerate(self.dataloaders[split]):
                for data in batch:
                    if data is None:
                        continue
                    counts += data.x.sum(dim=0)

        counts = counts / counts.sum()
        return counts

    def edge_counts(self):
        num_classes = 5

        d = torch.zeros(num_classes)

        for split in ['train', 'val', 'test']:
            for i, batch in enumerate(self.dataloaders[split]):
                for data in batch:
                    if data is None:
                        continue
                    unique, counts = torch.unique(data.batch, return_counts=True)

                    all_pairs = 0
                    for count in counts:
                        all_pairs += count * (count - 1)

                    num_edges = data.edge_index.shape[1]
                    num_non_edges = all_pairs - num_edges
                    edge_types = data.edge_attr.sum(dim=0)
                    assert num_non_edges >= 0
                    d[0] += num_non_edges
                    d[1:] += edge_types[1:]

        d = d / d.sum()
        return d

    def valency_count(self, max_n_nodes):
        valencies = torch.zeros(3 * max_n_nodes - 2)  # Max valency possible if everything is connected

        multiplier = torch.Tensor([0, 1, 2, 3, 1.5])

        for split in ['train', 'val', 'test']:
            for i, batch in enumerate(self.dataloaders[split]):
                for data in batch:
                    if data is None:
                        continue

                    n = data.x.shape[0]

                    for atom in range(n):
                        edges = data.edge_attr[data.edge_index[0] == atom]
                        edges_total = edges.sum(dim=0)
                        valency = (edges_total * multiplier).sum()
                        valencies[valency.long().item()] += 1
        valencies = valencies / valencies.sum()
        return valencies



class AbstractDataModule(Mixin, LightningDataset):
    def __init__(self, cfg, train_dataset, val_dataset, test_dataset):
        super().__init__(train_dataset, val_dataset, test_dataset, batch_size=cfg.train.batch_size,
                         num_workers=cfg.train.num_workers, shuffle='debug' not in cfg.general.name,
                         pin_memory=getattr(cfg.dataset, "pin_memory", False))
        self.cfg = cfg


class AbstractAdaptiveDataModule(Mixin, AdaptiveLightningDataset):
    def __init__(self, cfg, train_dataset, val_dataset, test_dataset):
        super().__init__(train_dataset, val_dataset, test_dataset, batch_size=cfg.train.batch_size,
                         reference_batch_size=cfg.train.reference_batch_size,
                         num_workers=cfg.train.num_workers, shuffle='debug' not in cfg.general.name,
                         pin_memory=getattr(cfg.dataset, "pin_memory", False))
        self.cfg = cfg


class AbstractDatasetInfos:
    def complete_infos(self, statistics, atom_encoder):
        self.atom_decoder = [key for key in atom_encoder.keys()]
        self.num_atom_types = len(self.atom_decoder)

        # Train + val + test for n_nodes
        train_n_nodes = statistics['train'].num_nodes
        val_n_nodes = statistics['val'].num_nodes
        test_n_nodes = statistics['test'].num_nodes
        max_n_nodes = max(max(train_n_nodes.keys()), max(val_n_nodes.keys()), max(test_n_nodes.keys()))
        n_nodes = torch.zeros(max_n_nodes + 1, dtype=torch.long)
        for c in [train_n_nodes, val_n_nodes, test_n_nodes]:
            for key, value in c.items():
                n_nodes[key] += value

        self.n_nodes = n_nodes / n_nodes.sum()
        self.atom_types = statistics['train'].atom_types
        self.edge_types = statistics['train'].bond_types
        self.charges_types = statistics['train'].charge_types
        self.charges_marginals = (self.charges_types * self.atom_types[:, None]).sum(dim=0)
        self.valency_distribution = statistics['train'].valencies
        self.max_n_nodes = len(n_nodes) - 1
        self.nodes_dist = DistributionNodes(n_nodes)
