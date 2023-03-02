import collections
import os
from collections import Counter
import math

import numpy as np
import wandb
import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics import Metric, MeanSquaredError, MetricCollection

from src.utils import NoSyncMetric as Metric, NoSyncMetricCollection as MetricCollection
from src.analysis.rdkit_functions import compute_molecular_metrics, check_stability
from src.utils import NoSyncMAE as MeanAbsoluteError
from src.metrics.metrics_utils import counter_to_tensor, wasserstein1d, total_variation1d


class SamplingMetrics:
    def __init__(self, train_smiles, dataset_infos, test):
        self.dataset_infos = dataset_infos
        self.train_smiles = train_smiles
        self.test = test

    def __call__(self, molecules: list, name, current_epoch, local_rank):
        stability, rdkit_metrics, all_smiles = compute_molecular_metrics(molecules,
                                                                         self.train_smiles,
                                                                         self.dataset_infos)
        if self.test:
            with open(r'final_smiles.txt', 'w') as fp:
                for smiles in all_smiles:
                    # write each item on a new line
                    fp.write("%s\n" % smiles)
                print('All smiles saved')

        stat = self.dataset_infos.statistics['test'] if self.test else self.dataset_infos.statistics['val']

        # Compute statistics
        num_nodes_w1 = number_nodes_distance(molecules, stat.num_nodes)

        atom_types_tv, atom_tv_per_class = atom_types_distance(molecules, stat.atom_types, save_histogram=self.test)

        edge_types_tv, bond_tv_per_class = bond_types_distance(molecules, stat.bond_types, save_histogram=self.test)

        charge_w1, charge_w1_per_class = charge_distance(molecules, stat.charge_types, stat.atom_types,
                                                         self.dataset_infos)

        valency_w1, valency_w1_per_class = valency_distance(molecules, stat.valencies, stat.atom_types,
                                                            self.dataset_infos.atom_encoder)

        bond_lengths_w1, bond_lengths_w1_per_type = bond_length_distance(molecules, stat.bond_lengths, stat.bond_types)

        angles_w1, angles_w1_per_type = angle_distance(molecules, stat.bond_angles, stat.atom_types, stat.valencies,
                                                       atom_decoder=self.dataset_infos.atom_decoder,
                                                       save_histogram=self.test)

        to_log = {'sampling/NumNodesW1': num_nodes_w1, 'sampling/AtomTypesTV': atom_types_tv,
                  'sampling/EdgeTypesTV': edge_types_tv, 'sampling/ChargeW1': charge_w1,
                  'sampling/ValencyW1': valency_w1, 'sampling/BondLengthsW1': bond_lengths_w1,
                  'sampling/AnglesW1': angles_w1}
        print(f"Sampling metrics", {key: round(val, 3) for key, val in to_log.items()})

        for i, atom_type in enumerate(self.dataset_infos.atom_decoder):
            to_log[f'sampling_per_class/{atom_type}_TV'] = atom_tv_per_class[i].item()
            to_log[f'sampling_per_class/{atom_type}_ValencyW1'] = valency_w1_per_class[i].item()
            to_log[f'sampling_per_class/{atom_type}_BondAnglesW1'] = angles_w1_per_type[i].item()
            to_log[f'sampling_per_class/{atom_type}_ChargesW1'] = charge_w1_per_class[i].item()

        for j, bond_type in enumerate(['No bond', 'Single', 'Double', 'Triple', 'Aromatic']):
            to_log[f'sampling_per_class/{bond_type}_TV'] = bond_tv_per_class[j].item()
            if j > 0:
                to_log[f'sampling_per_class/{bond_type}_BondLengthsW1'] = bond_lengths_w1_per_type[j - 1].item()

        if wandb.run:
            wandb.log(to_log)

            valid_unique_molecules = rdkit_metrics[1]
            to_write = []
            for mol in valid_unique_molecules:
                if mol is not None:
                    to_write.append(mol)
            os.makedirs('graphs', exist_ok=True)
            textfile = open(f'graphs/valid_unique_molecules_e{current_epoch}_GR{local_rank}.txt', "w")
            textfile.writelines(to_write)
            textfile.close()
        print("Stability metrics:", stability, "--", rdkit_metrics[0])


def number_nodes_distance(molecules, dataset_counts):
    max_number_nodes = max(dataset_counts.keys())
    reference_n = torch.zeros(max_number_nodes + 1)
    for n, count in dataset_counts.items():
        reference_n[n] = count

    c = Counter()
    for molecule in molecules:
        c[molecule.num_nodes] += 1

    generated_n = counter_to_tensor(c)
    return wasserstein1d(generated_n, reference_n)


def atom_types_distance(molecules, target, save_histogram=False):
    generated_distribution = torch.zeros_like(target)
    for molecule in molecules:
        for atom_type in molecule.atom_types:
            generated_distribution[atom_type] += 1
    if save_histogram:
        np.save('generated_atom_types.npy', generated_distribution.cpu().numpy())
    return total_variation1d(generated_distribution, target)


def bond_types_distance(molecules, target, save_histogram=False):
    generated_distribution = torch.zeros_like(target)
    for molecule in molecules:
        bond_types = molecule.bond_types
        mask = torch.ones_like(bond_types)
        mask = torch.triu(mask, diagonal=1).bool()
        bond_types = bond_types[mask]
        unique_edge_types, counts = torch.unique(bond_types, return_counts=True)
        for type, count in zip(unique_edge_types, counts):
            generated_distribution[type] += count
    if save_histogram:
        np.save('generated_bond_types.npy', generated_distribution.cpu().numpy())
    return total_variation1d(generated_distribution, target)


def charge_distance(molecules, target, atom_types_probabilities, dataset_infos):
    generated_distribution = torch.zeros_like(target)
    for molecule in molecules:
        for atom_type in range(target.shape[0]):
            mask = molecule.atom_types == atom_type
            if mask.sum() > 0:
                at_charges = dataset_infos.one_hot_charges(molecule.charges[mask])
                generated_distribution[atom_type] += at_charges.sum(dim=0)

    s = generated_distribution.sum(dim=1, keepdim=True)
    s[s == 0] = 1
    generated_distribution = generated_distribution / s

    cs_generated = torch.cumsum(generated_distribution, dim=1)
    cs_target = torch.cumsum(target, dim=1)

    w1_per_class = torch.sum(torch.abs(cs_generated - cs_target), dim=1)

    w1 = torch.sum(w1_per_class * atom_types_probabilities).item()

    return w1, w1_per_class


def valency_distance(molecules, target_valencies, atom_types_probabilities, atom_encoder):
    # Build a dict for the generated molecules that is similar to the target one
    num_atom_types = len(atom_types_probabilities)
    generated_valencies = {i: Counter() for i in range(num_atom_types)}
    for molecule in molecules:
        edge_types = molecule.bond_types
        edge_types[edge_types == 4] = 1.5
        valencies = torch.sum(edge_types, dim=0)
        for atom, val in zip(molecule.atom_types, valencies):
            generated_valencies[atom.item()][val.item()] += 1

    # Convert the valencies to a tensor of shape (num_atom_types, max_valency)
    max_valency_target = max(max(vals.keys()) if len(vals) > 0 else -1 for vals in target_valencies.values())
    max_valency_generated = max(max(vals.keys()) if len(vals) > 0 else -1 for vals in generated_valencies.values())
    max_valency = max(max_valency_target, max_valency_generated)

    valencies_target_tensor = torch.zeros(num_atom_types, max_valency + 1)
    for atom_type, valencies in target_valencies.items():
        for valency, count in valencies.items():
            valencies_target_tensor[atom_encoder[atom_type], valency] = count

    valencies_generated_tensor = torch.zeros(num_atom_types, max_valency + 1)
    for atom_type, valencies in generated_valencies.items():
        for valency, count in valencies.items():
            valencies_generated_tensor[atom_type, valency] = count

    # Normalize the distributions
    s1 = torch.sum(valencies_target_tensor, dim=1, keepdim=True)
    s1[s1 == 0] = 1
    valencies_target_tensor = valencies_target_tensor / s1

    s2 = torch.sum(valencies_generated_tensor, dim=1, keepdim=True)
    s2[s2 == 0] = 1
    valencies_generated_tensor = valencies_generated_tensor / s2

    cs_target = torch.cumsum(valencies_target_tensor, dim=1)
    cs_generated = torch.cumsum(valencies_generated_tensor, dim=1)

    w1_per_class = torch.sum(torch.abs(cs_target - cs_generated), dim=1)

    total_w1 = torch.sum(w1_per_class * atom_types_probabilities).item()
    return total_w1, w1_per_class


def bond_length_distance(molecules, target, bond_types_probabilities):
    generated_bond_lenghts = {1: Counter(), 2: Counter(), 3: Counter(), 4: Counter()}
    for molecule in molecules:
        cdists = torch.cdist(molecule.positions.unsqueeze(0),
                             molecule.positions.unsqueeze(0)).squeeze(0)
        for bond_type in range(1, 5):
            edges = torch.nonzero(molecule.bond_types == bond_type)
            bond_distances = cdists[edges[:, 0], edges[:, 1]]
            distances_to_consider = torch.round(bond_distances, decimals=2)
            for d in distances_to_consider:
                generated_bond_lenghts[bond_type][d.item()] += 1

    # Normalizing the bond lenghts
    for bond_type in range(1, 5):
        s = sum(generated_bond_lenghts[bond_type].values())
        if s == 0:
            s = 1
        for d, count in generated_bond_lenghts[bond_type].items():
            generated_bond_lenghts[bond_type][d] = count / s

    # Convert both dictionaries to tensors
    min_generated_length = min(min(d.keys()) if len(d) > 0 else 1e4 for d in generated_bond_lenghts.values())
    min_target_length = min(min(d.keys()) if len(d) > 0 else 1e4 for d in target.values())
    min_length = min(min_generated_length, min_target_length)

    max_generated_length = max(max(bl.keys()) if len(bl) > 0 else -1 for bl in generated_bond_lenghts.values())
    max_target_length = max(max(bl.keys()) if len(bl) > 0 else -1 for bl in target.values())
    max_length = max(max_generated_length, max_target_length)

    num_bins = int((max_length - min_length) * 100) + 1
    generated_bond_lengths = torch.zeros(4, num_bins)
    target_bond_lengths = torch.zeros(4, num_bins)

    for bond_type in range(1, 5):
        for d, count in generated_bond_lenghts[bond_type].items():
            bin = int((d - min_length) * 100)
            generated_bond_lengths[bond_type - 1, bin] = count
        for d, count in target[bond_type].items():
            bin = int((d - min_length) * 100)
            target_bond_lengths[bond_type - 1, bin] = count

    cs_generated = torch.cumsum(generated_bond_lengths, dim=1)
    cs_target = torch.cumsum(target_bond_lengths, dim=1)

    w1_per_class = torch.sum(torch.abs(cs_generated - cs_target), dim=1) / 100    # 100 because of bin size
    weighted = w1_per_class * bond_types_probabilities[1:]
    return torch.sum(weighted).item(), w1_per_class


def angle_distance(molecules, target_angles, atom_types_probabilities, valencies, atom_decoder, save_histogram: bool):
    num_atom_types = len(atom_types_probabilities)
    generated_angles = torch.zeros(num_atom_types, 180 * 10 + 1)
    for molecule in molecules:
        adj = molecule.bond_types
        pos = molecule.positions
        for atom in range(adj.shape[0]):
            p_a = pos[atom]
            neighbors = torch.nonzero(adj[atom]).squeeze(1)
            for i in range(len(neighbors)):
                p_i = pos[neighbors[i]]
                for j in range(i + 1, len(neighbors)):
                    p_j = pos[neighbors[j]]
                    v1 = p_i - p_a
                    v2 = p_j - p_a
                    assert not torch.isnan(v1).any()
                    assert not torch.isnan(v2).any()
                    prod = torch.dot(v1 / (torch.norm(v1) + 1e-6), v2 / (torch.norm(v2) + 1e-6))
                    if prod > 1:
                        print(f"Invalid angle {i} {j} -- {prod} -- {v1 / (torch.norm(v1) + 1e-6)} --"
                              f" {v2 / (torch.norm(v2) + 1e-6)}")
                    prod.clamp(min=0, max=1)
                    angle = torch.acos(prod)
                    if torch.isnan(angle).any():
                        print(f"Nan obtained in angle {i} {j} -- {prod} -- {v1 / (torch.norm(v1) + 1e-6)} --"
                              f" {v2 / (torch.norm(v2) + 1e-6)}")
                    else:
                        bin = int(torch.round(angle * 180 / math.pi, decimals=1).item() * 10)
                        generated_angles[molecule.atom_types[atom], bin] += 1


    s = torch.sum(generated_angles, dim=1, keepdim=True)
    s[s == 0] = 1
    generated_angles = generated_angles / s
    if save_histogram:
        np.save('generated_angles_historgram.npy', generated_angles.numpy())

    if type(target_angles) in [np.array, np.ndarray]:
        target_angles = torch.from_numpy(target_angles).float()

    cs_generated = torch.cumsum(generated_angles, dim=1)
    cs_target = torch.cumsum(target_angles, dim=1)

    w1_per_type = torch.sum(torch.abs(cs_generated - cs_target), dim=1) / 10

    # The atoms that have a valency less than 2 should not matter
    valency_weight = torch.zeros(len(w1_per_type), device=w1_per_type.device)
    for i in range(len(w1_per_type)):
        valency_weight[i] = 1 - valencies[atom_decoder[i]][0] - valencies[atom_decoder[i]][1]

    weighted = w1_per_type * atom_types_probabilities * valency_weight
    return (torch.sum(weighted) / (torch.sum(atom_types_probabilities * valency_weight) + 1e-5)).item(), w1_per_type



class HistogramsMAE(MeanAbsoluteError):
    def __init__(self, target_histogram, **kwargs):
        """ Compute the distance between histograms. """
        super().__init__(**kwargs)
        assert (target_histogram.sum() - 1).abs() < 1e-3
        self.target_histogram = target_histogram

    def update(self, pred):
        pred = pred / pred.sum()
        target = self.target_histogram.to(pred.device)
        super().update(pred, target)


class CEPerClass(Metric):
    full_state_update = True

    def __init__(self, class_id):
        super().__init__()
        self.class_id = class_id
        self.add_state('total_ce', default=torch.tensor(0.), dist_reduce_fx="sum")
        self.add_state('total_samples', default=torch.tensor(0.), dist_reduce_fx="sum")
        self.softmax = torch.nn.Softmax(dim=-1)
        self.binary_cross_entropy = torch.nn.BCELoss(reduction='sum')

    def update(self, preds: Tensor, target: Tensor) -> None:
        """Update state with predictions and targets.
        Args:
            preds: Predictions from model   (bs, n, d) or (bs, n, n, d)
            target: Ground truth values     (bs, n, d) or (bs, n, n, d)
        """
        target = target.reshape(-1, target.shape[-1])
        mask = (target != 0.).any(dim=-1)

        prob = self.softmax(preds)[..., self.class_id]
        prob = prob.flatten()[mask]

        target = target[:, self.class_id]
        target = target[mask]

        output = self.binary_cross_entropy(prob, target)
        self.total_ce += output
        self.total_samples += prob.numel()

    def compute(self):
        return self.total_ce / self.total_samples



class MeanNumberEdge(Metric):
    full_state_update = True

    def __init__(self):
        super().__init__()
        self.add_state('total_edge', default=torch.tensor(0.), dist_reduce_fx="sum")
        self.add_state('total_samples', default=torch.tensor(0.), dist_reduce_fx="sum")

    def update(self, molecules, weight=1.0) -> None:
        for molecule in molecules:
            _, edge_types = molecule
            triu_edge_types = torch.triu(edge_types, diagonal=1)
            bonds = torch.nonzero(triu_edge_types)
            self.total_edge += len(bonds)
        self.total_samples += len(molecules)

    def compute(self):
        return self.total_edge / self.total_samples



class HydrogenCE(CEPerClass):
    def __init__(self, i):
        super().__init__(i)


class CarbonCE(CEPerClass):
    def __init__(self, i):
        super().__init__(i)


class NitroCE(CEPerClass):
    def __init__(self, i):
        super().__init__(i)

class AlCE(CEPerClass):
    def __init__(self, i):
        super().__init__(i)


class OxyCE(CEPerClass):
    def __init__(self, i):
        super().__init__(i)


class FluorCE(CEPerClass):
    def __init__(self, i):
        super().__init__(i)


class BoronCE(CEPerClass):
    def __init__(self, i):
        super().__init__(i)


class BrCE(CEPerClass):
    def __init__(self, i):
        super().__init__(i)


class ClCE(CEPerClass):
    def __init__(self, i):
        super().__init__(i)


class AsCE(CEPerClass):
    def __init__(self, i):
        super().__init__(i)

class IodineCE(CEPerClass):
    def __init__(self, i):
        super().__init__(i)


class PhosphorusCE(CEPerClass):
    def __init__(self, i):
        super().__init__(i)


class SulfurCE(CEPerClass):
    def __init__(self, i):
        super().__init__(i)


class HgCE(CEPerClass):
    def __init__(self, i):
        super().__init__(i)


class BiCE(CEPerClass):
    def __init__(self, i):
        super().__init__(i)


class SeCE(CEPerClass):
    def __init__(self, i):
        super().__init__(i)


class SiCE(CEPerClass):
    def __init__(self, i):
        super().__init__(i)


class NoBondCE(CEPerClass):
    def __init__(self, i):
        super().__init__(i)


class SingleCE(CEPerClass):
    def __init__(self, i):
        super().__init__(i)


class DoubleCE(CEPerClass):
    def __init__(self, i):
        super().__init__(i)


class TripleCE(CEPerClass):
    def __init__(self, i):
        super().__init__(i)


class AromaticCE(CEPerClass):
    def __init__(self, i):
        super().__init__(i)


class AtomMetricsCE(MetricCollection):
    def __init__(self, dataset_infos):
        atom_decoder = dataset_infos.atom_decoder

        class_dict = {'H': HydrogenCE, 'B': BoronCE, 'C': CarbonCE, 'N': NitroCE, 'O': OxyCE, 'F': FluorCE,
                      'Al': AlCE, 'Si': SiCE, 'P': PhosphorusCE, 'S': SulfurCE, 'Cl': ClCE, 'As': AsCE,
                      'Br': BrCE,  'I': IodineCE, 'Hg': HgCE, 'Bi': BiCE}

        metrics_list = []
        for i, atom_type in enumerate(atom_decoder):
            metrics_list.append(class_dict[atom_type](i))
        super().__init__(metrics_list)


class BondMetricsCE(MetricCollection):
    def __init__(self):
        ce_no_bond = NoBondCE(0)
        ce_SI = SingleCE(1)
        ce_DO = DoubleCE(2)
        ce_TR = TripleCE(3)
        ce_AR = AromaticCE(4)
        super().__init__([ce_no_bond, ce_SI, ce_DO, ce_TR, ce_AR])


class TrainMolecularMetrics(nn.Module):
    def __init__(self, dataset_infos):
        super().__init__()
        self.train_atom_metrics = AtomMetricsCE(dataset_infos=dataset_infos)
        self.train_bond_metrics = BondMetricsCE()

    def forward(self, masked_pred, masked_true, log: bool):
        self.train_atom_metrics(masked_pred.X, masked_true.X)
        self.train_bond_metrics(masked_pred.E, masked_true.E)
        if not log:
            return

        to_log = {}
        for key, val in self.train_atom_metrics.compute().items():
            to_log['train/' + key] = val.item()
        for key, val in self.train_bond_metrics.compute().items():
            to_log['train/' + key] = val.item()
        if wandb.run:
            wandb.log(to_log, commit=False)
        return to_log

    def reset(self):
        for metric in [self.train_atom_metrics, self.train_bond_metrics]:
            metric.reset()

    def log_epoch_metrics(self, current_epoch, local_rank):
        epoch_atom_metrics = self.train_atom_metrics.compute()
        epoch_bond_metrics = self.train_bond_metrics.compute()

        to_log = {}
        for key, val in epoch_atom_metrics.items():
            to_log['train_epoch/' + key] = val.item()
        for key, val in epoch_bond_metrics.items():
            to_log['train_epoch/' + key] = val.item()

        if wandb.run:
            wandb.log(to_log, commit=False)

        for key, val in epoch_atom_metrics.items():
            epoch_atom_metrics[key] = round(val.item(), 3)
        for key, val in epoch_bond_metrics.items():
            epoch_bond_metrics[key] = round(val.item(), 3)
        print(f"Epoch {current_epoch} on rank {local_rank}: {epoch_atom_metrics} -- {epoch_bond_metrics}")

        return to_log



if __name__ == '__main__':
    # Do not move these imports, the order seems to matter
    from rdkit import Chem
    import torch
    import pytorch_lightning as pl
    import torch_geometric

    import hydra
    import omegaconf

    from src.datasets import qm9_dataset, geom_dataset
    from src.utils import setup_wandb
    from src.analysis.rdkit_functions import Molecule

    @hydra.main(version_base='1.3', config_path='../../configs', config_name='config')
    def main(cfg: omegaconf.DictConfig):
        dataset_config = cfg.dataset
        pl.seed_everything(cfg.train.seed)

        assert cfg.train.batch_size == 1
        setup_wandb(cfg)

        if dataset_config.name in ['qm9', "geom"]:
            if dataset_config.name == 'qm9':
                datamodule = qm9_dataset.QM9DataModule(cfg)
                dataset_infos = qm9_dataset.QM9infos(datamodule=datamodule, cfg=cfg)

            else:
                datamodule = geom_dataset.GeomDataModule(cfg)
                dataset_infos = geom_dataset.GeomInfos(datamodule=datamodule, cfg=cfg)

        else:
            raise NotImplementedError("Unknown dataset {}".format(cfg["dataset"]))

        train_smiles = list(datamodule.train_dataloader().dataset.smiles)
        sampling_metrics = SamplingMetrics(train_smiles=train_smiles, dataset_infos=dataset_infos, test=True)

        all_train_samples = []
        loader = datamodule.train_dataloader()
        # c = Counter()
        for i, data in enumerate(loader):
            if data.edge_index.numel() == 0:
                print("Data without edges")
                continue

            atom_types = data.x
            bonds = torch_geometric.utils.to_dense_adj(edge_index=data.edge_index, edge_attr=data.edge_attr,
                                                       max_num_nodes=len(atom_types))[0]
            charges = data.charges
            pos = data.pos
            mol = Molecule(atom_types=atom_types, bond_types=bonds, charges=charges, positions=pos,
                           atom_decoder=dataset_infos.atom_decoder)
            all_train_samples.append(mol)

        sampling_metrics(molecules=all_train_samples, name='train_set', current_epoch=-1, local_rank=0)

    main()
