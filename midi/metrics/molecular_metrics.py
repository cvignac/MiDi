from rdkit import Chem
import os
from collections import Counter
import math

import torch
from torch import Tensor
import torch.nn as nn
import numpy as np
import wandb
from torchmetrics import MeanMetric, MaxMetric

from midi.utils import NoSyncMetric as Metric, NoSyncMetricCollection as MetricCollection
from midi.analysis.rdkit_functions import check_stability
from midi.utils import NoSyncMAE as MeanAbsoluteError
from midi.metrics.metrics_utils import counter_to_tensor, wasserstein1d, total_variation1d


class SamplingMetrics(nn.Module):
    def __init__(self, train_smiles, dataset_infos, test):
        super().__init__()
        self.dataset_infos = dataset_infos
        self.atom_decoder = dataset_infos.atom_decoder

        self.train_smiles = train_smiles
        self.test = test

        self.atom_stable = MeanMetric()
        self.mol_stable = MeanMetric()

        # Retrieve dataset smiles only for qm9 currently.
        self.train_smiles = set(train_smiles)
        self.validity_metric = MeanMetric()
        self.uniqueness = MeanMetric()
        self.novelty = MeanMetric()
        self.mean_components = MeanMetric()
        self.max_components = MaxMetric()
        self.num_nodes_w1 = MeanMetric()
        self.atom_types_tv = MeanMetric()
        self.edge_types_tv = MeanMetric()
        self.charge_w1 = MeanMetric()
        self.valency_w1 = MeanMetric()
        self.bond_lengths_w1 = MeanMetric()
        self.angles_w1 = MeanMetric()

    def reset(self):
        for metric in [self.atom_stable, self.mol_stable, self.validity_metric, self.uniqueness,
                       self.novelty, self.mean_components, self.max_components, self.num_nodes_w1,
                       self.atom_types_tv, self.edge_types_tv, self.charge_w1, self.valency_w1,
                       self.bond_lengths_w1, self.angles_w1]:
            metric.reset()

    def compute_validity(self, generated):
        """ generated: list of couples (positions, atom_types)"""
        valid = []
        num_components = []
        all_smiles = []
        error_message = Counter()
        for mol in generated:
            rdmol = mol.rdkit_mol
            if rdmol is not None:
                try:
                    mol_frags = Chem.rdmolops.GetMolFrags(rdmol, asMols=True, sanitizeFrags=False)
                    num_components.append(len(mol_frags))
                    if len(mol_frags) > 1:
                        error_message[4] += 1
                    largest_mol = max(mol_frags, default=mol, key=lambda m: m.GetNumAtoms())
                    Chem.SanitizeMol(largest_mol)
                    smiles = Chem.MolToSmiles(largest_mol)
                    valid.append(smiles)
                    all_smiles.append(smiles)
                except Chem.rdchem.AtomValenceException:
                    error_message[1] += 1
                    # print("Valence error in GetmolFrags")
                except Chem.rdchem.KekulizeException:
                    error_message[2] += 1
                    # print("Can't kekulize molecule")
                except Chem.rdchem.AtomKekulizeException or ValueError:
                    error_message[3] += 1
        print(f"Error messages: AtomValence {error_message[1]}, Kekulize {error_message[2]}, other {error_message[3]}, "
              f" -- No error {len(generated) - sum(error_message.values())} / {len(generated)}")
        self.validity_metric.update(value=len(valid) / len(generated), weight=len(generated))
        num_components = torch.tensor(num_components, device=self.mean_components.device)
        self.mean_components.update(num_components)
        self.max_components.update(num_components)
        not_connected = 100.0 * error_message[4] / len(generated)
        connected_components = 100.0 - not_connected
        return valid, connected_components, all_smiles, error_message


    def evaluate(self, generated, local_rank):
        """ generated: list of pairs (positions: n x 3, atom_types: n [int])
            the positions and atom types should already be masked. """
        # Validity
        valid, connected_components, all_smiles, error_message = self.compute_validity(generated)

        validity = self.validity_metric.compute()
        uniqueness, novelty = 0, 0
        mean_components = self.mean_components.compute()
        max_components = self.max_components.compute()

        # Uniqueness
        if len(valid) > 0:
            unique = list(set(valid))
            self.uniqueness.update(value=len(unique) / len(valid), weight=len(valid))
            uniqueness = self.uniqueness.compute()

            if self.train_smiles is not None:
                novel = []
                for smiles in unique:
                    if smiles not in self.train_smiles:
                        novel.append(smiles)
                self.novelty.update(value=len(novel) / len(unique), weight=len(unique))
            novelty = self.novelty.compute()

        if local_rank == 0:
            num_molecules = int(self.validity_metric.weight.item())
            print(f"Validity over {num_molecules} molecules:"
                  f" {validity * 100 :.2f}%")
            print(f"Number of connected components of {num_molecules} molecules: "
                  f"mean:{mean_components:.2f} max:{max_components:.2f}")
            print(f"Connected components of {num_molecules} molecules: "
                  f"{connected_components:.2f}")
            print(f"Uniqueness: {uniqueness * 100 :.2f}% WARNING: do not trust this metric on multi-gpu")
            print(f"Novelty: {novelty * 100 :.2f}%")

        if wandb.run:
            dic = {'Validity': validity,
                   'Uniqueness': uniqueness,
                   'Novelty': novelty,
                   'Connected_Components': connected_components,
                   'nc_mu': mean_components,
                   'nc_max': max_components}
            wandb.log(dic, commit=False)
        return all_smiles

    def __call__(self, molecules: list, name, current_epoch, local_rank):
        # Atom and molecule stability
        if not self.dataset_infos.remove_h:
            print(f'Analyzing molecule stability on {local_rank}...')
            for i, mol in enumerate(molecules):
                mol_stable, at_stable, num_bonds = check_stability(mol, self.dataset_infos)
                self.mol_stable.update(value=mol_stable)
                self.atom_stable.update(value=at_stable / num_bonds, weight=num_bonds)

            stability_dict = {'mol_stable': self.mol_stable.compute().item(),
                              'atm_stable': self.atom_stable.compute().item()}
            if local_rank == 0:
                print("Stability metrics:", stability_dict)
                if wandb.run:
                    wandb.log(stability_dict, commit=False)

        # Validity, uniqueness, novelty
        all_generated_smiles = self.evaluate(molecules, local_rank=local_rank)
        # Save in any case in the graphs folder
        os.makedirs('graphs', exist_ok=True)
        textfile = open(f'graphs/valid_unique_molecules_e{current_epoch}_GR{local_rank}.txt', "w")
        textfile.writelines(all_generated_smiles)
        textfile.close()
        # Save in the root folder if test_model
        if self.test:
            filename = f'final_smiles_GR{local_rank}_{0}.txt'
            for i in range(2, 10):
                if os.path.exists(filename):
                    filename = f'final_smiles_GR{local_rank}_{i}.txt'
                else:
                    break
            with open(filename, 'w') as fp:
                for smiles in all_generated_smiles:
                    # write each item on a new line
                    fp.write("%s\n" % smiles)
                print(f'All smiles saved on rank {local_rank}')

        # Compute statistics
        stat = self.dataset_infos.statistics['test'] if self.test else self.dataset_infos.statistics['val']

        self.num_nodes_w1(number_nodes_distance(molecules, stat.num_nodes))

        atom_types_tv, atom_tv_per_class = atom_types_distance(molecules, stat.atom_types, save_histogram=self.test)
        self.atom_types_tv(atom_types_tv)
        edge_types_tv, bond_tv_per_class, sparsity_level = bond_types_distance(molecules,
                                                                               stat.bond_types,
                                                                               save_histogram=self.test)
        print(f"Sparsity level: {int(100 * sparsity_level)} %")
        self.edge_types_tv(edge_types_tv)
        charge_w1, charge_w1_per_class = charge_distance(molecules, stat.charge_types, stat.atom_types,
                                                         self.dataset_infos)
        self.charge_w1(charge_w1)
        valency_w1, valency_w1_per_class = valency_distance(molecules, stat.valencies, stat.atom_types,
                                                            self.dataset_infos.atom_encoder)
        self.valency_w1(valency_w1)
        bond_lengths_w1, bond_lengths_w1_per_type = bond_length_distance(molecules, stat.bond_lengths, stat.bond_types)
        self.bond_lengths_w1(bond_lengths_w1)
        if sparsity_level < 0.7:
            print(f"Too many edges, skipping angle distance computation.")
            angles_w1 = 0
            angles_w1_per_type = [-1] *  len(self.dataset_infos.atom_decoder)
        else:
            angles_w1, angles_w1_per_type = angle_distance(molecules, stat.bond_angles, stat.atom_types, stat.valencies,
                                                           atom_decoder=self.dataset_infos.atom_decoder,
                                                           save_histogram=self.test)
        self.angles_w1(angles_w1)
        to_log = {'sampling/NumNodesW1': self.num_nodes_w1.compute(),
                  'sampling/AtomTypesTV': self.atom_types_tv.compute(),
                  'sampling/EdgeTypesTV': self.edge_types_tv.compute(),
                  'sampling/ChargeW1': self.charge_w1.compute(),
                  'sampling/ValencyW1': self.valency_w1.compute(),
                  'sampling/BondLengthsW1': self.bond_lengths_w1.compute(),
                  'sampling/AnglesW1': self.angles_w1.compute()}
        print(f"Sampling metrics", {key: round(val.item(), 3) for key, val in to_log.items()}, "on", local_rank)

        for i, atom_type in enumerate(self.dataset_infos.atom_decoder):
            to_log[f'sampling_per_class/{atom_type}_TV'] = atom_tv_per_class[i].item()
            to_log[f'sampling_per_class/{atom_type}_ValencyW1'] = valency_w1_per_class[i].item()
            to_log[f'sampling_per_class/{atom_type}_BondAnglesW1'] = angles_w1_per_type[i].item()\
                if angles_w1_per_type[i] != -1 else -1
            to_log[f'sampling_per_class/{atom_type}_ChargesW1'] = charge_w1_per_class[i].item()

        for j, bond_type in enumerate(['No bond', 'Single', 'Double', 'Triple', 'Aromatic']):
            to_log[f'sampling_per_class/{bond_type}_TV'] = bond_tv_per_class[j].item()
            if j > 0:
                to_log[f'sampling_per_class/{bond_type}_BondLengthsW1'] = bond_lengths_w1_per_type[j - 1].item()

        if wandb.run:
            wandb.log(to_log, commit=False)
        print(f"Sampling metrics done on {local_rank}.")
        self.reset()

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
    device = molecules[0].bond_types.device
    generated_distribution = torch.zeros_like(target).to(device)
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
    sparsity_level = generated_distribution[0] / torch.sum(generated_distribution)
    tv, tv_per_class = total_variation1d(generated_distribution, target.to(device))
    return tv, tv_per_class, sparsity_level


def charge_distance(molecules, target, atom_types_probabilities, dataset_infos):
    device = molecules[0].bond_types.device
    generated_distribution = torch.zeros_like(target).to(device)
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
    cs_target = torch.cumsum(target, dim=1).to(device)

    w1_per_class = torch.sum(torch.abs(cs_generated - cs_target), dim=1)

    w1 = torch.sum(w1_per_class * atom_types_probabilities.to(device)).item()

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
        # self.train_atom_metrics = AtomMetricsCE(dataset_infos=dataset_infos)
        # self.train_bond_metrics = BondMetricsCE()

    def forward(self, masked_pred, masked_true, log: bool):
        return None
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
        return
        for metric in [self.train_atom_metrics, self.train_bond_metrics]:
            metric.reset()

    def log_epoch_metrics(self, current_epoch, local_rank):
        return
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

    from midi.datasets import qm9_dataset, geom_dataset
    from midi.utils import setup_wandb
    from midi.analysis.rdkit_functions import Molecule

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
            # print("SMILES", data.smiles)

        #     bonds = torch_geometric.utils.to_dense_adj(edge_index=data.edge_index, edge_attr=data.edge_attr,
        #                                                max_num_nodes=len(data.x))[0]
        #     bonds[bonds == 4] = 1.5
        #     valencies = torch.sum(bonds, dim=-1)
        #
        #     for at_type, charge, val in zip(data.x, data.charges, valencies):
        #         c[dataset_infos.atom_decoder[at_type.item()], charge.item(), val.item()] += 1
        #
        # print("Atom-charge-valency counter", c)
        # #
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


