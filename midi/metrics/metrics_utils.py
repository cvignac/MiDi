import math
from collections import Counter

import numpy as np
import torch
from torch_geometric.data import Data
import torch.nn.functional as F
from midi.datasets.dataset_utils import Statistics
from torchmetrics import MeanAbsoluteError


def molecules_to_datalist(molecules):
    data_list = []
    for molecule in molecules:
        x = molecule.atom_types.long()
        bonds = molecule.bond_types.long()
        positions = molecule.positions
        charges = molecule.charges
        edge_index = bonds.nonzero().contiguous().T
        bond_types = bonds[edge_index[0], edge_index[1]]
        edge_attr = bond_types.long()
        data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, pos=positions, charges=charges)
        data_list.append(data)

    return data_list


def compute_all_statistics(data_list, atom_encoder, charges_dic):
    num_nodes = node_counts(data_list)
    atom_types = atom_type_counts(data_list, num_classes=len(atom_encoder))
    print(f"Atom types: {atom_types}")
    bond_types = edge_counts(data_list)
    print(f"Bond types: {bond_types}")
    charge_types = charge_counts(data_list, num_classes=len(atom_encoder), charges_dic=charges_dic)
    print(f"Charge types: {charge_types}")
    valency = valency_count(data_list, atom_encoder)
    print("Valency: ", valency)
    bond_lengths = bond_lengths_counts(data_list)
    print("Bond lengths: ", bond_lengths)
    angles = bond_angles(data_list, atom_encoder)
    return Statistics(num_nodes=num_nodes, atom_types=atom_types, bond_types=bond_types, charge_types=charge_types,
                      valencies=valency, bond_lengths=bond_lengths, bond_angles=angles)


def node_counts(data_list):
    print("Computing node counts...")
    all_node_counts = Counter()
    for i, data in enumerate(data_list):
        num_nodes = data.num_nodes
        all_node_counts[num_nodes] += 1
    print("Done.")
    return all_node_counts


def atom_type_counts(data_list, num_classes):
    print("Computing node types distribution...")
    counts = np.zeros(num_classes)
    for data in data_list:
        x = torch.nn.functional.one_hot(data.x, num_classes=num_classes)
        counts += x.sum(dim=0).numpy()

    counts = counts / counts.sum()
    print("Done.")
    return counts


def edge_counts(data_list, num_bond_types=5):
    print("Computing edge counts...")
    d = np.zeros(num_bond_types)

    for data in data_list:
        total_pairs = data.num_nodes * (data.num_nodes - 1)

        num_edges = data.edge_attr.shape[0]
        num_non_edges = total_pairs - num_edges
        assert num_non_edges >= 0

        edge_types = torch.nn.functional.one_hot(data.edge_attr - 1, num_classes=num_bond_types - 1).sum(dim=0).numpy()
        d[0] += num_non_edges
        d[1:] += edge_types

    d = d / d.sum()
    return d


def charge_counts(data_list, num_classes, charges_dic):
    print("Computing charge counts...")
    d = np.zeros((num_classes, len(charges_dic)))


    for data in data_list:
        for atom, charge in zip(data.x, data.charges):
            assert charge in [-2, -1, 0, 1, 2, 3]
            d[atom.item(), charges_dic[charge.item()]] += 1

    s = np.sum(d, axis=1, keepdims=True)
    s[s == 0] = 1
    d = d / s
    print("Done.")
    return d


def valency_count(data_list, atom_encoder):
    atom_decoder = {v: k for k, v in atom_encoder.items()}
    print("Computing valency counts...")
    valencies = {atom_type: Counter() for atom_type in atom_encoder.keys()}

    for data in data_list:
        edge_attr = data.edge_attr
        edge_attr[edge_attr == 4] = 1.5
        bond_orders = edge_attr

        for atom in range(data.num_nodes):
            edges = bond_orders[data.edge_index[0] == atom]
            valency = edges.sum(dim=0)
            valencies[atom_decoder[data.x[atom].item()]][valency.item()] += 1

    # Normalizing the valency counts
    for atom_type in valencies.keys():
        s = sum(valencies[atom_type].values())
        for valency, count in valencies[atom_type].items():
            valencies[atom_type][valency] = count / s
    print("Done.")
    return valencies



def bond_lengths_counts(data_list, num_bond_types=5):
    """ Compute the bond lenghts separetely for each bond type. """
    print("Computing bond lengths...")
    all_bond_lenghts = {1: Counter(), 2: Counter(), 3: Counter(), 4: Counter()}
    for data in data_list:
        cdists = torch.cdist(data.pos.unsqueeze(0), data.pos.unsqueeze(0)).squeeze(0)
        bond_distances = cdists[data.edge_index[0], data.edge_index[1]]
        for bond_type in range(1, num_bond_types):
            bond_type_mask = data.edge_attr == bond_type
            distances_to_consider = bond_distances[bond_type_mask]
            distances_to_consider = torch.round(distances_to_consider, decimals=2)
            for d in distances_to_consider:
                all_bond_lenghts[bond_type][d.item()] += 1

    # Normalizing the bond lenghts
    for bond_type in range(1, num_bond_types):
        s = sum(all_bond_lenghts[bond_type].values())
        for d, count in all_bond_lenghts[bond_type].items():
            all_bond_lenghts[bond_type][d] = count / s
    print("Done.")
    return all_bond_lenghts


def bond_angles(data_list, atom_encoder):
    atom_decoder = {v: k for k, v in atom_encoder.items()}
    print("Computing bond angles...")
    all_bond_angles = np.zeros((len(atom_encoder.keys()), 180 * 10 + 1))
    for data in data_list:
        assert not torch.isnan(data.pos).any()
        for i in range(data.num_nodes):
            neighbors = data.edge_index[1][data.edge_index[0] == i]
            for j in neighbors:
                for k in neighbors:
                    if j == k:
                        continue
                    assert i != j and i != k and j != k, "i, j, k: {}, {}, {}".format(i, j, k)
                    a = data.pos[j] - data.pos[i]
                    b = data.pos[k] - data.pos[i]

                    # print(a, b, torch.norm(a) * torch.norm(b))
                    angle = torch.acos(torch.dot(a, b) / (torch.norm(a) * torch.norm(b) + 1e-6))
                    angle = angle * 180 / math.pi

                    bin = int(torch.round(angle, decimals=1) * 10)
                    all_bond_angles[data.x[i].item(), bin] += 1

    # Normalizing the angles
    s = all_bond_angles.sum(axis=1, keepdims=True)
    s[s == 0] = 1
    all_bond_angles = all_bond_angles / s
    print("Done.")
    return all_bond_angles


def counter_to_tensor(c: Counter):
    max_key = max(c.keys())
    assert type(max_key) == int
    arr = torch.zeros(max_key + 1, dtype=torch.float)
    for k, v in c.items():
        arr[k] = v
    arr / torch.sum(arr)
    return arr


def wasserstein1d(preds, target, step_size=1):
        """ preds and target are 1d tensors. They contain histograms for bins that are regularly spaced """
        target = normalize(target) / step_size
        preds = normalize(preds) / step_size
        max_len = max(len(preds), len(target))
        preds = F.pad(preds, (0, max_len - len(preds)))
        target = F.pad(target, (0, max_len - len(target)))

        cs_target = torch.cumsum(target, dim=0)
        cs_preds = torch.cumsum(preds, dim=0)
        return torch.sum(torch.abs(cs_preds - cs_target)).item()


def total_variation1d(preds, target):
    assert target.dim() == 1 and preds.shape == target.shape, f"preds: {preds.shape}, target: {target.shape}"
    target = normalize(target)
    preds = normalize(preds)
    return torch.sum(torch.abs(preds - target)).item(), torch.abs(preds - target)


def normalize(tensor):
    s = tensor.sum()
    assert s > 0
    return tensor / s
