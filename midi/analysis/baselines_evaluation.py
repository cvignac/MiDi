import torch
import numpy as np
import glob

from src.analysis.rdkit_functions import Molecule

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
from midi.metrics.molecular_metrics import SamplingMetrics


atom_encoder_dict = {'qm9_with_h': {'H': 0, 'C': 1, 'N': 2, 'O': 3, 'F': 4},
                     'qm9_no_h': {'C': 0, 'N': 1, 'O': 2, 'F': 3},
                     'geom_with_h': {'H': 0, 'B': 1, 'C': 2, 'N': 3, 'O': 4, 'F': 5, 'Al': 6, 'Si': 7,'P': 8,
                                     'S': 9, 'Cl': 10, 'As': 11, 'Br': 12, 'I': 13, 'Hg': 14, 'Bi': 15},
                     'geom_no_h': {'B': 0, 'C': 1, 'N': 2, 'O': 3, 'F': 4, 'Al': 5, 'Si': 6, 'P': 7, 'S': 8,
                                   'Cl': 9, 'As': 10,'Br': 11, 'I': 12, 'Hg': 13, 'Bi': 14}}

atom_decoder_dict = {'qm9_with_h': ['H', 'C', 'N', 'O', 'F'],
                     'qm9_no_h': ['C', 'N', 'O', 'F'],
                     'geom_with_h': ['H', 'B', 'C', 'N', 'O', 'F', 'Al', 'Si', 'P', 'S', 'Cl', 'As', 'Br',
                                     'I', 'Hg', 'Bi'],
                     'geom_no_h': ['B', 'C', 'N', 'O', 'F', 'Al', 'Si', 'P', 'S', 'Cl', 'As', 'Br', 'I',
                                   'Hg', 'Bi']}

bonds1 = {'H': {'H': 74, 'C': 109, 'N': 101, 'O': 96, 'F': 92,
                'B': 119, 'Si': 148, 'P': 144, 'As': 152, 'S': 134,
                'Cl': 127, 'Br': 141, 'I': 161},
          'C': {'H': 109, 'C': 154, 'N': 147, 'O': 143, 'F': 135,
                'Si': 185, 'P': 184, 'S': 182, 'Cl': 177, 'Br': 194,
                'I': 214},
          'N': {'H': 101, 'C': 147, 'N': 145, 'O': 140, 'F': 136,
                'Cl': 175, 'Br': 214, 'S': 168, 'I': 222, 'P': 177},
          'O': {'H': 96, 'C': 143, 'N': 140, 'O': 148, 'F': 142,
                'Br': 172, 'S': 151, 'P': 163, 'Si': 163, 'Cl': 164,
                'I': 194},
          'F': {'H': 92, 'C': 135, 'N': 136, 'O': 142, 'F': 142,
                'S': 158, 'Si': 160, 'Cl': 166, 'Br': 178, 'P': 156,
                'I': 187},
          'B': {'H':  119, 'Cl': 175},
          'Si': {'Si': 233, 'H': 148, 'C': 185, 'O': 163, 'S': 200,
                 'F': 160, 'Cl': 202, 'Br': 215, 'I': 243 },
          'Cl': {'Cl': 199, 'H': 127, 'C': 177, 'N': 175, 'O': 164,
                 'P': 203, 'S': 207, 'B': 175, 'Si': 202, 'F': 166,
                 'Br': 214},
          'S': {'H': 134, 'C': 182, 'N': 168, 'O': 151, 'S': 204,
                'F': 158, 'Cl': 207, 'Br': 225, 'Si': 200, 'P': 210,
                'I': 234},
          'Br': {'Br': 228, 'H': 141, 'C': 194, 'O': 172, 'N': 214,
                 'Si': 215, 'S': 225, 'F': 178, 'Cl': 214, 'P': 222},
          'P': {'P': 221, 'H': 144, 'C': 184, 'O': 163, 'Cl': 203,
                'S': 210, 'F': 156, 'N': 177, 'Br': 222},
          'I': {'H': 161, 'C': 214, 'Si': 243, 'N': 222, 'O': 194,
                'S': 234, 'F': 187, 'I': 266},
          'As': {'H': 152}
          }

bonds2 = {'C': {'C': 134, 'N': 129, 'O': 120, 'S': 160},
          'N': {'C': 129, 'N': 125, 'O': 121},
          'O': {'C': 120, 'N': 121, 'O': 121, 'P': 150},
          'P': {'O': 150, 'S': 186},
          'S': {'P': 186}}


bonds3 = {'C': {'C': 120, 'N': 116, 'O': 113},
          'N': {'C': 116, 'N': 110},
          'O': {'C': 113}}

margin1, margin2, margin3 = 10, 5, 3

def get_bond_order(atom1, atom2, distance, check_exists=False):
    distance = 100 * distance  # We change the metric

    # Check exists for large molecules where some atom pairs do not have a
    # typical bond length.
    if check_exists:
        if atom1 not in bonds1:
            return 0
        if atom2 not in bonds1[atom1]:
            return 0

    # margin1, margin2 and margin3 have been tuned to maximize the stability of
    # the QM9 true samples.
    if distance < bonds1[atom1][atom2] + margin1:

        # Check if atoms in bonds2 dictionary.
        if atom1 in bonds2 and atom2 in bonds2[atom1]:
            thr_bond2 = bonds2[atom1][atom2] + margin2
            if distance < thr_bond2:
                if atom1 in bonds3 and atom2 in bonds3[atom1]:
                    thr_bond3 = bonds3[atom1][atom2] + margin3
                    if distance < thr_bond3:
                        return 3        # Triple
                return 2            # Double
        return 1                # Single
    return 0                    # No bond


def geom_predictor(p, l, margin1=5, limit_bonds_to_one=False):
    """ p: atom pair (couple of str)
        l: bond length (float)"""
    bond_order = get_bond_order(p[0], p[1], l, check_exists=True)

    # If limit_bonds_to_one is enabled, every bond type will return 1.
    if limit_bonds_to_one:
        return 1 if bond_order > 0 else 0
    else:
        return bond_order


def build_xae_molecule(positions, atom_types, dataset_info, atom_decoder):
    """ Returns a triplet (X, A, E): atom_types, adjacency matrix, edge_types
        args:
        positions: N x 3  (already masked to keep final number nodes)
        atom_types: N
        returns:
        X: N         (int)
        A: N x N     (bool)                  (binary adjacency matrix)
        E: N x N     (int)  (bond type, 0 if no bond) such that A = E.bool()
    """
    n = positions.shape[0]
    X = atom_types
    A = torch.zeros((n, n), dtype=torch.bool)
    E = torch.zeros((n, n), dtype=torch.int)

    pos = positions.unsqueeze(0)
    dists = torch.cdist(pos, pos, p=2).squeeze(0)
    for i in range(n):
        for j in range(i):
            pair = sorted([atom_types[i], atom_types[j]])
            if dataset_info['name'] == 'qm9':
                order = get_bond_order(atom_decoder[pair[0]], atom_decoder[pair[1]], dists[i, j])
            elif dataset_info['name'] == 'geom':
                order = geom_predictor((atom_decoder[pair[0]], atom_decoder[pair[1]]), dists[i, j], limit_bonds_to_one=True)
            # TODO: a batched version of get_bond_order to avoid the for loop
            if order > 0:
                # Warning: the graph should be DIRECTED
                A[i, j] = 1
                E[i, j] = order
                E[j, i] = order
    return X, A, E

def data_atom_xyz(path, name, dataset_info):
    atom_encoder = atom_encoder_dict[name]
    atom_decoder = atom_decoder_dict[name]
    file_list = glob.glob(path + "*.xyz")

    all_data = []
    for file_path in file_list:
        with open(file_path, "r") as f:
            f.readline()
            data = np.loadtxt(f, dtype=[("symbol", "U10"), ("x", float), ("y", float), ("z", float)])
        all_data.append(data)

    all_mols = []
    for data in all_data:
        atom_types_str = data['symbol']
        atom_types = []
        for type in atom_types_str:
            atom_type = atom_encoder[type]
            atom_types.append(atom_type)
        atom_types = torch.tensor(atom_types)
        pos = data[['x', 'y', 'z']]
        positions = pos.tolist()
        positions = torch.tensor(positions)

        X, A, E = build_xae_molecule(positions=positions, atom_types=atom_types, dataset_info=dataset_info, atom_decoder=atom_decoder)
        charges = torch.zeros(X.shape)
        E = E.to(dtype=torch.int64)
        molecule = Molecule(atom_types=X, bond_types=E, positions=positions, charges=charges, atom_decoder=atom_decoder)
        molecule.build_molecule(atom_decoder=atom_decoder)
        all_mols.append(molecule)

    return all_mols


def open_babel_preprocess(file, name):
    """
    :param file: str
    :param name: 'qm9_with_h', 'qm9_no_h, 'geom_with_h', 'geom_no_h'
    :return:
    """
    atom_encoder = atom_encoder_dict[name]
    atom_decoder = atom_decoder_dict[name]

    with open(file, "r") as f:
        lines = f.readlines()[3:]

    result = []
    temp = []

    for line in lines:
        line = line.strip()

        if not line or "M" in line or "$" in line or "OpenBabel" in line:
            continue

        vec = line.split()
        if vec != ['end']:
            temp.append(vec)
        else:
            result.append(temp)
            temp = []

    all_mols = []

    for array in result:
        atom_temp = []
        pos_temp = []
        new_pos = []
        col = row = array[0][0]
        for i in range(int(col)):
            element = array[i + 1][3]
            x = atom_encoder.get(element, None)
            if x is None:
                # Handle elements not in the map
                print('Element ' + element + ' is not handled in the current mapping')
            atom_temp.append(x)
            x_pos = array[i + 1][0]
            x_pos = float(x_pos)
            y_pos = array[i + 1][1]
            y_pos = float(y_pos)
            z_pos = array[i + 1][2]
            z_pos = float(z_pos)
            pos_temp.append([x_pos, y_pos, z_pos])
        new_pos.append(pos_temp)

        iteration = array[0][1]
        cols, rows = int(col), int(row)
        matrix = [[0 for x in range(cols)] for y in range(rows)]
        for j in range(int(iteration)):
            d = j + int(col) + 1
            a = int(array[d][0]) - 1
            b = int(array[d][1]) - 1
            c = int(array[d][2])
            matrix[a][b] = c
            matrix[b][a] = c

        X = torch.tensor(atom_temp)
        charges = torch.zeros(X.shape)
        E = torch.tensor(matrix)
        posis = torch.tensor(new_pos[0])
        molecule = Molecule(atom_types=X, bond_types=E, positions=posis, charges=charges, atom_decoder=atom_decoder)
        molecule.build_molecule(atom_decoder=atom_decoder)
        all_mols.append(molecule)

    return all_mols


@hydra.main(version_base='1.3', config_path='../../configs', config_name='config')
def open_babel_eval(cfg: omegaconf.DictConfig, file: str = None):
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

    name = dataset_config.name + ("_no_h" if cfg.dataset.remove_h else "_with_h")

    open_babel_mols = open_babel_preprocess(file, name)

    sampling_metrics(molecules=open_babel_mols, name='openbabel', current_epoch=-1, local_rank=0)


open_babel_eval(file=None)
