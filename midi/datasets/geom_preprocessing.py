from rdkit import Chem
import numpy as np
import torch
from torch_geometric import data
import numpy.random as npr
import pickle
import pathlib
import os
import os.path as osp
# import msgpack


data_path = '/home/vignac/MoleculeDiffusion/data/geom/rdkit_folder/rdkit_mols.pickle'


def save_pickle(array, path):
    with open(path, 'wb') as f:
        pickle.dump(array, f)

def load_pickle(path):
    with open(path, 'rb') as f:
        return pickle.load(f)


def random_split(data_list, val_proportion=0.1, test_proportion=0.1, seed=0):
    num_mol = len(data_list)
    num_test_mols = int(num_mol * test_proportion)
    num_val_mols = int(num_mol * val_proportion)
    num_train_mols = int(num_mol - num_test_mols - num_val_mols)

    split = ['test'] * num_test_mols + ['val'] * num_val_mols + ['train'] * num_train_mols
    shuffle = np.random.RandomState(seed).permutation(num_mol)
    split = np.array(split)[shuffle]

    train_data = []
    val_data = []
    test_data = []
    for data, s in zip(data_list, split):
        if s == 'train':
            train_data.append(data)
        elif s == 'val':
            val_data.append(data)
        else:
            test_data.append(data)

    save_pickle(train_data, '/home/vignac/MoleculeDiffusion/data/geom/rdkit_folder/train_data.pickle')
    save_pickle(val_data, '/home/vignac/MoleculeDiffusion/data/geom/rdkit_folder/val_data.pickle')
    save_pickle(test_data, '/home/vignac/MoleculeDiffusion/data/geom/rdkit_folder/test_data.pickle')
    return


def save_train_smiles(train_data_path, output_file='train_smiles_set.pickle'):
    train_data = load_pickle(train_data_path)
    train_smiles = []
    for smiles, _ in train_data:
        train_smiles.append(smiles)
    train_smiles = set(train_smiles)
    output_path = osp.join(pathlib.Path(train_data_path).parent, output_file)
    save_pickle(train_smiles, output_path)


def extract_conformers(args):
    drugs_file = os.path.join(args.data_dir, args.data_file)
    save_file = f"geom_drugs_{'no_h_' if args.remove_h else ''}{args.conformations}"
    smiles_list_file = 'geom_drugs_smiles.txt'
    number_atoms_file = f"geom_drugs_n_{'no_h_' if args.remove_h else ''}{args.conformations}"

    unpacker = msgpack.Unpacker(open(drugs_file, "rb"))

    all_smiles = []
    all_number_atoms = []
    dataset_conformers = []
    mol_id = 0
    for i, drugs_1k in enumerate(unpacker):
        print(f"Unpacking file {i}...")
        for smiles, all_info in drugs_1k.items():
            all_smiles.append(smiles)
            conformers = all_info['conformers']
            # Get the energy of each conformer. Keep only the lowest values
            all_energies = []
            for conformer in conformers:
                all_energies.append(conformer['totalenergy'])
            all_energies = np.array(all_energies)
            argsort = np.argsort(all_energies)
            lowest_energies = argsort[:args.conformations]
            for id in lowest_energies:
                conformer = conformers[id]
                coords = np.array(conformer['xyz']).astype(float)        # n x 4
                if args.remove_h:
                    mask = coords[:, 0] != 1.0
                    coords = coords[mask]
                n = coords.shape[0]
                all_number_atoms.append(n)
                mol_id_arr = mol_id * np.ones((n, 1), dtype=float)
                id_coords = np.hstack((mol_id_arr, coords))

                dataset_conformers.append(id_coords)
                mol_id += 1

    print("Total number of conformers saved", mol_id)
    all_number_atoms = np.array(all_number_atoms)
    dataset = np.vstack(dataset_conformers)

    print("Total number of atoms in the dataset", dataset.shape[0])
    print("Average number of atoms per molecule", dataset.shape[0] / mol_id)

    # Save conformations
    np.save(os.path.join(args.data_dir, save_file), dataset)
    # Save SMILES
    with open(os.path.join(args.data_dir, smiles_list_file), 'w') as f:
        for s in all_smiles:
            f.write(s)
            f.write('\n')

    # Save number of atoms per conformation
    np.save(os.path.join(args.data_dir, number_atoms_file), all_number_atoms)
    print("Dataset processed.")



# data = load_pickle(data_path)
# random_split(data, 0.1, 0.1, seed=0)

# save_train_smiles('/home/vignac/MoleculeDiffusion/data/geom/rdkit_folder/train_data.pickle')
