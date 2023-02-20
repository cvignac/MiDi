import json
import os
import os.path as osp
import pathlib
import pickle
from tqdm import tqdm
import numpy as np

conformations = 30
data_dir = '/home/vignac/MoleculeDiffusion/data/geom/rdkit_folder'
save_to = 'rdkit_mols.pickle'
summary = osp.join(data_dir, 'summary_drugs.json')

with open(summary, "r") as f:
    drugs_summ = json.load(f)

    all_data = []
    num_charged_molecules = 0
    for i, (smiles, d) in tqdm(enumerate(drugs_summ.items())):
        if d['charge'] != 0:
            # print(f"Warning: charge={d['charge']}: {smiles}")
            num_charged_molecules += 1
        if 'pickle_path' not in d.keys():
            continue
        pickle_path = d['pickle_path']
        with open(osp.join(data_dir, pickle_path), "rb") as f:
            dic = pickle.load(f)
            conformers = dic['conformers']
            # Get the energy of each conformer. Keep only the lowest values
            all_energies = []
            for conformer in conformers:
                all_energies.append(conformer['totalenergy'])
            all_energies = np.array(all_energies)
            argsort = np.argsort(all_energies)
            lowest_energies = argsort[:conformations]

            saved_conformers = []
            for id in lowest_energies:
                conformer = conformers[id]
                rdkit_mol = conformer['rd_mol']
                saved_conformers.append(rdkit_mol)
            all_data.append((smiles, saved_conformers))

        # if i > 1000:
        #     break
with open(osp.join(data_dir, save_to), 'wb') as f:
    pickle.dump(all_data, f)


with open(osp.join(data_dir, save_to), 'rb') as f:
    test = pickle.load(f)

print(test)
print("Number of charged molecules", num_charged_molecules)

def charges_analysis(path):
    with open(path, 'rb') as f:
        l =  pickle.load(f)
    for (smiles, conformers) in l:
        for mol in conformers:
            if mol.GetFormalCharge() != 0:
                print(smiles, mol.GetFormalCharge())