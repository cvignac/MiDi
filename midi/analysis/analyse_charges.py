import pickle
from tqdm import tqdm
from rdkit import Chem


file = '/home/vignac/MoleculeDiffusion/data/geom/rdkit_folder/rdkit_mols.pickle'

def load_pickle(path):
    with open(path, 'rb') as f:
        return pickle.load(f)


def extract_smiles_and_save(path):
    all_smiles = [smiles for (smiles, _) in tqdm(load_pickle(path))]
    with open('all_smiles.txt', 'w') as f:
        for smiles in all_smiles:
            f.write(smiles + '\n')
    return


# extract_smiles_and_save(file)


def extract_charges(smiles_file, cheap=True):
    all_charged_atoms = []
    if cheap:
        for i, smiles in tqdm(enumerate(open(smiles_file, 'r'))):
            if '+' in smiles:
                substring = smiles.split('+')[0][-3:] + '+'
                if substring not in all_charged_atoms:
                    all_charged_atoms.append(substring)

        print("all_charged_atoms", all_charged_atoms)
    else:
        for smiles in tqdm(open(smiles_file, 'r')):
            mol = Chem.MolFromSmiles(smiles)
            charged_atoms = [atom.GetFormalCharge() for atom in mol.GetAtoms()]

smiles_file = '/Users/clementvignac/src/github_cvignac/MoleculeDiffusion/data/geom/all_smiles.txt'


extract_charges(smiles_file, cheap=True)

