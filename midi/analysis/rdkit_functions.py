import torch
from rdkit import Chem, RDLogger
from rdkit.Geometry import Point3D

lg = RDLogger.logger()
lg.setLevel(RDLogger.CRITICAL)

allowed_bonds = {'H': {0: 1, 1: 0, -1: 0},
                 'C': {0: [3, 4], 1: 3, -1: 3},
                 'N': {0: [2, 3], 1: [2, 3, 4], -1: 2},    # In QM9, N+ seems to be present in the form NH+ and NH2+
                 'O': {0: 2, 1: 3, -1: 1},
                 'F': {0: 1, -1: 0},
                 'B': 3, 'Al': 3, 'Si': 4,
                 'P': {0: [3, 5], 1: 4},
                 'S': {0: [2, 6], 1: [2, 3], 2: 4, 3: 5, -1: 3},
                 'Cl': 1, 'As': 3,
                 'Br': {0: 1, 1: 2}, 'I': 1, 'Hg': [1, 2], 'Bi': [3, 5], 'Se': [2, 4, 6]}
bond_dict = [None, Chem.rdchem.BondType.SINGLE, Chem.rdchem.BondType.DOUBLE, Chem.rdchem.BondType.TRIPLE,
             Chem.rdchem.BondType.AROMATIC]
ATOM_VALENCY = {6: 4, 7: 3, 8: 2, 9: 1, 15: 3, 16: 2, 17: 1, 35: 1, 53: 1}


class Molecule:
    def __init__(self, atom_types, bond_types, positions, charges, atom_decoder):
        """ atom_types: n      LongTensor
            charges: n         LongTensor
            bond_types: n x n  LongTensor
            positions: n x 3   FloatTensor
            atom_decoder: extracted from dataset_infos. """
        assert atom_types.dim() == 1 and atom_types.dtype == torch.long, f"shape of atoms {atom_types.shape} " \
                                                                         f"and dtype {atom_types.dtype}"
        assert bond_types.dim() == 2 and bond_types.dtype == torch.long, f"shape of bonds {bond_types.shape} --" \
                                                                         f" {bond_types.dtype}"
        assert len(atom_types.shape) == 1
        assert len(bond_types.shape) == 2
        assert len(positions.shape) == 2

        self.atom_types = atom_types.long()
        self.bond_types = bond_types.long()
        self.positions = positions
        self.charges = charges
        self.rdkit_mol = self.build_molecule(atom_decoder)
        self.num_nodes = len(atom_types)
        self.num_atom_types = len(atom_decoder)

    def build_molecule(self, atom_decoder, verbose=False):
        """ If positions is None,
        """
        if verbose:
            print("building new molecule")

        mol = Chem.RWMol()
        for atom, charge in zip(self.atom_types, self.charges):
            if atom == -1:
                continue
            a = Chem.Atom(atom_decoder[int(atom.item())])
            if charge.item() != 0:
                a.SetFormalCharge(charge.item())
            mol.AddAtom(a)
            if verbose:
                print("Atom added: ", atom.item(), atom_decoder[atom.item()])

        edge_types = torch.triu(self.bond_types, diagonal=1)
        edge_types[edge_types == -1] = 0
        all_bonds = torch.nonzero(edge_types)
        for i, bond in enumerate(all_bonds):
            if bond[0].item() != bond[1].item():
                mol.AddBond(bond[0].item(), bond[1].item(), bond_dict[edge_types[bond[0], bond[1]].item()])
                if verbose:
                    print("bond added:", bond[0].item(), bond[1].item(), edge_types[bond[0], bond[1]].item(),
                          bond_dict[edge_types[bond[0], bond[1]].item()])

        try:
            mol = mol.GetMol()
        except Chem.KekulizeException:
            print("Can't kekulize molecule")
            return None

        # Set coordinates
        positions = self.positions.double()
        conf = Chem.Conformer(mol.GetNumAtoms())
        for i in range(mol.GetNumAtoms()):
            conf.SetAtomPosition(i, Point3D(positions[i][0].item(), positions[i][1].item(), positions[i][2].item()))
        mol.AddConformer(conf)

        return mol


def check_stability(molecule, dataset_info, debug=False, atom_decoder=None, smiles=None):
    """ molecule: Molecule object. """
    device = molecule.atom_types.device
    if atom_decoder is None:
        atom_decoder = dataset_info.atom_decoder

    atom_types = molecule.atom_types
    edge_types = molecule.bond_types

    edge_types[edge_types == 4] = 1.5
    edge_types[edge_types < 0] = 0

    valencies = torch.sum(edge_types, dim=-1).long()

    n_stable_bonds = 0
    mol_stable = True
    for i, (atom_type, valency, charge) in enumerate(zip(atom_types, valencies, molecule.charges)):
        atom_type = atom_type.item()
        valency = valency.item()
        charge = charge.item()
        possible_bonds = allowed_bonds[atom_decoder[atom_type]]
        if type(possible_bonds) == int:
            is_stable = possible_bonds == valency
        elif type(possible_bonds) == dict:
            expected_bonds = possible_bonds[charge] if charge in possible_bonds.keys() else possible_bonds[0]
            is_stable = expected_bonds == valency if type(expected_bonds) == int else valency in expected_bonds
        else:
            is_stable = valency in possible_bonds
        if not is_stable:
            mol_stable = False
        if not is_stable and debug:
            if smiles is not None:
                print(smiles)
            print(f"Invalid atom {atom_decoder[atom_type]}: valency={valency}, charge={charge}")
            print()
        n_stable_bonds += int(is_stable)

    return torch.tensor([mol_stable], dtype=torch.float, device=device),\
           torch.tensor([n_stable_bonds], dtype=torch.float, device=device),\
           len(atom_types)

