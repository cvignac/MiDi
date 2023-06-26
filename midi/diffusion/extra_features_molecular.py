import torch

import midi.utils as utils


class ChargeFeature:
    def __init__(self, remove_h, valencies):
        self.remove_h = remove_h
        self.valencies = valencies

    def __call__(self, z_t):
        bond_orders = torch.tensor([0, 1, 2, 3, 1.5], device=z_t.E.device).reshape(1, 1, 1, -1)
        weighted_E = z_t.E * bond_orders      # (bs, n, n, de)
        current_valencies = weighted_E.argmax(dim=-1).sum(dim=-1)   # (bs, n)

        valencies = torch.tensor(self.valencies, device=z_t.X.device).reshape(1, 1, -1)
        X = z_t.X * valencies  # (bs, n, dx)
        normal_valencies = torch.argmax(X, dim=-1)               # (bs, n)

        return (normal_valencies - current_valencies).to(z_t.X.device)


class ValencyFeature:
    def __init__(self):
        pass

    def __call__(self, z_t):
        orders = torch.tensor([0, 1, 2, 3, 1.5], device=z_t.E.device).reshape(1, 1, 1, -1)
        E = z_t.E * orders      # (bs, n, n, de)
        valencies = E.argmax(dim=-1).sum(dim=-1)    # (bs, n)
        return valencies.to(z_t.X.device)


class WeightFeature:
    def __init__(self, max_weight, atom_weights):
        self.max_weight = max_weight
        self.atom_weight_list = torch.Tensor(list(atom_weights.values()))

    def __call__(self, z_t):
        X = torch.argmax(z_t.X, dim=-1)     # (bs, n)
        X_weights = self.atom_weight_list.to(X.device)[X]            # (bs, n)
        return X_weights.sum(dim=-1).unsqueeze(-1) / self.max_weight     # (bs, 1)
