import torch
import torch.nn as nn
from torch.nn import init


class PositionsMLP(nn.Module):
    def __init__(self, hidden_dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.mlp = nn.Sequential(nn.Linear(1, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 1))

    def forward(self, pos, node_mask):
        norm = torch.norm(pos, dim=-1, keepdim=True)           # bs, n, 1
        new_norm = self.mlp(norm)                              # bs, n, 1
        new_pos = pos * new_norm / (norm + self.eps)
        new_pos = new_pos * node_mask.unsqueeze(-1)
        new_pos = new_pos - torch.mean(new_pos, dim=1, keepdim=True)
        return new_pos


class SE3Norm(nn.Module):
    def __init__(self, eps: float = 1e-5, device=None, dtype=None) -> None:
        """ Note: There is a relatively similar layer implemented by NVIDIA:
            https://catalog.ngc.nvidia.com/orgs/nvidia/resources/se3transformer_for_pytorch.
            It computes a ReLU on a mean-zero normalized norm, which I find surprising.
        """
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        self.normalized_shape = (1,)                   # type: ignore[arg-type]
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(self.normalized_shape, **factory_kwargs))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        init.ones_(self.weight)

    def forward(self, pos, node_mask):
        norm = torch.norm(pos, dim=-1, keepdim=True)           # bs, n, 1
        mean_norm = torch.sum(norm, dim=1, keepdim=True) / torch.sum(node_mask, dim=1, keepdim=True)      # bs, 1, 1
        new_pos = self.weight * pos / (mean_norm + self.eps)
        return new_pos

    def extra_repr(self) -> str:
        return '{normalized_shape}, eps={eps}'.format(**self.__dict__)


class Xtoy(nn.Module):
    def __init__(self, dx, dy):
        """ Map node features to global features """
        super().__init__()
        self.lin = nn.Linear(4 * dx, dy)

    def forward(self, X, x_mask):
        """ X: bs, n, dx. """
        x_mask = x_mask.expand(-1, -1, X.shape[-1])
        float_imask = 1 - x_mask.float()
        m = X.sum(dim=1) / torch.sum(x_mask, dim=1)
        mi = (X + 1e5 * float_imask).min(dim=1)[0]
        ma = (X - 1e5 * float_imask).max(dim=1)[0]
        std = torch.sum(((X - m[:, None, :]) ** 2) * x_mask, dim=1) / torch.sum(x_mask, dim=1)
        z = torch.hstack((m, mi, ma, std))
        out = self.lin(z)
        return out


class Etoy(nn.Module):
    def __init__(self, d, dy):
        """ Map edge features to global features. """
        super().__init__()
        self.lin = nn.Linear(4 * d, dy)

    def forward(self, E, e_mask1, e_mask2):
        """ E: bs, n, n, de
            Features relative to the diagonal of E could potentially be added.
        """
        mask = (e_mask1 * e_mask2).expand(-1, -1, -1, E.shape[-1])
        float_imask = 1 - mask.float()
        divide = torch.sum(mask, dim=(1, 2))
        m = E.sum(dim=(1, 2)) / divide
        mi = (E + 1e5 * float_imask).min(dim=2)[0].min(dim=1)[0]
        ma = (E - 1e5 * float_imask).max(dim=2)[0].max(dim=1)[0]
        std = torch.sum(((E - m[:, None, None, :]) ** 2) * mask, dim=(1, 2)) / divide
        z = torch.hstack((m, mi, ma, std))
        out = self.lin(z)
        return out


class EtoX(nn.Module):
    def __init__(self, de, dx):
        super().__init__()
        self.lin = nn.Linear(4 * de, dx)

    def forward(self, E, e_mask2):
        """ E: bs, n, n, de"""
        bs, n, _, de = E.shape
        e_mask2 = e_mask2.expand(-1, n, -1, de)
        float_imask = 1 - e_mask2.float()
        m = E.sum(dim=2) / torch.sum(e_mask2, dim=2)
        mi = (E + 1e5 * float_imask).min(dim=2)[0]
        ma = (E - 1e5 * float_imask).max(dim=2)[0]
        std = torch.sum(((E - m[:, :, None, :]) ** 2) * e_mask2, dim=2) / torch.sum(e_mask2, dim=2)
        z = torch.cat((m, mi, ma, std), dim=2)
        out = self.lin(z)
        return out


def masked_softmax(x, mask, **kwargs):
    if torch.sum(mask) == 0:
        return x
    x_masked = x.clone()
    x_masked[mask == 0] = -float("inf")
    return torch.softmax(x_masked, **kwargs)


class SetNorm(nn.LayerNorm):
    def __init__(self, feature_dim=None, **kwargs):
        super().__init__(normalized_shape=feature_dim, **kwargs)
        self.weights = nn.Parameter(torch.empty(1, 1, feature_dim))
        self.biases = nn.Parameter(torch.empty(1, 1, feature_dim))
        torch.nn.init.constant_(self.weights, 1.)
        torch.nn.init.constant_(self.biases, 0.)

    def forward(self, x, x_mask):
        bs, n, d = x.shape
        divide = torch.sum(x_mask, dim=1, keepdim=True) * d      # bs
        means = torch.sum(x * x_mask, dim=[1, 2], keepdim=True) / divide
        var = torch.sum((x - means) ** 2 * x_mask, dim=[1, 2], keepdim=True) / (divide + self.eps)
        out = (x - means) / (torch.sqrt(var) + self.eps)
        out = out * self.weights + self.biases
        out = out * x_mask
        return out


class GraphNorm(nn.LayerNorm):
    def __init__(self, feature_dim=None, **kwargs):
        super().__init__(normalized_shape=feature_dim, **kwargs)
        self.weights = nn.Parameter(torch.empty(1, 1, 1, feature_dim))
        self.biases = nn.Parameter(torch.empty(1, 1, 1, feature_dim))
        torch.nn.init.constant_(self.weights, 1.)
        torch.nn.init.constant_(self.biases, 0.)

    def forward(self, E, emask1, emask2):
        bs, n, _, d = E.shape
        divide = torch.sum(emask1 * emask2, dim=[1, 2], keepdim=True) * d      # bs
        means = torch.sum(E * emask1 * emask2, dim=[1, 2], keepdim=True) / divide
        var = torch.sum((E - means) ** 2 * emask1 * emask2, dim=[1, 2], keepdim=True) / (divide + self.eps)
        out = (E - means) / (torch.sqrt(var) + self.eps)
        out = out * self.weights + self.biases
        out = out * emask1 * emask2
        return out