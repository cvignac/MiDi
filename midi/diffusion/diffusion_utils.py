import torch
from torch.nn import functional as F
import numpy as np
import math
import matplotlib.pyplot as plt
from midi.utils import PlaceHolder, remove_mean_with_mask


def sum_except_batch(x):
    return x.reshape(x.size(0), -1).sum(dim=-1)


def assert_correctly_masked(variable, node_mask):
    assert not torch.isnan(variable).any(), f"Shape:{variable.shape}"
    assert (variable * (1 - node_mask.long())).abs().max().item() < 1e-4, \
        f'Variables not masked properly. {variable * (1 - node_mask.long())}'


def sample_gaussian_with_mask(size, node_mask):
    x = torch.randn(size).to(node_mask.device)
    x_masked = x * node_mask
    return x_masked


def remove_mean_with_mask(pos, node_mask):
    """ pos: bs x n x 3 (float32)
        node_mask: bs x n (bool)"""
    assert node_mask.dtype == torch.bool, f"Wrong dtype for the mask: {node_mask.dtype}"
    node_mask = node_mask.unsqueeze(-1)
    masked_max_abs_value = (pos * (~node_mask)).abs().sum().item()
    assert masked_max_abs_value < 1e-5, f'Error {masked_max_abs_value} too high'
    N = node_mask.sum(1, keepdims=True)

    mean = torch.sum(pos, dim=1, keepdim=True) / N
    pos = pos - mean * node_mask
    return pos


def clip_noise_schedule(alphas2, clip_value=0.001):
    """
    For a noise schedule given by alpha^2, this clips alpha_t / alpha_t-1. This may help improve stability during
    sampling.
    """
    alphas2 = np.concatenate([np.ones(1), alphas2], axis=0)
    alphas_step = (alphas2[1:] / alphas2[:-1])
    alphas_step = np.clip(alphas_step, a_min=clip_value, a_max=1.)
    alphas2 = np.cumprod(alphas_step, axis=0)
    return alphas2


def cosine_beta_schedule(timesteps, s=0.008, raise_to_power: float = 1):
    """ Cosine schedule as proposed in https://openreview.net/forum?id=-NEXDKk8gZ. """
    steps = timesteps + 2
    x = np.linspace(0, steps, steps)
    alphas_cumprod = np.cos(((x / steps) + s) / (1 + s) * np.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    betas = np.clip(betas, a_min=0, a_max=0.999)
    alphas = 1. - betas
    alphas_cumprod = np.cumprod(alphas, axis=0)

    if raise_to_power != 1:
        alphas_cumprod = np.power(alphas_cumprod, raise_to_power)

    return alphas_cumprod


def cosine_beta_schedule_discrete(timesteps, nu_arr, s=0.008):
    """ Cosine schedule as proposed in https://openreview.net/forum?id=-NEXDKk8gZ. """
    steps = timesteps + 2
    x = np.linspace(0, steps, steps)
    x = np.expand_dims(x, 0)  # ((1, steps))

    nu_arr = np.array(nu_arr)  # (components, )  # X, charges, E, y, pos
    nu_arr = np.expand_dims(nu_arr, 1)  # ((components, 1))

    alphas_cumprod = np.cos(0.5 * np.pi * (((x / steps) ** nu_arr) + s) / (1 + s)) ** 2  # ((components, steps))
    # divide every element of alphas_cumprod by the first element of alphas_cumprod
    alphas_cumprod_new = alphas_cumprod / np.expand_dims(alphas_cumprod[:, 0], 1)
    # remove the first element of alphas_cumprod and then multiply every element by the one before it
    alphas = (alphas_cumprod_new[:, 1:] / alphas_cumprod_new[:, :-1])

    betas = 1 - alphas  # ((components, steps)) # X, charges, E, y, pos
    betas = np.swapaxes(betas, 0, 1)
    # plt.figure()
    # plt.plot(x[0, 1:], alphas[-1, ...], label='alpha')
    # plt.plot(x[0, 1:], betas[..., -1], label='betas')
    # plt.plot(x[0, ], alphas_cumprod[-1, ...], label='alpha_bar')
    # plt.show()
    # assert False
    return betas


def gaussian_KL(q_mu, q_sigma):
    """Computes the KL distance between a normal distribution and the standard normal.
        Args:
            q_mu: Mean of distribution q.
            q_sigma: Standard deviation of distribution q.
            p_mu: Mean of distribution p.
            p_sigma: Standard deviation of distribution p.
        Returns:
            The KL distance, summed over all dimensions except the batch dim.
        """
    return torch.log(1 / q_sigma) + 0.5 * (q_sigma ** 2 + q_mu ** 2) - 0.5


def cdf_std_gaussian(x):
    return 0.5 * (1. + torch.erf(x / math.sqrt(2)))


def SNR(gamma):
    """Computes signal to noise ratio (alpha^2/sigma^2) given gamma."""
    return torch.exp(-gamma)


def inflate_batch_array(array, target_shape):
    """
    Inflates the batch array (array) with only a single axis (i.e. shape = (batch_size,), or possibly more empty
    axes (i.e. shape (batch_size, 1, ..., 1)) to match the target shape.
    """
    target_shape = (array.size(0),) + (1,) * (len(target_shape) - 1)
    return array.view(target_shape)


def sigma(gamma, target_shape):
    """Computes sigma given gamma."""
    return inflate_batch_array(torch.sqrt(torch.sigmoid(gamma)), target_shape)


def alpha(gamma, target_shape):
    """Computes alpha given gamma."""
    return inflate_batch_array(torch.sqrt(torch.sigmoid(-gamma)), target_shape)


def check_mask_correct(variables, node_mask):
    for i, variable in enumerate(variables):
        if len(variable) > 0:
            assert_correctly_masked(variable, node_mask)


def check_tensor_same_size(*args):
    for i, arg in enumerate(args):
        if i == 0:
            continue
        assert args[0].size() == arg.size()


def sigma_and_alpha_t_given_s(gamma_t: torch.Tensor, gamma_s: torch.Tensor, target_size: torch.Size):
    """
    Computes sigma t given s, using gamma_t and gamma_s. Used during sampling.

    These are defined as:
        alpha t given s = alpha t / alpha s,
        sigma t given s = sqrt(1 - (alpha t given s) ^2 ).
    """
    sigma2_t_given_s = inflate_batch_array(
        -torch.expm1(F.softplus(gamma_s) - F.softplus(gamma_t)), target_size
    )

    # alpha_t_given_s = alpha_t / alpha_s
    log_alpha2_t = F.logsigmoid(-gamma_t)
    log_alpha2_s = F.logsigmoid(-gamma_s)
    log_alpha2_t_given_s = log_alpha2_t - log_alpha2_s

    alpha_t_given_s = torch.exp(0.5 * log_alpha2_t_given_s)
    alpha_t_given_s = inflate_batch_array(alpha_t_given_s, target_size)

    sigma_t_given_s = torch.sqrt(sigma2_t_given_s)

    return sigma2_t_given_s, sigma_t_given_s, alpha_t_given_s


def reverse_tensor(x):
    return x[torch.arange(x.size(0) - 1, -1, -1)]


def check_issues_norm_values(gamma, norm_val1, norm_val2, num_stdevs=8):
    """ Check if 1 / norm_value is still larger than 10 * standard deviation. """
    zeros = torch.zeros((1, 1))
    gamma_0 = gamma(zeros)
    sigma_0 = sigma(gamma_0, target_shape=zeros.size()).item()
    max_norm_value = max(norm_val1, norm_val2)
    if sigma_0 * num_stdevs > 1. / max_norm_value:
        raise ValueError(
            f'Value for normalization value {max_norm_value} probably too '
            f'large with sigma_0 {sigma_0:.5f}*{num_stdevs} and '
            f'1 / norm_value = {1. / max_norm_value}')


def sample_discrete_features(probX, probE, prob_charges, node_mask):
    ''' Sample features from multinomial distribution with given probabilities (probX, probE, proby)
        :param probX: bs, n, dx_out        node features
        :param probE: bs, n, n, de_out     edge features
        :param proby: bs, dy_out           global features.
    '''
    bs, n = node_mask.shape
    # Noise X
    # The masked rows should define probability distributions as well
    probX[~node_mask] = 1 / probX.shape[-1]
    prob_charges[~node_mask] = 1 / prob_charges.shape[-1]

    # Flatten the probability tensor to sample with multinomial
    probX = probX.reshape(bs * n, -1)       # (bs * n, dx_out)
    prob_charges = prob_charges.reshape(bs * n, -1)  # (bs * n, dx_out)

    # Sample X
    X_t = probX.multinomial(1)                                  # (bs * n, 1)
    X_t = X_t.reshape(bs, n)     # (bs, n)

    charges_t = prob_charges.multinomial(1)
    charges_t = charges_t.reshape(bs, n)

    # Noise E
    # The masked rows should define probability distributions as well
    inverse_edge_mask = ~(node_mask.unsqueeze(1) * node_mask.unsqueeze(2))
    diag_mask = torch.eye(n).unsqueeze(0).expand(bs, -1, -1)

    probE[inverse_edge_mask] = 1 / probE.shape[-1]
    probE[diag_mask.bool()] = 1 / probE.shape[-1]

    probE = probE.reshape(bs * n * n, -1)

    # Sample E
    E_t = probE.multinomial(1).reshape(bs, n, n)   # (bs, n, n)
    E_t = torch.triu(E_t, diagonal=1)
    E_t = (E_t + torch.transpose(E_t, 1, 2))

    return PlaceHolder(X=X_t, charges=charges_t, E=E_t, y=torch.zeros(bs, 0, device=X_t.device), pos=None)


def compute_posterior_distribution(M, M_t, Qt_M, Qsb_M, Qtb_M):
    ''' M: X, E or charges
        Compute xt @ Qt.T * x0 @ Qsb / x0 @ Qtb @ xt.T
    '''
    # Flatten feature tensors
    M = M.flatten(start_dim=1, end_dim=-2).to(torch.float32)        # (bs, N, d) with N = n or n * n
    M_t = M_t.flatten(start_dim=1, end_dim=-2).to(torch.float32)    # same

    Qt_M_T = torch.transpose(Qt_M, -2, -1)      # (bs, d, d)

    left_term = M_t @ Qt_M_T   # (bs, N, d)
    right_term = M @ Qsb_M     # (bs, N, d)
    product = left_term * right_term    # (bs, N, d)

    denom = M @ Qtb_M     # (bs, N, d) @ (bs, d, d) = (bs, N, d)
    denom = (denom * M_t).sum(dim=-1)   # (bs, N, d) * (bs, N, d) + sum = (bs, N)
    # denom = product.sum(dim=-1)
    # denom[denom == 0.] = 1

    # stabilizing never hurts
    prob = product / (1e-19 + denom.unsqueeze(-1))    # (bs, N, d)

    return prob


def compute_batched_over0_posterior_distribution(X_t, Qt, Qsb, Qtb):
    """ M: X, E or charges
        Compute xt @ Qt.T * x0 @ Qsb / x0 @ Qtb @ xt.T for each possible value of x0
        X_t: bs, n, dt          or bs, n, n, dt
        Qt: bs, d_t-1, dt
        Qsb: bs, d0, d_t-1
        Qtb: bs, d0, dt.
    """
    # Flatten feature tensors
    # Careful with this line. It does nothing if X is a node feature. If X is an edge features it maps to
    # bs x (n ** 2) x d
    X_t = X_t.flatten(start_dim=1, end_dim=-2).to(torch.float32)            # bs x N x dt

    Qt_T = Qt.transpose(-1, -2)                 # bs, dt, d_t-1
    left_term = X_t @ Qt_T                      # bs, N, d_t-1
    left_term = left_term.unsqueeze(dim=2)      # bs, N, 1, d_t-1

    right_term = Qsb.unsqueeze(1)               # bs, 1, d0, d_t-1
    numerator = left_term * right_term          # bs, N, d0, d_t-1

    X_t_transposed = X_t.transpose(-1, -2)      # bs, dt, N

    prod = Qtb @ X_t_transposed                 # bs, d0, N
    prod = prod.transpose(-1, -2)               # bs, N, d0
    denominator = prod.unsqueeze(-1)            # bs, N, d0, 1
    denominator[denominator == 0] = 1e-6

    out = numerator / denominator
    return out


def mask_distributions(probs, node_mask):
    # Set masked rows to arbitrary distributions, so it doesn't contribute to loss
    device = probs.X.device
    row_X = torch.zeros(probs.X.size(-1), dtype=torch.float, device=device)
    row_X[0] = 1.

    row_charges = torch.zeros(probs.charges.size(-1), dtype=torch.float, device=device)
    row_charges[0] = 1.

    row_E = torch.zeros(probs.E.size(-1), dtype=torch.float, device=device)
    row_E[0] = 1.

    probs.X[~node_mask] = row_X
    probs.charges[~node_mask] = row_charges

    diag_mask = ~torch.eye(node_mask.size(1), device=node_mask.device, dtype=torch.bool).unsqueeze(0)
    probs.E[~(node_mask.unsqueeze(1) * node_mask.unsqueeze(2) * diag_mask), :] = row_E

    probs.X = probs.X + 1e-7
    probs.X = probs.X / torch.sum(probs.X, dim=-1, keepdim=True)

    probs.charges = probs.charges + 1e-7
    probs.charges = probs.charges / torch.sum(probs.charges, dim=-1, keepdim=True)

    probs.E = probs.E + 1e-7
    probs.E = probs.E / torch.sum(probs.E, dim=-1, keepdim=True)
    return probs


def posterior_distributions(clean_data, noisy_data, Qt, Qsb, Qtb):
    prob_X = compute_posterior_distribution(M=clean_data.X, M_t=noisy_data.X,
                                            Qt_M=Qt.X, Qsb_M=Qsb.X, Qtb_M=Qtb.X)   # (bs, n, dx)
    prob_c = compute_posterior_distribution(M=clean_data.charges, M_t=noisy_data.charges,
                                            Qt_M=Qt.charges, Qsb_M=Qsb.charges, Qtb_M=Qtb.charges)   # (bs, n * n, de)
    prob_E = compute_posterior_distribution(M=clean_data.E, M_t=noisy_data.E,
                                            Qt_M=Qt.E, Qsb_M=Qsb.E, Qtb_M=Qtb.E)   # (bs, n * n, de)

    return PlaceHolder(X=prob_X, E=prob_E, charges=prob_c, y=None, pos=None)

