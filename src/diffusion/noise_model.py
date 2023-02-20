import torch
import torch.nn.functional as F
import numpy as np

import src.utils as utils
from src.diffusion import diffusion_utils


class NoiseModel:
    def __init__(self, noise_schedule, timesteps):
        # Define the transition matrices for the discrete features
        self.Px = None
        self.Pe = None
        self.Py = None
        self.Pcharges = None
        self.X_classes = None
        self.charges_classes = None
        self.E_classes = None
        self.y_classes = None
        self.X_marginals = None
        self.charges_marginals = None
        self.E_marginals = None
        self.y_marginals = None

        self.T = timesteps

        if noise_schedule == 'cosine':
            betas = diffusion_utils.cosine_beta_schedule_discrete(timesteps)
        elif noise_schedule == 'custom':
            betas = diffusion_utils.custom_beta_schedule_discrete(timesteps)
        else:
            raise NotImplementedError(noise_schedule)

        self.betas = torch.from_numpy(betas).float()
        self.alphas = 1 - torch.clamp(self.betas, min=0, max=0.9999)

        log_alpha = torch.log(self.alphas)
        log_alpha_bar = torch.cumsum(log_alpha, dim=0)
        self.alphas_bar = torch.exp(log_alpha_bar)
        self.sigma_bar = torch.sqrt(1 - self.alphas_bar ** 2)
        self.sigma2 = 1 - self.alphas_bar ** 2
        snr = (self.alphas_bar ** 2) / (1 - self.alphas_bar ** 2)
        self.gamma = - torch.log(snr)
        # print(f"[Noise schedule: {noise_schedule}] alpha_bar:", self.alphas_bar)

    def move_P_device(self, tensor):
        return self.Px.float().to(tensor.device), self.Pcharges.float().to(tensor.device),\
               self.Pe.float().to(tensor.device).float(), self.Py.float().to(tensor.device)

    def get_Qt(self, t_int):
        """ Returns one-step transition matrices for X and E, from step t - 1 to step t.
        Qt = (1 - beta_t) * I + beta_t / K

        beta_t: (bs)                         noise level between 0 and 1
        returns: qx (bs, dx, dx), qe (bs, de, de), qy (bs, dy, dy).
        """
        b = self.betas.to(t_int.device)
        beta_t = b[t_int].unsqueeze(1)
        Px, Pcharges, Pe, Py = self.move_P_device(t_int)

        q_x = beta_t * Px + (1 - beta_t) * torch.eye(self.X_classes,
                                                     device=t_int.device,
                                                     dtype=torch.float32).unsqueeze(0)
        q_charges = beta_t * Pcharges + (1 - beta_t) * torch.eye(self.charges_classes,
                                                                 device=t_int.device,
                                                                 dtype=torch.float32).unsqueeze(0)
        q_e = beta_t * Pe + (1 - beta_t) * torch.eye(self.E_classes,
                                                     device=t_int.device,
                                                     dtype=torch.float32).unsqueeze(0)
        q_y = beta_t * Py + (1 - beta_t) * torch.eye(self.y_classes,
                                                     device=t_int.device,
                                                     dtype=torch.float32).unsqueeze(0)

        return utils.PlaceHolder(X=q_x, charges=q_charges, E=q_e, y=q_y, pos=None)

    def get_Qt_bar(self, t_int):
        """ Returns t-step transition matrices for X and E, from step 0 to step t.
            Qt = prod(1 - beta_t) * I + (1 - prod(1 - beta_t)) / K

            alpha_bar_t: (bs)         Product of the (1 - beta_t) for each time step from 0 to t.
            returns: qx (bs, dx, dx), qe (bs, de, de), qy (bs, dy, dy).
        """
        alpha_bar_t = self.get_alpha_bar(t_int=t_int).unsqueeze(1)

        Px, Pc, Pe, Py = self.move_P_device(t_int)

        q_x = alpha_bar_t * torch.eye(self.X_classes, device=Px.device).unsqueeze(0) + (1 - alpha_bar_t) * Px
        q_c = alpha_bar_t * torch.eye(self.charges_classes, device=Px.device).unsqueeze(0) + (1 - alpha_bar_t) * Pc
        q_e = alpha_bar_t * torch.eye(self.E_classes, device=Px.device).unsqueeze(0) + (1 - alpha_bar_t) * Pe
        q_y = alpha_bar_t * torch.eye(self.y_classes, device=Px.device).unsqueeze(0) + (1 - alpha_bar_t) * Py

        assert ((q_x.sum(dim=2) - 1.).abs() < 1e-4).all(), q_x.sum(dim=2) - 1
        assert ((q_e.sum(dim=2) - 1.).abs() < 1e-4).all()

        return utils.PlaceHolder(X=q_x, charges=q_c, E=q_e, y=q_y, pos=None)

    def get_beta(self, t_normalized=None, t_int=None):
        assert int(t_normalized is None) + int(t_int is None) == 1
        if t_int is None:
            t_int = torch.round(t_normalized * self.T)
        return self.betas.to(t_int.device)[t_int.long()]

    def get_alpha_bar(self, t_normalized=None, t_int=None):
        assert int(t_normalized is None) + int(t_int is None) == 1
        if t_int is None:
            t_int = torch.round(t_normalized * self.T)
        return self.alphas_bar.to(t_int.device)[t_int.long()]

    def get_sigma_bar(self, t_normalized=None, t_int=None):
        assert int(t_normalized is None) + int(t_int is None) == 1
        if t_int is None:
            t_int = torch.round(t_normalized * self.T)
        return self.sigma_bar.to(t_int.device)[t_int]

    def sigma_t_over_s_sq(self, t_int, s_int):
        gamma = self.gamma.to(t_int.device)
        delta_soft = F.softplus(gamma[s_int]) - F.softplus(gamma[t_int])
        sigma_squared = - torch.expm1(delta_soft)
        return sigma_squared

    def get_ratio_sigma_ts(self, t_int, s_int):
        """ Compute sigma_t_over_s^2 / (1 - sigma_t_over_s^2)"""
        delta_soft = F.softplus(self.gamma[t_int]) - F.softplus(self.gamma[s_int])
        return torch.expm1(delta_soft)

    def get_alpha_t_over_s(self, t_int, s_int):
        a = self.alphas_bar.to(t_int.device)
        return a[t_int] / a[s_int]

    def apply_noise(self, dense_data):
        """ Sample noise and apply it to the data. """
        device = dense_data.X.device
        t_int = torch.randint(0, self.T + 1, size=(dense_data.X.size(0), 1), device=device)
        t_float = t_int.float() / self.T

        # beta_t and alpha_s_bar are used for denoising/loss computation
        alpha_t_bar = self.get_alpha_bar(t_int=t_int)      # (bs, 1)
        sigma_t_bar = self.get_sigma_bar(t_int=t_int)

        # Qtb returns two matrices of shape (bs, dx_in, dx_out) and (bs, de_in, de_out)
        Qtb = self.get_Qt_bar(t_int=t_int)

        # Compute transition probabilities
        probX = dense_data.X @ Qtb.X  # (bs, n, dx_out)
        prob_charges = dense_data.charges @ Qtb.charges
        probE = dense_data.E @ Qtb.E.unsqueeze(1)  # (bs, n, n, de_out)

        sampled_t = diffusion_utils.sample_discrete_features(probX=probX, probE=probE, prob_charges=prob_charges,
                                                             node_mask=dense_data.node_mask)

        X_t = F.one_hot(sampled_t.X, num_classes=self.X_classes)
        E_t = F.one_hot(sampled_t.E, num_classes=self.E_classes)
        charges_t = F.one_hot(sampled_t.charges, num_classes=self.charges_classes)
        assert (dense_data.X.shape == X_t.shape) and (dense_data.E.shape == E_t.shape)

        noise_pos = torch.randn(dense_data.pos.shape, device=dense_data.pos.device)
        noise_pos_masked = noise_pos * dense_data.node_mask.unsqueeze(-1)
        noise_pos_masked = utils.remove_mean_with_mask(noise_pos_masked, dense_data.node_mask)

        a = alpha_t_bar.unsqueeze(-1)
        s = sigma_t_bar.unsqueeze(-1)
        pos_t = a * dense_data.pos + s * noise_pos_masked

        z_t = utils.PlaceHolder(X=X_t, charges=charges_t, E=E_t, y=dense_data.y, pos=pos_t, t_int=t_int,
                                t=t_float, node_mask=dense_data.node_mask).mask()
        return z_t

    def get_limit_dist(self):
        X_marginals = self.X_marginals + 1e-7
        X_marginals = X_marginals / torch.sum(X_marginals)
        E_marginals = self.E_marginals + 1e-7
        E_marginals = E_marginals / torch.sum(E_marginals)
        charges_marginals = self.charges_marginals + 1e-7
        charges_marginals = charges_marginals / torch.sum(charges_marginals)
        limit_dist = utils.PlaceHolder(X=X_marginals, E=E_marginals, charges=charges_marginals,
                                       y=None, pos=None)
        return limit_dist

    def sample_limit_dist(self, node_mask):
        """ Sample from the limit distribution of the diffusion process"""

        bs, n_max = node_mask.shape
        x_limit = self.X_marginals.expand(bs, n_max, -1)
        e_limit = self.E_marginals[None, None, None, :].expand(bs, n_max, n_max, -1)
        charges_limit = self.charges_marginals.expand(bs, n_max, -1)

        U_X = x_limit.flatten(end_dim=-2).multinomial(1).reshape(bs, n_max).to(node_mask.device)
        U_c = charges_limit.flatten(end_dim=-2).multinomial(1).reshape(bs, n_max).to(node_mask.device)
        U_E = e_limit.flatten(end_dim=-2).multinomial(1).reshape(bs, n_max, n_max).to(node_mask.device)
        U_y = torch.zeros((bs, 0), device=node_mask.device)

        U_X = F.one_hot(U_X, num_classes=x_limit.shape[-1]).float()
        U_E = F.one_hot(U_E, num_classes=e_limit.shape[-1]).float()
        U_c = F.one_hot(U_c, num_classes=charges_limit.shape[-1]).float()

        # Get upper triangular part of edge noise, without main diagonal
        upper_triangular_mask = torch.zeros_like(U_E)
        indices = torch.triu_indices(row=U_E.size(1), col=U_E.size(2), offset=1)
        upper_triangular_mask[:, indices[0], indices[1], :] = 1

        U_E = U_E * upper_triangular_mask
        U_E = (U_E + torch.transpose(U_E, 1, 2))
        assert (U_E == torch.transpose(U_E, 1, 2)).all()

        pos = torch.randn(node_mask.shape[0], node_mask.shape[1], 3, device=node_mask.device)
        pos = pos * node_mask.unsqueeze(-1)
        pos = utils.remove_mean_with_mask(pos, node_mask)

        t_array = pos.new_ones((pos.shape[0], 1))
        t_int_array = self.T * t_array.long()
        return utils.PlaceHolder(X=U_X, charges=U_c, E=U_E, y=U_y, pos=pos, t_int=t_int_array, t=t_array,
                                 node_mask=node_mask).mask(node_mask)

    def sample_zs_from_zt_and_pred(self, z_t, pred, s_int):
        """Samples from zs ~ p(zs | zt). Only used during sampling.
           if last_step, return the graph prediction as well"""
        bs, n, dxs = z_t.X.shape
        node_mask = z_t.node_mask
        t_int = z_t.t_int

        # Retrieve transitions matrix
        Qtb = self.get_Qt_bar(t_int=t_int)
        Qsb = self.get_Qt_bar(t_int=s_int)
        Qt = self.get_Qt(t_int)

        # Sample the positions
        s2 = self.sigma2.to(s_int.device)
        a = self.alphas_bar.to(s_int.device)
        z_t_prefactor = self.get_alpha_t_over_s(t_int=t_int, s_int=s_int) * s2[s_int] / s2[t_int]
        z_t_prefactor = z_t_prefactor.unsqueeze(-1)
        x_prefactor = a[s_int] * self.sigma_t_over_s_sq(t_int=t_int, s_int=s_int) / s2[t_int]
        x_prefactor = x_prefactor.unsqueeze(-1)

        mu = z_t_prefactor * z_t.pos + x_prefactor * pred.pos

        sampled_pos = torch.randn(z_t.pos.shape, device=z_t.pos.device) * node_mask.unsqueeze(-1)
        noise = utils.remove_mean_with_mask(sampled_pos, node_mask=node_mask)
        s = self.sigma_bar.to(t_int.device)
        noise_prefactor = (torch.sqrt(self.sigma_t_over_s_sq(t_int=t_int, s_int=s_int)) *
                           s[s_int] / s[t_int]).unsqueeze(-1)

        pos = mu + noise_prefactor * noise

        # Normalize predictions for the categorical features
        pred_X = F.softmax(pred.X, dim=-1)               # bs, n, d0
        pred_E = F.softmax(pred.E, dim=-1)               # bs, n, n, d0
        pred_charges = F.softmax(pred.charges, dim=-1)

        p_s_and_t_given_0_X = diffusion_utils.compute_batched_over0_posterior_distribution(X_t=z_t.X,
                                                                                           Qt=Qt.X,
                                                                                           Qsb=Qsb.X,
                                                                                           Qtb=Qtb.X)

        p_s_and_t_given_0_E = diffusion_utils.compute_batched_over0_posterior_distribution(X_t=z_t.E,
                                                                                           Qt=Qt.E,
                                                                                           Qsb=Qsb.E,
                                                                                           Qtb=Qtb.E)
        p_s_and_t_given_0_c = diffusion_utils.compute_batched_over0_posterior_distribution(X_t=z_t.charges,
                                                                                           Qt=Qt.charges,
                                                                                           Qsb=Qsb.charges,
                                                                                           Qtb=Qtb.charges)

        # Dim of these two tensors: bs, N, d0, d_t-1
        weighted_X = pred_X.unsqueeze(-1) * p_s_and_t_given_0_X         # bs, n, d0, d_t-1
        unnormalized_prob_X = weighted_X.sum(dim=2)                     # bs, n, d_t-1
        unnormalized_prob_X[torch.sum(unnormalized_prob_X, dim=-1) == 0] = 1e-5
        prob_X = unnormalized_prob_X / torch.sum(unnormalized_prob_X, dim=-1, keepdim=True)  # bs, n, d_t-1

        weighted_c = pred_charges.unsqueeze(-1) * p_s_and_t_given_0_c         # bs, n, d0, d_t-1
        unnormalized_prob_c = weighted_c.sum(dim=2)                     # bs, n, d_t-1
        unnormalized_prob_c[torch.sum(unnormalized_prob_c, dim=-1) == 0] = 1e-5
        prob_c = unnormalized_prob_c / torch.sum(unnormalized_prob_c, dim=-1, keepdim=True)  # bs, n, d_t-1

        pred_E = pred_E.reshape((bs, -1, pred_E.shape[-1]))
        weighted_E = pred_E.unsqueeze(-1) * p_s_and_t_given_0_E        # bs, N, d0, d_t-1
        unnormalized_prob_E = weighted_E.sum(dim=-2)
        unnormalized_prob_E[torch.sum(unnormalized_prob_E, dim=-1) == 0] = 1e-5
        prob_E = unnormalized_prob_E / torch.sum(unnormalized_prob_E, dim=-1, keepdim=True)
        prob_E = prob_E.reshape(bs, n, n, pred_E.shape[-1])

        assert ((prob_X.sum(dim=-1) - 1).abs() < 1e-4).all()
        assert ((prob_c.sum(dim=-1) - 1).abs() < 1e-4).all()
        assert ((prob_E.sum(dim=-1) - 1).abs() < 1e-4).all()

        sampled_s = diffusion_utils.sample_discrete_features(prob_X, prob_E, prob_c, node_mask=z_t.node_mask)

        X_s = F.one_hot(sampled_s.X, num_classes=self.X_classes).float()
        charges_s = F.one_hot(sampled_s.charges, num_classes=self.charges_classes).float()
        E_s = F.one_hot(sampled_s.E, num_classes=self.E_classes).float()

        assert (E_s == torch.transpose(E_s, 1, 2)).all()
        assert (z_t.X.shape == X_s.shape) and (z_t.E.shape == E_s.shape)

        z_s = utils.PlaceHolder(X=X_s, charges=charges_s,
                                E=E_s, y=torch.zeros(z_t.y.shape[0], 0, device=X_s.device), pos=pos,
                                t_int=s_int, t=s_int / self.T, node_mask=node_mask).mask(node_mask)
        return z_s


class DiscreteUniformTransition(NoiseModel):
    def __init__(self, output_dims, noise_schedule, timesteps):
        super().__init__(noise_schedule=noise_schedule, timesteps=timesteps)
        self.X_classes = output_dims.X
        self.charges_classes = output_dims.charges
        self.E_classes = output_dims.E
        self.y_classes = output_dims.y
        self.X_marginals = torch.ones(self.X_classes) / self.X_classes
        self.charges_marginals = torch.ones(self.charges_classes) / self.charges_classes
        self.E_marginals = torch.ones(self.E_classes) / self.E_classes
        self.y_marginals = torch.ones(self.y_classes) / self.y_classes
        self.Px = torch.ones(1, self.X_classes, self.X_classes) / self.X_classes
        self.Pcharges = torch.ones(1, self.charges_classes, self.charges_classes) / self.charges_classes
        self.Pe = torch.ones(1, self.E_classes, self.E_classes) / self.E_classes
        self.Pe = torch.ones(1, self.y_classes, self.y_classes) / self.y_classes


class MarginalUniformTransition(NoiseModel):
    def __init__(self, x_marginals, e_marginals, charges_marginals, y_classes, noise_schedule, timesteps):
        super().__init__(noise_schedule=noise_schedule, timesteps=timesteps)
        self.X_classes = len(x_marginals)
        self.E_classes = len(e_marginals)
        self.charges_classes = len(charges_marginals)
        self.y_classes = y_classes
        self.X_marginals = x_marginals
        self.E_marginals = e_marginals
        self.charges_marginals = charges_marginals
        self.y_marginals = torch.ones(self.y_classes) / self.y_classes

        self.Px = x_marginals.unsqueeze(0).expand(self.X_classes, -1).unsqueeze(0)
        self.Pe = e_marginals.unsqueeze(0).expand(self.E_classes, -1).unsqueeze(0)
        self.Pcharges = charges_marginals.unsqueeze(0).expand(self.charges_classes, -1).unsqueeze(0)
        self.Py = torch.ones(1, self.y_classes, self.y_classes) / self.y_classes

