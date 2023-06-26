import torch
import midi.utils as utils


class ExtraFeatures:
    def __init__(self, extra_features_type, dataset_info):
        self.max_n_nodes = dataset_info.max_n_nodes
        self.ncycles = NodeCycleFeatures()
        self.features_type = extra_features_type
        if extra_features_type in ['eigenvalues', 'all']:
            self.eigenfeatures = EigenFeatures(mode=extra_features_type)

    def update_input_dims(self, input_dims):
        if self.features_type is None:
            return input_dims
        elif self.features_type == 'connectivity':
            input_dims.E += 1

        elif self.features_type == 'all':
            input_dims.X += 6
            input_dims.E += 1
            input_dims.y += 11
        else:
            raise NotImplementedError(f"'{self.features_type}' feature type not implemented.")
        return input_dims

    def __call__(self, z_t):
        pos = z_t.pos
        X = z_t.X
        E = z_t.E
        y = z_t.y
        charges = z_t.charges
        if self.features_type is None:
            empty_pos = pos.new_zeros((*pos.shape[:-1], 0))
            empty_x = X.new_zeros((*X.shape[:-1], 0))
            empty_e = E.new_zeros((*E.shape[:-1], 0))
            empty_y = y.new_zeros((y.shape[0], 0))
            empty_charges = z_t.charges.new_zeros((*charges.shape[:-1], 0))
            return utils.PlaceHolder(X=empty_x, charges=empty_charges, E=empty_e, y=empty_y, pos=empty_pos)

        elif self.features_type == 'connectivity':
            extra_E = connectivity_feature(z_t)
            empty_pos = pos.new_zeros((*pos.shape[:-1], 0))
            empty_x = X.new_zeros((*X.shape[:-1], 0))
            empty_y = y.new_zeros((y.shape[0], 0))
            empty_charges = z_t.charges.new_zeros((*charges.shape[:-1], 0))
            return utils.PlaceHolder(X=empty_x, charges=empty_charges, E=extra_E, y=empty_y, pos=empty_pos)

        n = z_t.node_mask.sum(dim=1).unsqueeze(1) / self.max_n_nodes
        x_cycles, y_cycles = self.ncycles(z_t)       # (bs, n_cycles)

        if self.features_type == 'cycles':
            E = z_t.E
            extra_edge_attr = torch.zeros((*E.shape[:-1], 0), device=E.device)
            return utils.PlaceHolder(X=x_cycles, E=extra_edge_attr, y=torch.hstack((n, y_cycles)))

        elif self.features_type == 'eigenvalues':
            eigenfeatures = self.eigenfeatures(z_t)
            E = z_t.E
            extra_edge_attr = torch.zeros((*E.shape[:-1], 0), device=E.device)
            n_components, batched_eigenvalues = eigenfeatures   # (bs, 1), (bs, 10)
            return utils.PlaceHolder(X=x_cycles, E=extra_edge_attr, y=torch.hstack((n, y_cycles, n_components,
                                                                                    batched_eigenvalues)))
        elif self.features_type == 'all':
            eigenfeatures = self.eigenfeatures(z_t)
            E = z_t.E
            extra_edge_attr = connectivity_feature(z_t).to(E.device)
            n_components, batched_eigenvalues, nonlcc_indicator, k_lowest_eigvec = eigenfeatures   # (bs, 1), (bs, 10),
                                                                                                # (bs, n, 1), (bs, n, 2)


            return utils.PlaceHolder(X=torch.cat((x_cycles, nonlcc_indicator, k_lowest_eigvec), dim=-1),
                                     E=extra_edge_attr,
                                     y=torch.hstack((n, y_cycles, n_components, batched_eigenvalues)),
                                     pos=None,
                                     charges=None)
        else:
            raise ValueError(f"Features type {self.features_type} not implemented")


class NodeCycleFeatures:
    def __init__(self):
        self.kcycles = KNodeCycles()

    def __call__(self, z_t):
        adj_matrix = z_t.E[..., 1:].sum(dim=-1).float()

        x_cycles, y_cycles = self.kcycles.k_cycles(adj_matrix=adj_matrix)   # (bs, n_cycles)
        x_cycles = x_cycles.to(adj_matrix.device) * z_t.node_mask.unsqueeze(-1)
        # Avoid large values when the graph is dense
        x_cycles = x_cycles / 10
        y_cycles = y_cycles / 10
        x_cycles[x_cycles > 1] = 1
        y_cycles[y_cycles > 1] = 1
        return x_cycles, y_cycles


class EigenFeatures:
    """
    Code taken from : https://github.com/Saro00/DGN/blob/master/models/pytorch/eigen_agg.py
    """
    def __init__(self, mode):
        """ mode: 'eigenvalues' or 'all' """
        self.mode = mode

    def __call__(self, z_t):
        E_t = z_t.E
        mask = z_t.node_mask
        A = E_t[..., 1:].sum(dim=-1).float() * mask.unsqueeze(1) * mask.unsqueeze(2)
        L = compute_laplacian(A, normalize=False)
        mask_diag = 2 * L.shape[-1] * torch.eye(A.shape[-1], device=L.device).unsqueeze(0)
        mask_diag = mask_diag * (~mask.unsqueeze(1)) * (~mask.unsqueeze(2))
        L = L * mask.unsqueeze(1) * mask.unsqueeze(2) + mask_diag

        if self.mode == 'eigenvalues':
            eigvals = torch.linalg.eigvalsh(L)        # bs, n
            eigvals = eigvals / torch.sum(mask, dim=1, keepdim=True)

            n_connected_comp, batch_eigenvalues = get_eigenvalues_features(eigenvalues=eigvals)
            return n_connected_comp, batch_eigenvalues

        elif self.mode == 'all':
            eigvals, eigvectors = torch.linalg.eigh(L)
            eigvals = eigvals / torch.sum(mask, dim=1, keepdim=True)
            eigvectors = eigvectors * mask.unsqueeze(2) * mask.unsqueeze(1)
            # Retrieve eigenvalues features
            n_connected_comp, batch_eigenvalues = get_eigenvalues_features(eigenvalues=eigvals)

            # Retrieve eigenvectors features
            nonlcc_indicator, k_lowest_eigenvector = get_eigenvectors_features(vectors=eigvectors,
                                                                               node_mask=z_t.node_mask,
                                                                               n_connected=n_connected_comp)
            return n_connected_comp, batch_eigenvalues, nonlcc_indicator, k_lowest_eigenvector
        else:
            raise NotImplementedError(f"Mode {self.mode} is not implemented")


def connectivity_feature(z_t, iterations=15):
    E_t = z_t.E
    mask = z_t.node_mask
    A = E_t[..., 1:].sum(dim=-1).float() * mask.unsqueeze(1) * mask.unsqueeze(2)
    current = A
    for i in range(iterations):
        current = A @ current
    current = (current == 0).float().unsqueeze(-1)
    return current


def compute_laplacian(adjacency, normalize: bool):
    """
    adjacency : batched adjacency matrix (bs, n, n)
    normalize: can be None, 'sym' or 'rw' for the combinatorial, symmetric normalized or random walk Laplacians
    Return:
        L (n x n ndarray): combinatorial or symmetric normalized Laplacian.
    """
    diag = torch.sum(adjacency, dim=-1)     # (bs, n)
    n = diag.shape[-1]
    D = torch.diag_embed(diag)      # Degree matrix      # (bs, n, n)
    combinatorial = D - adjacency                        # (bs, n, n)

    if not normalize:
        return (combinatorial + combinatorial.transpose(1, 2)) / 2

    diag0 = diag.clone()
    diag[diag == 0] = 1e-12

    diag_norm = 1 / torch.sqrt(diag)            # (bs, n)
    D_norm = torch.diag_embed(diag_norm)        # (bs, n, n)
    L = torch.eye(n).unsqueeze(0) - D_norm @ adjacency @ D_norm
    L[diag0 == 0] = 0
    return (L + L.transpose(1, 2)) / 2


def get_eigenvalues_features(eigenvalues, k=5):
    """
    values : eigenvalues -- (bs, n)
    node_mask: (bs, n)
    k: num of non zero eigenvalues to keep
    """
    ev = eigenvalues
    bs, n = ev.shape
    n_connected_components = (ev < 1e-5).sum(dim=-1)
    assert (n_connected_components > 0).all(), (n_connected_components, ev)

    to_extend = max(n_connected_components) + k - n
    if to_extend > 0:
        eigenvalues = torch.hstack((ev, 2 * torch.ones(bs, to_extend, device=ev.device)))
    indices = torch.arange(k, device=ev.device).unsqueeze(0) + n_connected_components.unsqueeze(1)
    first_k_ev = torch.gather(eigenvalues, dim=1, index=indices)
    return n_connected_components.unsqueeze(-1), first_k_ev


def get_eigenvectors_features(vectors, node_mask, n_connected, k=2):
    """
    vectors (bs, n, n) : eigenvectors of Laplacian IN COLUMNS
    returns:
        not_lcc_indicator : indicator vectors of largest connected component (lcc) for each graph  -- (bs, n, 1)
        k_lowest_eigvec : k first eigenvectors for the largest connected component   -- (bs, n, k)
    Warning: this function does not exactly return what is desired, the lcc might not be exactly the returned vector.
    """
    bs, n = vectors.size(0), vectors.size(1)

    # Create an indicator for the nodes outside the largest connected components
    k0 = min(n, 5)
    first_evs = vectors[:, :, :k0]                         # bs, n, k0
    quantized = torch.round(first_evs * 1000) / 1000       # bs, n, k0
    random_mask = 50 * torch.ones(bs, n, k0, device=vectors.device) * (~node_mask.unsqueeze(-1))         # bs, n, k0
    min_batched = torch.min(quantized + random_mask, dim=1).values.unsqueeze(1)       # bs, 1, k0
    max_batched = torch.max(quantized - random_mask, dim=1).values.unsqueeze(1)       # bs, 1, k0
    nonzero_mask = quantized.abs() >= 1e-5
    is_min = (quantized == min_batched) * nonzero_mask * node_mask.unsqueeze(2)                      # bs, n, k0
    is_max = (quantized == max_batched) * nonzero_mask * node_mask.unsqueeze(2)                      # bs, n, k0
    is_other = (quantized != min_batched) * (quantized != max_batched) * nonzero_mask * node_mask.unsqueeze(2)

    all_masks = torch.cat((is_min.unsqueeze(-1), is_max.unsqueeze(-1), is_other.unsqueeze(-1)), dim=3)    # bs, n, k0, 3
    all_masks = all_masks.flatten(start_dim=-2)      # bs, n, k0 x 3
    counts = torch.sum(all_masks, dim=1)      # bs, k0 x 3

    argmax_counts = torch.argmax(counts, dim=1)       # bs
    lcc_indicator = all_masks[torch.arange(bs), :, argmax_counts]                   # bs, n
    not_lcc_indicator = ((~lcc_indicator).float() * node_mask).unsqueeze(2)

    # Get the eigenvectors corresponding to the first nonzero eigenvalues
    to_extend = max(n_connected) + k - n
    if to_extend > 0:
        vectors = torch.cat((vectors, vectors.new_zeros((bs, n, to_extend))), dim=2)   # bs, n , n + to_extend
    indices = torch.arange(k, device=vectors.device)[None, None, :] + n_connected.unsqueeze(2)    # bs, 1, k
    indices = indices.expand(-1, n, -1)                                               # bs, n, k
    first_k_ev = torch.gather(vectors, dim=2, index=indices)       # bs, n, k
    first_k_ev = first_k_ev * node_mask.unsqueeze(2)

    return not_lcc_indicator, first_k_ev


def batch_trace(X):
    """
    Expect a matrix of shape B N N, returns the trace in shape B
    :param X:
    :return:
    """
    diag = torch.diagonal(X, dim1=-2, dim2=-1)
    trace = diag.sum(dim=-1)
    return trace


def batch_diagonal(X):
    """
    Extracts the diagonal from the last two dims of a tensor
    :param X:
    :return:
    """
    return torch.diagonal(X, dim1=-2, dim2=-1)


class KNodeCycles:
    """ Builds cycle counts for each node in a graph.
    """

    def __init__(self):
        super().__init__()

    def calculate_kpowers(self):
        self.k1_matrix = self.adj_matrix.float()
        self.d = self.adj_matrix.sum(dim=-1)
        self.k2_matrix = self.k1_matrix @ self.adj_matrix.float()
        self.k3_matrix = self.k2_matrix @ self.adj_matrix.float()
        self.k4_matrix = self.k3_matrix @ self.adj_matrix.float()
        self.k5_matrix = self.k4_matrix @ self.adj_matrix.float()
        self.k6_matrix = self.k5_matrix @ self.adj_matrix.float()

    def k3_cycle(self):
        """ tr(A ** 3). """
        c3 = batch_diagonal(self.k3_matrix)
        return (c3 / 2).unsqueeze(-1).float(), (torch.sum(c3, dim=-1) / 6).unsqueeze(-1).float()

    def k4_cycle(self):
        diag_a4 = batch_diagonal(self.k4_matrix)
        c4 = diag_a4 - self.d * (self.d - 1) - (self.adj_matrix @ self.d.unsqueeze(-1)).sum(dim=-1)
        return (c4 / 2).unsqueeze(-1).float(), (torch.sum(c4, dim=-1) / 8).unsqueeze(-1).float()

    def k5_cycle(self):
        diag_a5 = batch_diagonal(self.k5_matrix)
        triangles = batch_diagonal(self.k3_matrix)

        c5 = diag_a5 - 2 * triangles * self.d - (self.adj_matrix @ triangles.unsqueeze(-1)).sum(dim=-1) + triangles
        return (c5 / 2).unsqueeze(-1).float(), (c5.sum(dim=-1) / 10).unsqueeze(-1).float()

    def k6_cycle(self):
        term_1_t = batch_trace(self.k6_matrix)
        term_2_t = batch_trace(self.k3_matrix ** 2)
        term3_t = torch.sum(self.adj_matrix * self.k2_matrix.pow(2), dim=[-2, -1])
        d_t4 = batch_diagonal(self.k2_matrix)
        a_4_t = batch_diagonal(self.k4_matrix)
        term_4_t = (d_t4 * a_4_t).sum(dim=-1)
        term_5_t = batch_trace(self.k4_matrix)
        term_6_t = batch_trace(self.k3_matrix)
        term_7_t = batch_diagonal(self.k2_matrix).pow(3).sum(-1)
        term8_t = torch.sum(self.k3_matrix, dim=[-2, -1])
        term9_t = batch_diagonal(self.k2_matrix).pow(2).sum(-1)
        term10_t = batch_trace(self.k2_matrix)

        c6_t = (term_1_t - 3 * term_2_t + 9 * term3_t - 6 * term_4_t + 6 * term_5_t - 4 * term_6_t + 4 * term_7_t +
                3 * term8_t - 12 * term9_t + 4 * term10_t)
        return None, (c6_t / 12).unsqueeze(-1).float()

    def k_cycles(self, adj_matrix, verbose=False):
        self.adj_matrix = adj_matrix
        self.calculate_kpowers()

        k3x, k3y = self.k3_cycle()
        assert (k3x >= -0.1).all()

        k4x, k4y = self.k4_cycle()
        assert (k4x >= -0.1).all()

        k5x, k5y = self.k5_cycle()
        assert (k5x >= -0.1).all(), k5x

        _, k6y = self.k6_cycle()
        assert (k6y >= -0.1).all()

        kcyclesx = torch.cat([k3x, k4x, k5x], dim=-1)
        kcyclesy = torch.cat([k3y, k4y, k5y, k6y], dim=-1)
        return kcyclesx, kcyclesy