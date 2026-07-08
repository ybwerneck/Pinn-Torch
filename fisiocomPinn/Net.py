from .dependencies import *


class FullyConnectedNetwork(nn.Module):
    def __init__(self, input_shape, output_shape, hidden_sizes=0, dtype=torch.float32):
        super(FullyConnectedNetwork, self).__init__()
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.dtype = dtype
        self.layers = nn.ModuleList()
        if hidden_sizes != 0:
            self.initFullyConnected(input_shape, output_shape, hidden_sizes)

    def initFullyConnected(self, input_shape, output_shape, hidden_sizes):
        in_features = input_shape

        for hidden_size in hidden_sizes:
            self.layers.append(nn.Linear(in_features, hidden_size, bias=True))
            self.layers.append(nn.Tanh())  # Adding ReLU activation function
            in_features = hidden_size

        self.layers.append(nn.Linear(in_features, output_shape, bias=True))

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)

        return x


class EigenDirectionNet(FullyConnectedNetwork):
    """
    Direction-field network for Delta-PINNs on triangular meshes.

    Follows Costabal et al. (2022): the input x is the eigenfunction values
    at the queried nodes — the caller decides which nodes (full mesh or a
    mini-batch of sampled nodes). The model is agnostic to the sampling.

    Output is a unit-normalised direction field (n_nodes, 2).
    The eps guard prevents NaN gradients when p_raw passes through zero.

    Parameters
    ----------
    Ne     : int   — number of eigenfunctions (input dimension)
    n_layers : int — number of hidden layers
    width  : int   — neurons per hidden layer
    eps    : float — unit-normalisation denominator guard
    dtype  : torch dtype
    """

    def __init__(self, Ne, n_layers, width, eps=1e-8, dtype=torch.float32):
        super().__init__(
            input_shape=Ne,
            output_shape=2,
            hidden_sizes=[width] * n_layers,
            dtype=dtype,
        )
        self.eps = eps

    def forward(self, x):
        p_raw = super().forward(x)                       # (n_nodes, 2)
        norm  = torch.norm(p_raw, dim=-1, keepdim=True)  # (n_nodes, 1)
        return p_raw / (norm + self.eps)                  # (n_nodes, 2) unit vectors


class EnsembleNet(nn.Module):
    """
    M independently initialised EigenDirectionNets trained in parallel.

    forward(x) returns (M, N_nodes, 2) — stacked outputs from all members.
    Because members share no parameters, each member's gradient comes only
    from its own loss component; training is equivalent to M independent runs.

    Parameters
    ----------
    M        : int — number of ensemble members
    Ne       : int — eigenfunctions (input dimension)
    n_layers : int
    width    : int
    **kwargs : forwarded to EigenDirectionNet (eps, dtype, …)
    """

    def __init__(self, M, Ne, n_layers, width, **kwargs):
        super().__init__()
        self.members = nn.ModuleList([
            EigenDirectionNet(Ne, n_layers, width, **kwargs)
            for _ in range(M)
        ])

    def __len__(self):
        return len(self.members)

    def __getitem__(self, i):
        return self.members[i]

    def forward(self, x):
        return torch.stack([m(x) for m in self.members], dim=0)  # (M, N, 2)


activation_map = {
    "Elu": nn.ELU,
    "LeakyReLU": nn.LeakyReLU,
    "Sigmoid": nn.Sigmoid,
    "Softplus": nn.Softplus,
    "Tanh": nn.Tanh,
    "Linear": nn.Linear,
    "ReLU": nn.ReLU,
    "RReLU": nn.RReLU,
    "SELU": nn.SELU,
    "CELU": nn.CELU,
    "GELU": nn.GELU,
    "SiLU": nn.SiLU,
    "GLU": nn.GLU,
}


class FullyConnectedNetworkMod(FullyConnectedNetwork):
    def __init__(self, input_shape, output_shape, hidden_sizes, dtype=torch.float32):
        super(FullyConnectedNetworkMod, self).__init__(
            input_shape, output_shape, dtype=dtype
        )
        in_features = input_shape

        for act, hidden_size in hidden_sizes:

            act = activation_map[act]
            self.layers.append(
                nn.Linear(in_features, hidden_size, bias=True, dtype=self.dtype)
            )

            self.layers.append(act())  # Adding ReLU activation function
            in_features = hidden_size

        self.layers.append(
            nn.Linear(in_features, output_shape, bias=True, dtype=self.dtype)
        )

        self.layers.append(nn.Tanh())
