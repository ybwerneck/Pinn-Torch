import sys
import os
import numpy as np


def make_phi_true(vertices, stim_point, D=None):
    """
    Exact activation time map for a point stimulus.

    Solves the anisotropic eikonal  sqrt(∇φ · D ∇φ) = 1  analytically
    for a constant conductivity tensor D and a point source.

    The solution is the Riemannian distance in the D-metric:
        φ(x) = sqrt( (x - x_s)^T  D^{-1}  (x - x_s) )

    Derivation: with Cholesky D = L L^T,  D^{-1} = L^{-T} L^{-1}, so
        φ(x) = || L^{-1} (x - x_s) ||

    isotropic  (D=None or D=I) → ordinary Euclidean distance.
    anisotropic (constant D)   → elliptic wavefront, exact.

    Parameters
    ----------
    vertices   : (N, 2) array — mesh node coordinates
    stim_point : (2,)   array — stimulus location
    D          : (2, 2) array or None

    Returns
    -------
    phi : (N,) array, gauge-fixed so phi.min() = 0
    """
    stim = np.array(stim_point, dtype=np.float64)
    diff = vertices - stim                           # (N, 2)

    if D is None:
        phi = np.linalg.norm(diff, axis=1)
    else:
        D = np.array(D, dtype=np.float64)
        L = np.linalg.cholesky(D)                    # D = L L^T
        Linv_diff = np.linalg.solve(L, diff.T).T     # (N, 2)
        phi = np.linalg.norm(Linv_diff, axis=1)

    phi -= phi.min()
    return phi


def make_phi_true_multisource(vertices, stim_points, D=None):
    """
    Exact activation map for simultaneous point stimuli.

    Each source fires at t=0; the wavefront from source s reaches node x
    at time φ_s(x) = Riemannian distance(x, s).  The activation time is
    the earliest arrival:

        φ(x) = min_s  φ_s(x)

    isotropic  (D=None) → min Euclidean distance to any source.
    anisotropic (constant D) → min elliptic distance.

    Parameters
    ----------
    vertices    : (N, 2) array
    stim_points : list of (2,) array-likes — source locations
    D           : (2, 2) array or None

    Returns
    -------
    phi : (N,) array, gauge-fixed so phi.min() = 0
    """
    phi_per_source = []
    for stim in stim_points:
        phi_per_source.append(make_phi_true(vertices, stim, D=D))
    # undo per-source gauge fix before taking the min
    # (make_phi_true already subtracts its own min which is 0 at the source)
    phi = np.min(np.stack(phi_per_source, axis=1), axis=1)
    phi -= phi.min()
    return phi


def make_t_grid(phi_true, Nt=100, margin=1.2):
    """
    Time grid that covers the full activation range.
    T = margin * max(phi_true) ensures late-activating regions
    still contribute signal to the ECG leads.

    Parameters
    ----------
    phi_true : (N,) array
    Nt       : int   — number of time steps
    margin   : float — overshoot factor

    Returns
    -------
    t_grid : (Nt,) array, starting at 0
    """
    T = margin * phi_true.max()
    return np.linspace(0.0, T, Nt)


# ------------------------------------------------------------------
# Lead field model
# ------------------------------------------------------------------

def make_electrodes(L=1.0, h=0.3, n_elec=9):
    """
    Regular grid of electrodes above [0,L]² at height h.
    Returns (n_elec, 3) array of (x, y, z) positions.
    """
    side = int(np.ceil(np.sqrt(n_elec)))
    xs   = np.linspace(0.15, L - 0.15, side)
    ys   = np.linspace(0.15, L - 0.15, side)
    pos  = [(x, y, h) for x in xs for y in ys]
    return np.array(pos[:n_elec])


def precompute_lead_gradients(grid, electrodes, sigma=1.0):
    """
    Analytic gradient of the lead field at each triangle centroid.

        Z_l(x,y) = 1 / (4π σ r),   r = √((x-xe)²+(y-ye)²+h²)
        ∇Z_l     = -(x-xe, y-ye) / (4π σ r³)

    Parameters
    ----------
    grid       : Grid
    electrodes : (n_elec, 3) — (x, y, z) positions
    sigma      : float — bath conductivity

    Returns
    -------
    grad_Z : (n_elec, F, 2)
    """
    v, f      = grid.vertices, grid.faces
    centroids = (v[f[:, 0]] + v[f[:, 1]] + v[f[:, 2]]) / 3.0   # (F, 2)

    n_elec = len(electrodes)
    grad_Z = np.zeros((n_elec, len(f), 2))

    for l, (xe, ye, h) in enumerate(electrodes):
        dx = centroids[:, 0] - xe
        dy = centroids[:, 1] - ye
        r3 = (dx**2 + dy**2 + h**2) ** 1.5
        c  = -1.0 / (4.0 * np.pi * sigma * r3)
        grad_Z[l, :, 0] = c * dx
        grad_Z[l, :, 1] = c * dy

    return grad_Z


def _vtilde_prime(xi, V0=-80.0, V1=20.0):
    """Ṽ'(ξ) = (V1-V0)/2 · sech²(ξ)  — time derivative of the action potential."""
    return 0.5 * (V1 - V0) / np.cosh(xi) ** 2


def ecg_forward(phi, grid, grad_Z, t_grid, G_in=1.0, V0=-80.0, V1=20.0):
    """
    Rapid lead-field ECG forward model.

        V_l(t) = Σ_T  G_in · Ṽ'(t − φ_T) · (∇φ_T · ∇Z_l_T) · A_T

    Uses grid.gradient() for ∇φ and the precomputed ∇Z_l per triangle.

    Parameters
    ----------
    phi    : (N,) activation times at nodes (numpy)
    grid   : Grid
    grad_Z : (n_elec, F, 2) from precompute_lead_gradients()
    t_grid : (Nt,)
    G_in   : scalar intracellular conductivity

    Returns
    -------
    V : (n_elec, Nt)
    """
    f = grid.faces

    # Activation time and gradient per triangle
    phi_T    = (phi[f[:, 0]] + phi[f[:, 1]] + phi[f[:, 2]]) / 3.0  # (F,)
    grad_phi = grid.gradient(phi)                                     # (F, 2)

    # dot(∇φ_T, ∇Z_l_T) · area_T  →  (n_elec, F)
    dot_w = np.einsum('efd,fd->ef', grad_Z, grad_phi) * grid._areas[None, :]
    dot_w *= G_in

    # Vectorise over time: src (Nt, F), V (n_elec, Nt)
    xi  = t_grid[:, None] - phi_T[None, :]        # (Nt, F)
    src = _vtilde_prime(xi, V0, V1)               # (Nt, F)
    V   = dot_w @ src.T                           # (n_elec, Nt)

    return V


# ------------------------------------------------------------------
# Standalone verification
# ------------------------------------------------------------------

if __name__ == '__main__':
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
    import matplotlib.pyplot as plt
    from fisiocomPinn import Grid, structured_mesh
    from visualizer import Visualizer

    # Mesh
    vertices, faces = structured_mesh(n=33, L=1.0)
    grid = Grid(vertices, faces)
    viz  = Visualizer(grid)

    # Ground truth activation map (anisotropic)
    D_ANISO  = np.array([[3.0, 0.5], [0.5, 1.0]])
    phi_true = make_phi_true(vertices, stim_point=(0.05, 0.05), D=D_ANISO)
    t_grid   = make_t_grid(phi_true, Nt=100)

    # Electrodes + lead field
    electrodes = make_electrodes(L=1.0, h=0.3, n_elec=9)
    grad_Z     = precompute_lead_gradients(grid, electrodes)

    # ECG from ground truth
    V_meas = ecg_forward(phi_true, grid, grad_Z, t_grid)
    print(f"V_meas shape: {V_meas.shape}  range: {V_meas.min():.4f} – {V_meas.max():.4f}")

    # Direction field at nodes (average adjacent triangle gradients, then normalise)
    grad_phi    = grid.gradient(phi_true)           # (F, 2)
    p_nodes     = np.zeros((grid.N, 2))
    count       = np.zeros(grid.N)
    for k in range(3):
        np.add.at(p_nodes, grid.faces[:, k], grad_phi)
        np.add.at(count,   grid.faces[:, k], 1)
    p_nodes /= count[:, None]
    p_nodes /= (np.linalg.norm(p_nodes, axis=1, keepdims=True) + 1e-8)

    # Summary plot
    fig = viz.plot_summary(phi_true, p_nodes, t_grid, V_meas,
                           electrodes=electrodes, n_leads_shown=9)
    fig.savefig('ground_truth_summary.png', dpi=120)
    print("Saved ground_truth_summary.png")

    # All leads
    fig2 = viz.plot_leads(t_grid, V_meas)
    fig2.savefig('ground_truth_leads.png', dpi=120)
    print("Saved ground_truth_leads.png")

    plt.show()
