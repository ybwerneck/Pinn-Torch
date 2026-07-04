import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as tri
import h5py


class Visualizer:
    """
    Plotting utilities for the delta-Poisson mock.
    All methods accept a Grid instance at construction and work directly
    with nodal or per-triangle fields produced by that grid.
    """

    def __init__(self, grid):
        self.grid = grid
        self.triang = tri.Triangulation(
            grid.vertices[:, 0],
            grid.vertices[:, 1],
            grid.faces,
        )

    # ------------------------------------------------------------------
    # Mesh
    # ------------------------------------------------------------------

    def plot_mesh(self, ax=None, **kwargs):
        """Wireframe of the triangulation."""
        ax = ax or plt.gca()
        ax.triplot(self.triang, color='k', linewidth=0.4, **kwargs)
        ax.set_aspect('equal')
        ax.set_title('Mesh')
        return ax

    # ------------------------------------------------------------------
    # Scalar fields
    # ------------------------------------------------------------------

    def plot_field(self, f, title='', ax=None, cmap='RdBu_r', show_mesh=False):
        """
        Colour-map a scalar field on the mesh nodes.

        Parameters
        ----------
        f     : (N,) array — nodal values (numpy or torch)
        title : str
        """
        f = _to_np(f)
        ax = ax or plt.gca()
        tc = ax.tripcolor(self.triang, f, shading='gouraud', cmap=cmap)
        if show_mesh:
            ax.triplot(self.triang, color='k', linewidth=0.2, alpha=0.3)
        plt.colorbar(tc, ax=ax)
        ax.set_aspect('equal')
        ax.set_title(title)
        return ax

    def plot_eigenfunctions(self, vecs, n_cols=4, figsize=None):
        """
        Grid of eigenfunction plots.

        Parameters
        ----------
        vecs   : (N, n_eig) array
        n_cols : int — columns in the subplot grid
        """
        vecs   = _to_np(vecs)
        n_eig  = vecs.shape[1]
        n_rows  = int(np.ceil(n_eig / n_cols))
        figsize = figsize or (3 * n_cols, 3 * n_rows)

        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
        axes = np.array(axes).ravel()

        for i in range(n_eig):
            self.plot_field(vecs[:, i], title=f'ψ_{i}', ax=axes[i])

        for ax in axes[n_eig:]:
            ax.set_visible(False)

        fig.suptitle('Laplace-Beltrami eigenfunctions', y=1.01)
        fig.tight_layout()
        return fig

    # ------------------------------------------------------------------
    # Vector fields
    # ------------------------------------------------------------------

    def plot_vector_field(self, q, title='', ax=None, stride=1, scale=None):
        """
        Quiver plot of a vector field at mesh nodes.

        Parameters
        ----------
        q      : (N, 2) array
        stride : int — subsample every `stride`-th node (reduces clutter)
        """
        q  = _to_np(q)
        ax = ax or plt.gca()
        x, y = self.grid.vertices[::stride, 0], self.grid.vertices[::stride, 1]
        u, v = q[::stride, 0], q[::stride, 1]
        ax.quiver(x, y, u, v, scale=scale, scale_units='xy', angles='xy')
        ax.set_aspect('equal')
        ax.set_title(title)
        return ax

    # ------------------------------------------------------------------
    # Summary: phi + direction field + ECG
    # ------------------------------------------------------------------

    def plot_summary(self, phi, p, t_grid, V, electrodes=None, V_pred=None,
                     n_leads_shown=4, figsize=(16, 4)):
        """
        Three-panel figure: activation map | direction field | ECG leads.

        Parameters
        ----------
        phi          : (N,) nodal activation times
        p            : (N, 2) or (F, 2) direction field (nodes or triangles)
        t_grid       : (Nt,)
        V            : (n_elec, Nt) ECG leads (ground truth / measured)
        electrodes   : (n_elec, 3) optional, overlaid on phi panel
        V_pred       : (n_elec, Nt) optional predicted leads
        n_leads_shown: how many leads to draw in the ECG panel
        """
        fig, axes = plt.subplots(1, 3, figsize=figsize)

        # Left: phi
        self.plot_field(phi, title='φ (activation time)', ax=axes[0], cmap='hot')
        if electrodes is not None:
            axes[0].scatter(electrodes[:, 0], electrodes[:, 1],
                            c='cyan', s=40, zorder=5, label='electrodes')
            axes[0].legend(fontsize=7)

        # Middle: direction field
        self.plot_vector_field(p, title='p (direction field)', ax=axes[1], stride=2)

        # Right: ECG leads (first n_leads_shown)
        ax = axes[2]
        cmap = plt.cm.tab10
        for i in range(min(n_leads_shown, len(V))):
            color = cmap(i / max(n_leads_shown, 1))
            ax.plot(t_grid, _to_np(V[i]), color=color,
                    linewidth=1.2, label=f'L{i}')
            if V_pred is not None:
                ax.plot(t_grid, _to_np(V_pred[i]), color=color,
                        linewidth=1.2, linestyle='--')
        ax.set_title('ECG leads')
        ax.set_xlabel('t')
        ax.legend(fontsize=7)

        fig.tight_layout()
        return fig

    # ------------------------------------------------------------------
    # ECG leads (time series)
    # ------------------------------------------------------------------

    def plot_leads(self, t_grid, V_meas, V_pred=None, figsize=None):
        """
        One subplot per lead.

        Parameters
        ----------
        t_grid : (Nt,)
        V_meas : (n_leads, Nt)
        V_pred : (n_leads, Nt) or None
        """
        V_meas = _to_np(V_meas)
        n_leads = V_meas.shape[0]
        n_cols  = min(4, n_leads)
        n_rows  = int(np.ceil(n_leads / n_cols))
        figsize = figsize or (4 * n_cols, 3 * n_rows)

        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize, sharex=True)
        axes = np.array(axes).ravel()

        for i in range(n_leads):
            axes[i].plot(t_grid, V_meas[i], 'k', label='meas', linewidth=1.2)
            if V_pred is not None:
                axes[i].plot(t_grid, _to_np(V_pred[i]), 'r--', label='pred', linewidth=1.2)
            axes[i].set_title(f'Lead {i}')
            if i == 0:
                axes[i].legend(fontsize=7)

        for ax in axes[n_leads:]:
            ax.set_visible(False)

        fig.tight_layout()
        return fig


    # ------------------------------------------------------------------
    # HDF5 snapshot reader
    # ------------------------------------------------------------------

    def plot_from_h5(self, h5_path, t_grid, n_leads_shown=4, figsize=(16, 4)):
        """
        Read a validator snapshot written by ECGValidator and produce a
        three-panel comparison: φ_hat vs φ_true error map | ECG overlay.

        Parameters
        ----------
        h5_path      : str — path to a snapshot HDF5 (ecg_val_XXXXXX.h5)
        t_grid       : (Nt,) array — time grid for ECG axis
        n_leads_shown: int — number of ECG leads to draw
        """
        with h5py.File(h5_path, "r") as hf:
            phi_hat  = np.array(hf["phi_hat"])
            phi_true = np.array(hf["phi_true"])
            V_pred   = np.array(hf["V_pred"])
            V_meas   = np.array(hf["V_meas"])

        fig, axes = plt.subplots(1, 4, figsize=figsize)

        self.plot_field(phi_true,                   title='φ true',  ax=axes[0], cmap='hot')
        self.plot_field(phi_hat,                    title='φ pred',  ax=axes[1], cmap='hot')
        self.plot_field(np.abs(phi_hat - phi_true), title='|error|', ax=axes[2], cmap='Reds')

        ax = axes[3]
        cmap = plt.cm.tab10
        for i in range(min(n_leads_shown, len(V_meas))):
            c = cmap(i / max(n_leads_shown, 1))
            ax.plot(t_grid, V_meas[i], color=c, linewidth=1.2, label=f'L{i}')
            ax.plot(t_grid, V_pred[i], color=c, linewidth=1.2, linestyle='--')
        ax.set_title('ECG: meas (—) vs pred (--)')
        ax.set_xlabel('t')
        ax.legend(fontsize=7)

        fig.tight_layout()
        return fig

    @staticmethod
    def plot_err_h5(err_h5_path, figsize=(6, 3)):
        """
        Plot mean and max φ error over validation calls from the
        *_err.h5 file written by ECGValidator.

        Parameters
        ----------
        err_h5_path : str — path to *_err.h5
        """
        with h5py.File(err_h5_path, "r") as hf:
            stats = np.array(hf["error_stats"])   # (n_calls, 2)

        fig, ax = plt.subplots(figsize=figsize)
        ax.semilogy(stats[:, 0], label='mean |φ error|')
        ax.semilogy(stats[:, 1], label='max  |φ error|')
        ax.set_xlabel('validation call')
        ax.set_ylabel('|φ_hat − φ_true|')
        ax.set_title('Validation error')
        ax.legend()
        fig.tight_layout()
        return fig


# ------------------------------------------------------------------

def _to_np(x):
    """Accept numpy, torch tensor, or list."""
    if hasattr(x, 'detach'):
        return x.detach().cpu().numpy()
    return np.asarray(x)
