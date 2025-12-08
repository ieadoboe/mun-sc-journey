"""
Plot spatiotemporal evolution from saved results.

Creates heatmaps showing u(x,t) for FD, PINN, and error.
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
from common import mpl_apply


def plot_spatiotemporal(results_file='outputs/burgers_results.npz', save_fig=True):
    """Plot spatiotemporal evolution heatmaps."""

    mpl_apply()

    # Load results
    data = np.load(results_file)

    print(f"Loaded results: ν={data['nu']}, T={data['T']}")

    # Create figure with 3 subplots
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # FD heatmap
    if 'fd_x' in data and 'fd_t' in data and 'fd_u' in data:
        x_fd = data['fd_x']
        t_fd = data['fd_t']
        u_fd = data['fd_u']

        X_fd, T_fd = np.meshgrid(x_fd, t_fd)
        c1 = axes[0].contourf(X_fd, T_fd, u_fd, levels=50, cmap='RdBu_r')
        axes[0].set_xlabel('x', fontsize=12)
        axes[0].set_ylabel('t', fontsize=12)
        axes[0].set_title('Finite Difference: u(x,t)',
                          fontsize=13, fontweight='bold')
        plt.colorbar(c1, ax=axes[0], label='u')

    # PINN heatmap
    if 'pinn_u' in data and 'ref_x' in data and 'ref_t' in data:
        x_pinn = data['ref_x']
        t_pinn = data['ref_t']
        u_pinn = data['pinn_u']

        X_pinn, T_pinn = np.meshgrid(x_pinn, t_pinn)
        c2 = axes[1].contourf(X_pinn, T_pinn, u_pinn, levels=50, cmap='RdBu_r')
        axes[1].set_xlabel('x', fontsize=12)
        axes[1].set_ylabel('t', fontsize=12)
        axes[1].set_title('PINN: u(x,t)', fontsize=13, fontweight='bold')
        plt.colorbar(c2, ax=axes[1], label='u')

    # Error heatmap
    if 'pinn_u' in data and 'ref_u' in data:
        u_ref = data['ref_u']
        u_pinn = data['pinn_u']
        error = np.abs(u_pinn - u_ref)

        c3 = axes[2].contourf(X_pinn, T_pinn, error, levels=50, cmap='hot')
        axes[2].set_xlabel('x', fontsize=12)
        axes[2].set_ylabel('t', fontsize=12)
        axes[2].set_title('PINN Absolute Error',
                          fontsize=13, fontweight='bold')
        plt.colorbar(c3, ax=axes[2], label='|u_PINN - u_ref|')

    plt.suptitle('Spatiotemporal Evolution',
                 fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()

    if save_fig:
        output_path = 'outputs/plot_spatiotemporal.pdf'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved to: {output_path}")

    plt.show()

    return fig


if __name__ == "__main__":
    results_file = sys.argv[1] if len(
        sys.argv) > 1 else 'outputs/burgers_results.npz'
    plot_spatiotemporal(results_file)
