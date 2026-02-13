"""
Plot solution comparison from saved results.

Creates the top panel showing FD vs PINN vs Reference at final time.
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
from common import mpl_apply


def plot_solution_comparison(results_file='outputs/burgers_results.npz', save_fig=True):
    """Plot solution comparison at final time."""

    mpl_apply()

    # Load results
    data = np.load(results_file)

    print(f"Loaded results: ν={data['nu']}, T={data['T']}")

    # Create figure
    fig, ax = plt.subplots(figsize=(12, 5))

    # Reference solution
    if 'ref_x' in data and 'ref_u' in data:
        x_ref = data['ref_x']
        u_ref = data['ref_u'][-1]  # Final time
        ax.plot(x_ref, u_ref, 'k-', linewidth=2.5,
                label=f'Reference (FD {len(x_ref)} pts)', alpha=0.7)

    # FD solution
    if 'fd_x' in data and 'fd_u' in data:
        x_fd = data['fd_x']
        u_fd = data['fd_u'][-1]  # Final time
        ax.plot(x_fd, u_fd, 'b-', linewidth=1.8,
                label=f'FD ({len(x_fd)} pts)',
                marker='o', markersize=4, markevery=10, alpha=0.8)

    # PINN solution
    if 'pinn_u' in data and 'ref_x' in data:
        x_pinn = data['ref_x']  # PINN was evaluated on reference grid
        u_pinn = data['pinn_u'][-1]  # Final time
        ax.plot(x_pinn, u_pinn, 'r--', linewidth=2,
                label='PINN', alpha=0.9)

    # Formatting
    ax.set_xlabel('x', fontsize=14)
    ax.set_ylabel('u(x, T)', fontsize=14)
    ax.set_title(f'Solution Comparison at t = {data["T"]}',
                 fontsize=16, fontweight='bold')
    ax.legend(fontsize=12, loc='best')
    ax.grid(True, alpha=0.3, linewidth=0.5)

    plt.tight_layout()

    if save_fig:
        output_path = 'outputs/plot_solution_comparison.pdf'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved to: {output_path}")

    plt.show()

    return fig


if __name__ == "__main__":
    results_file = sys.argv[1] if len(
        sys.argv) > 1 else 'outputs/burgers_results.npz'
    plot_solution_comparison(results_file)
