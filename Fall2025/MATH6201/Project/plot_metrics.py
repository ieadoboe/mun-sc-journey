"""
Plot training metrics and error comparison from saved results.

Creates plots showing PINN training loss and error bars.
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
from common import mpl_apply


def plot_metrics(results_file='outputs/burgers_results.npz', save_fig=True):
    """Plot training loss and error comparison."""

    mpl_apply()

    # Load results
    data = np.load(results_file)

    print(f"Loaded results: ν={data['nu']}, T={data['T']}")

    # Create figure with 2 subplots
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # === Left: PINN Training Loss ===
    if 'pinn_loss_history' in data:
        loss = data['pinn_loss_history']
        epochs = np.arange(len(loss))

        axes[0].semilogy(epochs, loss, 'b-', linewidth=2, label='Total Loss')

        # Add component losses if available
        if 'pinn_loss_pde' in data:
            axes[0].semilogy(epochs, data['pinn_loss_pde'], 'r--',
                             alpha=0.7, linewidth=1.5, label='PDE Loss')
        if 'pinn_loss_ic' in data:
            axes[0].semilogy(epochs, data['pinn_loss_ic'], 'g--',
                             alpha=0.7, linewidth=1.5, label='IC Loss')
        if 'pinn_loss_bc' in data:
            axes[0].semilogy(epochs, data['pinn_loss_bc'], 'm--',
                             alpha=0.7, linewidth=1.5, label='BC Loss')

        axes[0].set_xlabel('Epoch', fontsize=13)
        axes[0].set_ylabel('Loss', fontsize=13)
        axes[0].set_title('PINN Training History',
                          fontsize=14, fontweight='bold')
        axes[0].legend(fontsize=11)
        axes[0].grid(True, alpha=0.3, which='both')

        # Add final loss annotation
        final_loss = loss[-1]
        axes[0].annotate(f'Final: {final_loss:.2e}',
                         xy=(len(loss)-1, final_loss),
                         xytext=(-60, 20), textcoords='offset points',
                         fontsize=10, fontweight='bold',
                         bbox=dict(boxstyle='round',
                                   facecolor='wheat', alpha=0.8),
                         arrowprops=dict(arrowstyle='->', lw=1.5))

    # === Right: Error Comparison ===
    methods = []
    l2_errors = []
    linf_errors = []

    if 'fd_l2_error' in data:
        methods.append('FD')
        l2_errors.append(data['fd_l2_error'])
        linf_errors.append(data['fd_linf_error'])

    if 'pinn_l2_error' in data:
        methods.append('PINN')
        l2_errors.append(data['pinn_l2_error'])
        linf_errors.append(data['pinn_linf_error'])

    if methods:
        x_pos = np.arange(len(methods))
        width = 0.35

        bars1 = axes[1].bar(x_pos - width/2, l2_errors, width,
                            label=r'$L^2$ error', alpha=0.8, color='steelblue')
        bars2 = axes[1].bar(x_pos + width/2, linf_errors, width,
                            label=r'$L^\infty$ error', alpha=0.8, color='coral')

        axes[1].set_ylabel('Error', fontsize=13)
        axes[1].set_title('Error Comparison (vs Reference)',
                          fontsize=14, fontweight='bold')
        axes[1].set_xticks(x_pos)
        axes[1].set_xticklabels(methods, fontsize=12)
        axes[1].legend(fontsize=11)
        axes[1].set_yscale('log')
        axes[1].grid(True, alpha=0.3, axis='y', which='both')

        # Add value labels on bars
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                axes[1].text(bar.get_x() + bar.get_width()/2., height,
                             f'{height:.1e}',
                             ha='center', va='bottom', fontsize=9)

    plt.tight_layout()

    if save_fig:
        output_path = 'outputs/plot_metrics.pdf'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved to: {output_path}")

    plt.show()

    return fig


if __name__ == "__main__":
    results_file = sys.argv[1] if len(
        sys.argv) > 1 else 'outputs/burgers_results.npz'
    plot_metrics(results_file)
