"""
Plot computational cost comparison from saved results.
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
from common import mpl_apply


def plot_timing(results_file='outputs/burgers_results.npz', save_fig=True):
    """Plot computational cost comparison."""

    mpl_apply()
    # Load results
    data = np.load(results_file)

    print(f"Loaded results: ν={data['nu']}, T={data['T']}")

    # Create figure
    fig, ax = plt.subplots(figsize=(8, 6))

    methods = []
    times = []
    colors = []

    if 'fd_time' in data:
        methods.append('Finite\nDifference')
        times.append(float(data['fd_time']))
        colors.append('steelblue')

    if 'pinn_time' in data:
        methods.append('PINN')
        times.append(float(data['pinn_time']))
        colors.append('coral')

    if methods:
        bars = ax.bar(methods, times, color=colors, alpha=0.8,
                      edgecolor='black', linewidth=1.5)

        # Add value labels on bars
        for bar, time in zip(bars, times):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                    f'{time:.1f}s',
                    ha='center', va='bottom', fontsize=14, fontweight='bold')

        ax.set_ylabel('Wall-clock Time (seconds)', fontsize=14)
        ax.set_title('Computational Cost Comparison',
                     fontsize=16, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')

        # Add speedup annotation if both methods present
        if len(times) == 2:
            speedup = times[1] / times[0]  # PINN time / FD time
            ax.text(0.5, max(times) * 0.95,
                    f'PINN is {speedup:.0f}× slower',
                    ha='center', fontsize=12, style='italic',
                    bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.5))

    plt.tight_layout()

    if save_fig:
        output_path = 'outputs/plot_timing.pdf'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved to: {output_path}")

    plt.show()

    return fig


if __name__ == "__main__":
    results_file = sys.argv[1] if len(
        sys.argv) > 1 else 'outputs/burgers_results.npz'
    plot_timing(results_file)
