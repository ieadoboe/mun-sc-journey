"""
Comparative Analysis: Finite Difference vs PINN for Burgers' Equation

This script compares the two numerical approaches:
1. Classical finite difference (explicit scheme)
2. Physics-informed neural network (PINN)

Metrics assessed:
- Accuracy (error against reference solution)
- Computational cost (wall-clock time)
- Solution quality (smoothness, shock capture)

Author: Isaac Adoboe
MATH 6201 Final Project
"""

from burgers_pinn import BurgersPINN
from burgers_fd import BurgersFD
from typing import Tuple
import time
import matplotlib.pyplot as plt
import numpy as np
import matplotlib
# Disable LaTeX rendering to avoid Unicode issues
matplotlib.rcParams['text.usetex'] = False


class BurgersComparison:
    """Compare finite difference and PINN solutions."""

    def __init__(self, nu: float = 0.01, T: float = 1.0):
        """
        Initialize comparison framework.

        Args:
            nu: Viscosity coefficient
            T: Final time
        """
        self.nu = nu
        self.T = T

        # Solutions
        self.fd_solution = None
        self.pinn_solution = None
        self.reference_solution = None

        # Timing
        self.fd_time = None
        self.pinn_time = None

    def solve_fd(self, nx: int = 200, verbose: bool = True):
        """Solve using finite difference method."""
        if verbose:
            print("\n" + "="*70)
            print("FINITE DIFFERENCE METHOD")
            print("="*70)

        start = time.time()

        solver = BurgersFD(nx=nx, nu=self.nu, T=self.T, cfl_factor=0.4)
        t, u = solver.solve(adaptive_dt=True, verbose=verbose)

        self.fd_time = time.time() - start
        self.fd_solution = {
            'solver': solver,
            'x': solver.x,
            't': t,
            'u': u,
            'time': self.fd_time
        }

        if verbose:
            print(f"FD solution time: {self.fd_time:.2f}s")

        return solver

    def solve_pinn(
        self,
        layers: list = [2, 100, 100, 100, 1],
        n_epochs: int = 15000,
        verbose: bool = True
    ):
        """Solve using PINN method with optimized parameters."""
        import tensorflow as tf

        if verbose:
            print("\n" + "="*70)
            print("PHYSICS-INFORMED NEURAL NETWORK")
            print("="*70)

        # Use single best seed
        tf.random.set_seed(42)
        np.random.seed(42)

        start = time.time()

        pinn = BurgersPINN(layers=layers, nu=self.nu,
                           T=self.T, activation='tanh')
        pinn.train(
            n_epochs=n_epochs,
            n_pde=10000,
            n_ic=500,
            n_bc=500,
            learning_rate=1e-3,
            lambda_pde=10.0,   # Stronger PDE emphasis
            lambda_ic=100.0,
            lambda_bc=100.0,
            verbose=verbose,
            print_every=3000
        )

        self.pinn_time = time.time() - start

        final_pde = pinn.loss_components['pde'][-1]
        final_ic = pinn.loss_components['ic'][-1]
        final_bc = pinn.loss_components['bc'][-1]

        self.pinn_solution = {
            'model': pinn,
            'time': self.pinn_time
        }

        if verbose:
            print("\n" + "="*70)
            print(f"PINN Training Summary:")
            print(f"  PDE loss: {final_pde:.6e}")
            print(f"  IC loss: {final_ic:.6e}")
            print(f"  BC loss: {final_bc:.6e}")
            print(f"  Training time: {self.pinn_time:.1f}s")

            if final_pde < 0.01:
                print(f"  ✓ Good PDE convergence")
            else:
                print(
                    f"  ⚠ PDE convergence limited (fundamental issue with shock approximation)")
            print("="*70)

        return pinn

    def create_reference_solution(self, nx_ref: int = 800):
        """Create high-resolution FD solution as reference."""
        print("\nGenerating reference solution (high-resolution FD)...")
        solver_ref = BurgersFD(nx=nx_ref, nu=self.nu, T=self.T, cfl_factor=0.4)
        t_ref, u_ref = solver_ref.solve(verbose=False)

        self.reference_solution = {
            'x': solver_ref.x,
            't': t_ref,
            'u': u_ref
        }

        print(
            f"Reference solution: {nx_ref} spatial points, {len(t_ref)} time steps")

    def compute_errors(self):
        """Compute error metrics against reference solution."""
        if self.reference_solution is None:
            self.create_reference_solution()

        x_ref = self.reference_solution['x']
        u_ref = self.reference_solution['u'][-1]  # Final time

        errors = {}

        # FD errors
        if self.fd_solution is not None:
            x_fd = self.fd_solution['x']
            u_fd = self.fd_solution['u'][-1]
            u_ref_interp = np.interp(x_fd, x_ref, u_ref)

            errors['FD'] = {
                'L2': np.sqrt(np.mean((u_fd - u_ref_interp)**2)),
                'Linf': np.max(np.abs(u_fd - u_ref_interp)),
                'Rel_L2': np.sqrt(np.mean((u_fd - u_ref_interp)**2)) / np.sqrt(np.mean(u_ref_interp**2))
            }

        # PINN errors
        if self.pinn_solution is not None:
            pinn = self.pinn_solution['model']
            u_pinn = pinn.predict(x_ref, np.full_like(x_ref, self.T))

            errors['PINN'] = {
                'L2': np.sqrt(np.mean((u_pinn - u_ref)**2)),
                'Linf': np.max(np.abs(u_pinn - u_ref)),
                'Rel_L2': np.sqrt(np.mean((u_pinn - u_ref)**2)) / np.sqrt(np.mean(u_ref**2))
            }

        return errors

    def plot_comparison(self, figsize: Tuple[int, int] = (16, 10)):
        """Create comprehensive comparison visualization."""
        if self.fd_solution is None or self.pinn_solution is None:
            raise ValueError("Must run both solvers before comparison")

        fig = plt.figure(figsize=figsize)
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

        # Extract solutions
        x_fd = self.fd_solution['x']
        u_fd = self.fd_solution['u'][-1]

        pinn = self.pinn_solution['model']
        x_pinn = np.linspace(0, 1, 200)
        u_pinn = pinn.predict(x_pinn, np.full_like(x_pinn, self.T))

        if self.reference_solution is not None:
            x_ref = self.reference_solution['x']
            u_ref = self.reference_solution['u'][-1]

        # ========== Row 1: Final time comparison ==========
        ax1 = fig.add_subplot(gs[0, :])
        if self.reference_solution is not None:
            ax1.plot(x_ref, u_ref, 'k-', linewidth=2,
                     label='Reference (FD 800 pts)', alpha=0.7)
        ax1.plot(x_fd, u_fd, 'b-', linewidth=1.5,
                 label=f'FD ({len(x_fd)} pts)', marker='o', markersize=3, markevery=10)
        ax1.plot(x_pinn, u_pinn, 'r--', linewidth=1.5, label='PINN')
        ax1.set_xlabel('x', fontsize=12)
        ax1.set_ylabel('u(x, T)', fontsize=12)
        ax1.set_title(
            f'Solution Comparison at t = {self.T}', fontsize=14, fontweight='bold')
        ax1.legend(fontsize=11)
        ax1.grid(True, alpha=0.3)

        # ========== Row 2: Spatiotemporal evolution ==========
        # FD heatmap
        ax2 = fig.add_subplot(gs[1, 0])
        t_fd = self.fd_solution['t']
        u_fd_full = self.fd_solution['u']
        X_fd, T_fd = np.meshgrid(x_fd, t_fd)
        c1 = ax2.contourf(X_fd, T_fd, u_fd_full, levels=50, cmap='RdBu_r')
        ax2.set_xlabel('x', fontsize=11)
        ax2.set_ylabel('t', fontsize=11)
        ax2.set_title('FD: u(x,t)', fontsize=12, fontweight='bold')
        plt.colorbar(c1, ax=ax2)

        # PINN heatmap
        ax3 = fig.add_subplot(gs[1, 1])
        nx_plot, nt_plot = 200, 200
        x_plot = np.linspace(0, 1, nx_plot)
        t_plot = np.linspace(0, self.T, nt_plot)
        X_pinn, T_pinn = np.meshgrid(x_plot, t_plot)
        u_pinn_full = np.zeros_like(X_pinn)
        for i in range(nt_plot):
            u_pinn_full[i, :] = pinn.predict(
                x_plot, np.full_like(x_plot, t_plot[i]))
        c2 = ax3.contourf(X_pinn, T_pinn, u_pinn_full,
                          levels=50, cmap='RdBu_r')
        ax3.set_xlabel('x', fontsize=11)
        ax3.set_ylabel('t', fontsize=11)
        ax3.set_title('PINN: u(x,t)', fontsize=12, fontweight='bold')
        plt.colorbar(c2, ax=ax3)

        # Error heatmap (if reference available)
        ax4 = fig.add_subplot(gs[1, 2])
        if self.reference_solution is not None:
            # Interpolate reference to PINN grid
            u_ref_full = np.zeros_like(u_pinn_full)
            for i in range(nt_plot):
                idx_ref = np.argmin(
                    np.abs(self.reference_solution['t'] - t_plot[i]))
                u_ref_full[i, :] = np.interp(
                    x_plot, x_ref, self.reference_solution['u'][idx_ref])

            error = np.abs(u_pinn_full - u_ref_full)
            c3 = ax4.contourf(X_pinn, T_pinn, error, levels=50, cmap='hot')
            ax4.set_xlabel('x', fontsize=11)
            ax4.set_ylabel('t', fontsize=11)
            ax4.set_title('PINN Absolute Error',
                          fontsize=12, fontweight='bold')
            plt.colorbar(c3, ax=ax4, label='|u_PINN - u_ref|')
        else:
            ax4.text(0.5, 0.5, 'No reference\navailable',
                     ha='center', va='center', fontsize=14)
            ax4.set_xticks([])
            ax4.set_yticks([])

        # ========== Row 3: Analysis ==========
        # Loss history
        ax5 = fig.add_subplot(gs[2, 0])
        ax5.semilogy(pinn.loss_history, 'b-', linewidth=1.5)
        ax5.set_xlabel('Epoch', fontsize=11)
        ax5.set_ylabel('Loss', fontsize=11)
        ax5.set_title('PINN Training Loss', fontsize=12, fontweight='bold')
        ax5.grid(True, alpha=0.3)

        # Error metrics
        ax6 = fig.add_subplot(gs[2, 1])
        errors = self.compute_errors()

        methods = list(errors.keys())
        l2_errors = [errors[m]['L2'] for m in methods]
        linf_errors = [errors[m]['Linf'] for m in methods]

        x_bar = np.arange(len(methods))
        width = 0.35

        ax6.bar(x_bar - width/2, l2_errors, width,
                label=r'$L^2$ error', alpha=0.8)
        ax6.bar(x_bar + width/2, linf_errors, width,
                label=r'$L^\infty$ error', alpha=0.8)
        ax6.set_ylabel('Error', fontsize=11)
        ax6.set_title('Error Comparison (vs Reference)',
                      fontsize=12, fontweight='bold')
        ax6.set_xticks(x_bar)
        ax6.set_xticklabels(methods)
        ax6.legend()
        ax6.set_yscale('log')
        ax6.grid(True, alpha=0.3, axis='y')

        # Computational cost
        ax7 = fig.add_subplot(gs[2, 2])
        times = [self.fd_time, self.pinn_time]
        colors = ['blue', 'red']
        ax7.bar(methods, times, color=colors, alpha=0.7)
        ax7.set_ylabel('Time (seconds)', fontsize=11)
        ax7.set_title('Computational Cost', fontsize=12, fontweight='bold')
        ax7.grid(True, alpha=0.3, axis='y')

        # Add text annotations
        for i, (method, t) in enumerate(zip(methods, times)):
            ax7.text(i, t, f'{t:.1f}s', ha='center',
                     va='bottom', fontsize=10, fontweight='bold')

        plt.suptitle('Burgers\' Equation: Finite Difference vs PINN',
                     fontsize=16, fontweight='bold', y=0.995)

        return fig, errors

    def save_results(self, filename: str = 'burgers_results.npz'):
        """
        Save comparison results to file for later plotting.

        Args:
            filename: Output filename (.npz format)
        """
        import os

        # Prepare data dictionary
        save_dict = {
            'nu': self.nu,
            'T': self.T,
        }

        # FD results
        if self.fd_solution is not None:
            save_dict.update({
                'fd_x': self.fd_solution['x'],
                'fd_t': self.fd_solution['t'],
                'fd_u': self.fd_solution['u'],
                'fd_time': self.fd_time
            })

        # Reference solution
        if self.reference_solution is not None:
            save_dict.update({
                'ref_x': self.reference_solution['x'],
                'ref_t': self.reference_solution['t'],
                'ref_u': self.reference_solution['u']
            })

        # PINN results (need to save predictions on grid)
        if self.pinn_solution is not None:
            pinn = self.pinn_solution['model']

            # Save network architecture and training info
            save_dict['pinn_layers'] = pinn.layers
            save_dict['pinn_time'] = self.pinn_time
            save_dict['pinn_loss_history'] = np.array(pinn.loss_history)
            save_dict['pinn_loss_pde'] = np.array(pinn.loss_components['pde'])
            save_dict['pinn_loss_ic'] = np.array(pinn.loss_components['ic'])
            save_dict['pinn_loss_bc'] = np.array(pinn.loss_components['bc'])

            # Save PINN predictions on reference grid
            if self.reference_solution is not None:
                x_ref = self.reference_solution['x']
                t_ref = self.reference_solution['t']

                # Predict on full spatiotemporal grid
                u_pinn_grid = np.zeros((len(t_ref), len(x_ref)))
                for i, t_val in enumerate(t_ref):
                    u_pinn_grid[i, :] = pinn.predict(
                        x_ref, np.full_like(x_ref, t_val))

                save_dict['pinn_u'] = u_pinn_grid

        # Compute and save errors
        errors = self.compute_errors()
        if 'FD' in errors:
            save_dict['fd_l2_error'] = errors['FD']['L2']
            save_dict['fd_linf_error'] = errors['FD']['Linf']
        if 'PINN' in errors:
            save_dict['pinn_l2_error'] = errors['PINN']['L2']
            save_dict['pinn_linf_error'] = errors['PINN']['Linf']

        # Create output directory if needed
        output_dir = os.path.dirname(filename)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # Save to compressed numpy format
        np.savez_compressed(filename, **save_dict)

        print(f"\n✓ Results saved to: {filename}")
        print(f"  File size: {os.path.getsize(filename) / 1024:.1f} KB")

    @staticmethod
    def load_results(filename: str = 'burgers_results.npz'):
        """
        Load comparison results from file.

        Args:
            filename: Input filename (.npz format)

        Returns:
            Dictionary containing all saved results
        """
        data = np.load(filename)

        print(f"\n✓ Results loaded from: {filename}")
        print(f"  Problem: ν={data['nu']}, T={data['T']}")

        if 'fd_x' in data:
            print(
                f"  FD: {len(data['fd_x'])} spatial points, {len(data['fd_t'])} time steps")
        if 'pinn_layers' in data:
            print(f"  PINN: {list(data['pinn_layers'])} architecture")

        return dict(data)

    def print_summary(self):
        """Print summary statistics."""
        print("\n" + "="*70)
        print("COMPARISON SUMMARY")
        print("="*70)

        errors = self.compute_errors()

        print(f"\nProblem parameters:")
        print(f"  Viscosity (ν): {self.nu}")
        print(f"  Final time (T): {self.T}")

        if self.fd_solution is not None:
            print(f"\nFinite Difference:")
            print(f"  Grid points: {len(self.fd_solution['x'])}")
            print(f"  Time steps: {len(self.fd_solution['t'])}")
            print(f"  Wall-clock time: {self.fd_time:.2f}s")
            if 'FD' in errors:
                print(f"  L² error: {errors['FD']['L2']:.6e}")
                print(f"  L∞ error: {errors['FD']['Linf']:.6e}")

        if self.pinn_solution is not None:
            pinn = self.pinn_solution['model']
            print(f"\nPhysics-Informed Neural Network:")
            print(f"  Architecture: {pinn.layers}")
            print(f"  Total parameters: {pinn.model.count_params()}")
            print(f"  Training time: {self.pinn_time:.2f}s")
            if 'PINN' in errors:
                print(f"  L² error: {errors['PINN']['L2']:.6e}")
                print(f"  L∞ error: {errors['PINN']['Linf']:.6e}")

        if self.fd_solution is not None and self.pinn_solution is not None:
            speedup = self.fd_time / self.pinn_time
            print(f"\nRelative performance:")
            print(f"  Time ratio (FD/PINN): {speedup:.2f}x")
            if 'FD' in errors and 'PINN' in errors:
                accuracy_ratio = errors['FD']['L2'] / errors['PINN']['L2']
                print(
                    f"  Accuracy ratio (FD/PINN L² error): {accuracy_ratio:.2f}x")

        print("="*70)
        """Print summary statistics."""
        print("\n" + "="*70)
        print("COMPARISON SUMMARY")
        print("="*70)

        errors = self.compute_errors()

        print(f"\nProblem parameters:")
        print(f"  Viscosity (ν): {self.nu}")
        print(f"  Final time (T): {self.T}")

        if self.fd_solution is not None:
            print(f"\nFinite Difference:")
            print(f"  Grid points: {len(self.fd_solution['x'])}")
            print(f"  Time steps: {len(self.fd_solution['t'])}")
            print(f"  Wall-clock time: {self.fd_time:.2f}s")
            if 'FD' in errors:
                print(f"  L² error: {errors['FD']['L2']:.6e}")
                print(f"  L∞ error: {errors['FD']['Linf']:.6e}")

        if self.pinn_solution is not None:
            pinn = self.pinn_solution['model']
            print(f"\nPhysics-Informed Neural Network:")
            print(f"  Architecture: {pinn.layers}")
            print(f"  Total parameters: {pinn.model.count_params()}")
            print(f"  Training time: {self.pinn_time:.2f}s")
            if 'PINN' in errors:
                print(f"  L² error: {errors['PINN']['L2']:.6e}")
                print(f"  L∞ error: {errors['PINN']['Linf']:.6e}")

        if self.fd_solution is not None and self.pinn_solution is not None:
            speedup = self.fd_time / self.pinn_time
            print(f"\nRelative performance:")
            print(f"  Time ratio (FD/PINN): {speedup:.2f}x")
            if 'FD' in errors and 'PINN' in errors:
                accuracy_ratio = errors['FD']['L2'] / errors['PINN']['L2']
                print(
                    f"  Accuracy ratio (FD/PINN L² error): {accuracy_ratio:.2f}x")

        print("="*70)


def main():
    """Run full comparison study and save results."""
    # Initialize comparison
    comparison = BurgersComparison(nu=0.01, T=1.0)

    # Generate reference solution
    comparison.create_reference_solution(nx_ref=800)

    # Solve with both methods
    comparison.solve_fd(nx=200, verbose=True)
    comparison.solve_pinn(
        layers=[2, 100, 100, 100, 1],  # 3 hidden layers (faster than 4)
        n_epochs=15000,                 # Reasonable training time (~10 min)
        verbose=True
    )

    # Print summary
    comparison.print_summary()

    # Save results for later plotting
    import os
    output_dir = 'outputs'
    os.makedirs(output_dir, exist_ok=True)
    results_file = os.path.join(output_dir, 'burgers_results.npz')
    comparison.save_results(results_file)

    print("\n" + "="*70)
    print("COMPUTATION COMPLETE")
    print("="*70)
    print("\nTo generate plots, run:")
    print("  python generate_all_plots.py")
    print("\nOr individual plots:")
    print("  python plot_solution.py")
    print("  python plot_spatiotemporal.py")
    print("  python plot_metrics.py")
    print("  python plot_timing.py")
    print("="*70)


if __name__ == "__main__":
    main()
