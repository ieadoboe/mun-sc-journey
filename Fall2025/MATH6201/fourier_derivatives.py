import numpy as np
import matplotlib.pyplot as plt

# Define the function u(x) = sin(2πx) for x ∈ [0, 1]
N = 256  # Number of points
x = np.linspace(0, 1, N, endpoint=False)  # Domain [0, 1)
u = np.sin(2 * np.pi * x)

# Compute the Fourier transform using numpy.fft
u_hat = np.fft.fft(u)

# Get the wave numbers (frequency components)
# Note: numpy uses opposite sign convention, so k values need adjustment
k = np.fft.fftfreq(N, d=1/N) * 2 * np.pi  # Wave numbers

# Compute derivatives in Fourier space
# Note: numpy's convention is opposite, so we use +1j instead of -1j
u_x_hat = 1j * k * u_hat  # First derivative in Fourier space
u_xx_hat = -k**2 * u_hat  # Second derivative in Fourier space

# Transform back to physical space
u_x = np.fft.ifft(u_x_hat).real
u_xx = np.fft.ifft(u_xx_hat).real

# Analytical derivatives for comparison
u_x_exact = 2 * np.pi * np.cos(2 * np.pi * x)
u_xx_exact = -(2 * np.pi)**2 * np.sin(2 * np.pi * x)

# Create plots
fig, axes = plt.subplots(3, 2, figsize=(14, 12))

# Plot 1: Original function u(x)
axes[0, 0].plot(x, u, 'b-', linewidth=2, label='u(x) = sin(2πx)')
axes[0, 0].set_xlabel('x', fontsize=12)
axes[0, 0].set_ylabel('u(x)', fontsize=12)
axes[0, 0].set_title('Original Function u(x)', fontsize=14, fontweight='bold')
axes[0, 0].grid(True, alpha=0.3)
axes[0, 0].legend(fontsize=10)

# Plot 2: Fourier coefficients (magnitude)
axes[0, 1].stem(np.fft.fftshift(k), np.fft.fftshift(np.abs(u_hat)), basefmt=' ')
axes[0, 1].set_xlabel('Wave number k', fontsize=12)
axes[0, 1].set_ylabel('|û_k|', fontsize=12)
axes[0, 1].set_title('Magnitude of Fourier Coefficients', fontsize=14, fontweight='bold')
axes[0, 1].grid(True, alpha=0.3)
axes[0, 1].set_xlim([-20, 20])

# Plot 3: First derivative u_x (FFT vs Exact)
axes[1, 0].plot(x, u_x, 'r-', linewidth=2, label='u_x (FFT)', alpha=0.7)
axes[1, 0].plot(x, u_x_exact, 'k--', linewidth=2, label='u_x (Exact)', alpha=0.7)
axes[1, 0].set_xlabel('x', fontsize=12)
axes[1, 0].set_ylabel('du/dx', fontsize=12)
axes[1, 0].set_title('First Derivative: du/dx', fontsize=14, fontweight='bold')
axes[1, 0].grid(True, alpha=0.3)
axes[1, 0].legend(fontsize=10)

# Plot 4: Error in first derivative
error_ux = np.abs(u_x - u_x_exact)
axes[1, 1].semilogy(x, error_ux, 'g-', linewidth=2)
axes[1, 1].set_xlabel('x', fontsize=12)
axes[1, 1].set_ylabel('Absolute Error', fontsize=12)
axes[1, 1].set_title('Error in First Derivative', fontsize=14, fontweight='bold')
axes[1, 1].grid(True, alpha=0.3)

# Plot 5: Second derivative u_xx (FFT vs Exact)
axes[2, 0].plot(x, u_xx, 'b-', linewidth=2, label='u_xx (FFT)', alpha=0.7)
axes[2, 0].plot(x, u_xx_exact, 'k--', linewidth=2, label='u_xx (Exact)', alpha=0.7)
axes[2, 0].set_xlabel('x', fontsize=12)
axes[2, 0].set_ylabel('d²u/dx²', fontsize=12)
axes[2, 0].set_title('Second Derivative: d²u/dx²', fontsize=14, fontweight='bold')
axes[2, 0].grid(True, alpha=0.3)
axes[2, 0].legend(fontsize=10)

# Plot 6: Error in second derivative
error_uxx = np.abs(u_xx - u_xx_exact)
axes[2, 1].semilogy(x, error_uxx, 'm-', linewidth=2)
axes[2, 1].set_xlabel('x', fontsize=12)
axes[2, 1].set_ylabel('Absolute Error', fontsize=12)
axes[2, 1].set_title('Error in Second Derivative', fontsize=14, fontweight='bold')
axes[2, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('/mnt/user-data/outputs/fourier_derivatives_plots.png', dpi=300, bbox_inches='tight')
print("Plots saved to fourier_derivatives_plots.png")

# Print numerical results
print("\n" + "="*70)
print("NUMERICAL RESULTS")
print("="*70)
print(f"\nNumber of grid points: {N}")
print(f"Domain: x ∈ [0, 1)")
print(f"\nMaximum errors:")
print(f"  First derivative (u_x):  {np.max(error_ux):.2e}")
print(f"  Second derivative (u_xx): {np.max(error_uxx):.2e}")
print("\n" + "="*70)

# Display sample values
print("\nSample values at x = 0.25:")
idx = N // 4
print(f"  x = {x[idx]:.4f}")
print(f"  u(x) = {u[idx]:.6f}")
print(f"  u_x (FFT) = {u_x[idx]:.6f}, u_x (Exact) = {u_x_exact[idx]:.6f}")
print(f"  u_xx (FFT) = {u_xx[idx]:.6f}, u_xx (Exact) = {u_xx_exact[idx]:.6f}")
print("="*70)

plt.show()
