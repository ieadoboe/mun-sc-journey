import matplotlib.pyplot as plt
import numpy as np

# Set some fonts
plt.rc('text', usetex=True)
plt.rc('font', family='serif', size=15)

# Load data from file
rho          = np.load("vanka_sensitrivity_tau_1.05_dirichlet.npy")

# Details on collection
N            = 50
OMEGA_OUTER  = np.linspace(0.02, 1., N)
OMEGA_INNER  = np.linspace(0.02, 1., N)

# Hard-code best values
omega_1_opt  = 0.78
omega_0_opt  = 0.74

# Plot the heatmap
plt.imshow(rho, cmap='bwr',  vmin=np.min(rho), vmax=min(1, np.max(rho)),
           origin="lower",
           interpolation='bilinear',
           extent=[min(OMEGA_OUTER), max(OMEGA_OUTER), min(OMEGA_INNER), max(OMEGA_INNER)])

# Labels and color bar
ax = plt.gca()
ax.set_xlabel(r'$\omega_1$', fontsize=20)
ax.set_ylabel(r'$\omega_0$', fontsize=20)
plt.colorbar()

# Add marker for optimal value
plt.scatter(omega_1_opt, omega_0_opt, s=50, c='white', marker='x')

plt.show()
