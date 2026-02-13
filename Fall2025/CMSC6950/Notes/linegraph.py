import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

# Data from elsewhere
N = np.array([1024, 4096, 16384, 65536])
S_CPU = np.array([0.851, 2.832, 11.137, 33.840])
S_GPU = np.array([1.204, 1.390, 2.079, 4.375])
D_CPU = np.array([8.977, 510.795, 32585.736, np.nan])
D_GPU = np.array([1.013, 3.801, 177.179, np.nan])

# Set some defaults
params = {'figure.figsize': [6, 3]}
mpl.rcParams.update(params)

nolbl = '_nolegend_'

# Single figure
plt.figure()
ax = plt.gca()

# Common plot elements for Sparse data
kwargs = {'marker': 'v',
          'color': 'tab:green',
          'markersize': 8,
          'markerfacecolor': 'w',
          'markeredgewidth': 2.0,
          'lw': 2}

ax.loglog(N, S_CPU, label='Sparse CPU', **kwargs)
ax.loglog(N, S_GPU, label='Sparse GPU', **kwargs, linestyle='dotted')

# Common plot elements for Dense data (just changing markers)
kwargs = {'marker': 's',
          'color': 'tab:blue',
          'markersize': 8,
          'markerfacecolor': 'w',
          'markeredgewidth': 2.0,
          'lw': 2}

ax.loglog(N, D_CPU, label='Dense CPU', **kwargs)
ax.loglog(N, D_GPU, label='Dense GPU', **kwargs, linestyle='dotted')

# Scaling lines
l, = ax.loglog(N[1:], N[1:] * 2e-3, '--', color='k', label=nolbl, lw=1)
ax.text(N[-1], N[-1] * 4e-3, '$O\\left(N\\right)$', ha='right')
l, = ax.loglog(N[1:], (N[1:] * 3e-3)**3., '--', color='k', label=nolbl, lw=1)
ax.text(N[-1], N[-1] * 4e0, '$O\\left(N^3\\right)$', ha='right')

# Labels and ticks
ax.set_xticks([1e3, 1e4, 1e5])
ax.set_ylabel('Time (s)')
ax.set_xlabel('N')
ax.grid()

plt.legend()
plt.show()
