import matplotlib.pyplot as plt
import matplotlib as mpl
import pickle

# Load data
f = open('pcg.pkl', 'rb')
data = pickle.load(f)
lh = data['loss_history']
rh = data['res_history']

# Set some defaults
params = {'figure.figsize': [6, 3]}
mpl.rcParams.update(params)

# Plot training loss
fig, ax = plt.subplots()
ax.plot(lh, '-', color='k', label='Loss', markersize=1, lw=1)
ax.set_xlabel('Training epoch')
ax.set_ylabel('Loss')
ax.grid(True)

# Now on separate y-axis, plot residual
ax2 = ax.twinx()
kwargs = dict(marker='o', color='tab:red', mec='w', ms=5, lw=1.5,
              mew=0.75, linestyle='-')
ax2.semilogy(rh, label='Relative residual', **kwargs, markevery=50)
ax2.set_ylabel('Relative residual')

# Legend and show plot
fig.legend()
plt.show()
