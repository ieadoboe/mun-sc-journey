# Import matplotlib and set figure aspect ratio
import matplotlib as mpl
import matplotlib.pyplot as plt


# Define data
ks = [1, 2, 3, 4, 5]
val1 = [5, 3, 0, 0, 0]
val2 = [287, 74, 7, 1, 0]

params = {'figure.figsize': [6, 2]}
mpl.rcParams.update(params)

# Set up subplots
fig, ax = plt.subplots(nrows=1, ncols=2, sharey=True)

# Bar graphs and add text to label points
ax[0].bar(ks, val1, align='center', color='tab:blue')
for num, count in zip(ks, val1):
    ax[0].text(num, count+1.1, f'{count}',
               color='tab:gray', va='bottom', ha='center', fontsize=7)


ax[1].bar(ks, val2, align='center', color='tab:red')
for num, count in zip(ks, val2):
    ax[1].text(num, count+1.1, f'{count}',
               color='tab:gray', va='bottom', ha='center', fontsize=7)

# Titles on subplots
ax[0].set_title('With tiebreaking')
ax[1].set_title('Without tiebreaking')

# Axis labels
ax[0].set_ylabel('Count')
ax[0].set_xlabel(r'\# of zero-diameter aggregates')
ax[1].set_xlabel(r'\# of zero-diameter aggregates')

# Set y axis scaling and tick locations
ax[1].yaxis.tick_right()
ax[0].set_yscale('log')
ax[1].set_yscale('log')

# Set axis limits
ax[0].set_xlim(0.25, 5.5)
ax[1].set_xlim(0.25, 5.5)
ax[0].set_ylim(0.7, 2500)
ax[1].set_ylim(0.7, 2500)

# axis ticks
ax[0].set_xticks([1, 2, 3, 4, 5])
ax[1].set_xticks([1, 2, 3, 4, 5])
ax[0].set_yticks([1, 10, 100, 1000])
ax[1].set_yticks([1, 10, 100, 1000])

# Show the plot
plt.show()
