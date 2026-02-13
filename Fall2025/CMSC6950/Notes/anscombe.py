import pandas as pd
import matplotlib.pyplot as plt


def quartet():
    x1 = [10, 8, 13, 9, 11, 14, 6, 4, 12, 7, 5]
    y1 = [8.04, 6.95, 7.58, 8.81, 8.33, 9.96, 7.24, 4.26, 10.84, 4.82, 5.68]

    x2 = [10, 8, 13, 9, 11, 14, 6, 4, 12, 7, 5]
    y2 = [9.14, 8.14, 8.74, 8.77, 9.26, 8.10, 6.13, 3.10, 9.13, 7.26, 4.74]

    x3 = [10, 8, 13, 9, 11, 14, 6, 4, 12, 7, 5]
    y3 = [7.46, 6.77, 12.74, 7.11, 7.81, 8.84, 6.08, 5.39, 8.15, 6.42, 5.73]

    x4 = [8, 8, 8, 8, 8, 8, 8, 19, 8, 8, 8]
    y4 = [6.58, 5.76, 7.71, 8.84, 8.47, 7.04, 5.25, 12.50, 5.56, 7.91, 6.89]

    df1 = pd.DataFrame(zip(x1, y1), columns=('x', 'y'))
    df2 = pd.DataFrame(zip(x2, y2), columns=('x', 'y'))
    df3 = pd.DataFrame(zip(x3, y3), columns=('x', 'y'))
    df4 = pd.DataFrame(zip(x4, y4), columns=('x', 'y'))

    return df1, df2, df3, df4


def plot_data():
    df1, df2, df3, df4 = quartet()

    fig, axes = plt.subplots(2, 2, sharex=True, sharey=True)

    axes[0, 0].plot(df1['x'], df1['y'], 'ro', label='df1')
    axes[0, 0].set_xlim(0, 20)
    axes[0, 0].set_ylim(0, 20)
    axes[0, 0].set_ylabel(r'$y$')
    axes[0, 0].legend()

    axes[0, 1].plot(df2['x'], df2['y'], 'bo', label='df2')
    axes[0, 1].set_xlim(0, 20)
    axes[0, 1].set_ylim(0, 20)
    axes[0, 1].legend()

    axes[1, 0].plot(df3['x'], df3['y'], 'go', label='df3')
    axes[1, 0].set_xlim(0, 20)
    axes[1, 0].set_ylim(0, 20)
    axes[1, 0].set_xlabel(r'$x$')
    axes[1, 0].set_ylabel(r'$y$')
    axes[1, 0].legend()

    axes[1, 1].plot(df4['x'], df4['y'], 'ko', label='df4')
    axes[1, 1].set_xlim(0, 20)
    axes[1, 1].set_ylim(0, 20)
    axes[1, 1].set_xlabel(r'$x$')
    axes[1, 1].legend()

    plt.show()


def oneplot_data():
    df1, df2, df3, df4 = quartet()

    plt.plot(df1['x'], df1['y'], 'ro', label='df1')
    plt.plot(df2['x'], df2['y'], 'bx', label='df2')
    plt.plot(df3['x'], df3['y'], 'gs', label='df3')
    plt.plot(df4['x'], df4['y'], 'k^', label='df4')

    plt.xlabel(r'$x$')
    plt.ylabel(r'$y$')
    plt.xlim(0, 20)
    plt.ylim(0, 20)
    plt.legend()

    plt.show()
