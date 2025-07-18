import numpy as np
import matplotlib.pyplot as plt

def G_function(v, x, r, K_m, sigma_k, s, u_bar, sigma_s, d, epsilon):
    K_v = K_m * np.exp(-v**2 / sigma_k**2)
    
    G = r * (1 - x / K_v) - s * np.exp(-((v - u_bar)**2) / sigma_s**2) - d * epsilon
    return G



def match_ylim_by_row(ax_array, clip_bottom=None, round_up=None):
    """
    Match y-axis limits across each row of subplots.

    Parameters:
    - ax_array: ndarray of matplotlib Axes (e.g., from plt.subplots)
    - clip_bottom: if set, clip ymin to this value (e.g. clip_bottom=0)
    - round_up: if set, round ymax up to nearest multiple of this (e.g. round_up=10)
    """
    nrows = ax_array.shape[0]

    for i in range(nrows):
        row_axes = ax_array[i, :]
        y_lims = np.array([ax.get_ylim() for ax in row_axes])

        ymin = y_lims[:, 0].min()
        ymax = y_lims[:, 1].max()

        if clip_bottom is not None:
            ymin = max(ymin, clip_bottom)
        if round_up is not None:
            ymax = np.ceil(ymax / round_up) * round_up

        for ax in row_axes:
            ax.set_ylim(ymin, ymax)

def repeat_legend(ax, loc='lower right', fontsize=10):
    """
    Repeat legend for each subplot in a row.
    
    Parameters:
    - ax: matplotlib Axes object
    - label: label for the legend
    - loc: location of the legend
    - fontsize: font size of the legend text
    """

    for axis in ax.flat:
        axis.grid(True)
        axis.tick_params(labelbottom=True, labelleft=True)

        # Ensure unique legend on every subplot bottom right
        handles, labels = axis.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        axis.legend(by_label.values(), by_label.keys(), loc=loc, fontsize=fontsize)

