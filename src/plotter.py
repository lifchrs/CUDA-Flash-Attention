import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

import matplotlib.pyplot as plt
import numpy as np


def plot_and_save(df, x_col, y_col, filename, show_loglog=True):
    x = df[x_col]
    y = df[y_col]

    if show_loglog:
        # Perform log-log linear fit with base-2 logarithm
        log_x = np.log2(x)
        log_y = np.log2(y)
        slope, intercept = np.polyfit(log_x, log_y, 1)

        fig, ax = plt.subplots(1, 2, figsize=(12, 6))

        # Regular Plot
        ax[0].plot(x, y)
        ax[0].set_title(f'{x_col.capitalize()} vs Time(ms)')
        ax[0].set_xlabel(x_col)
        ax[0].set_ylabel(y_col)

        # Log-log plot with base 2
        ax[1].scatter(x, y, label='Data')
        ax[1].plot(x, 2**(slope * np.log2(x) + intercept), '--', color='red',
                   label=f'Fit: slope={slope:.2f}')
        ax[1].set_xscale('log', base=2)
        ax[1].set_yscale('log', base=2)
        ax[1].set_title('Log-Log')
        ax[1].set_xlabel(x_col)
        ax[1].set_ylabel(y_col)
        ax[1].legend()

    else:
        fig, ax = plt.subplots(1, 1, figsize=(6, 6))
        ax.plot(x, y)
        ax.set_title(f'{x_col.capitalize()} vs Time(ms)')
        ax.set_xlabel(x_col)
        ax.set_ylabel(y_col)

    plt.tight_layout()
    plt.savefig(filename)


def group_plot_and_save(df, x_col, y_col, group_col, filename):
    # Create subplots with two figures (1 row, 2 columns)
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))

    # Regular plot on the first axis
    for group_value in df[group_col].unique():
        group_data = df[df[group_col] == group_value]
        ax[0].plot(group_data[x_col], group_data[y_col], label=f'{group_col}={group_value}')
    
    ax[0].set_title('Timing')
    ax[0].set_xlabel(x_col)
    ax[0].set_ylabel(y_col)
    ax[0].legend()

    # Log-log plot on the second axis
    for group_value in df[group_col].unique():
        group_data = df[df[group_col] == group_value]
        ax[1].loglog(group_data[x_col], group_data[y_col], label=f'{group_col}={group_value}')
    
    ax[1].set_title('Log-Log Plot')
    ax[1].set_xlabel(x_col)
    ax[1].set_ylabel(y_col)
    ax[1].legend()

    # Adjust the layout to prevent overlap
    plt.tight_layout()

    # Save the figure as a file
    plt.savefig(filename)

    # Optionally, display the plot
    plt.show()
one_var = {"B_c":True,"B_r":True,"batch_size":False, "emb_dim":True,"num_heads":False, "seq_len":True}


df = pd.read_csv(f"timing_csvs/grid_search.csv")

df = df[df['time'] >= 0]
group_plot_and_save(df, "B_r", "time", "B_c", "timing_plots/grid_B_r.png")
group_plot_and_save(df, "B_c", "time", "B_r", "timing_plots/grid_B_c.png")

for var, log in one_var.items():
    df = pd.read_csv(f"timing_csvs/{var}.csv")
    print(df)
    df = df[df['time'] >= 0]
    plot_and_save(df, var, 'time', f"timing_plots/{var}.png", show_loglog=True  )


