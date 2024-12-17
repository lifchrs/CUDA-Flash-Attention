import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def plot_and_save(df, x_col, y_col, filename):
    x = df[x_col]
    y = df[y_col]

    fig, ax = plt.subplots(1, 2, figsize=(12, 6))

    ax[0].plot(x, y)
    ax[0].set_title('Regular Plot')
    ax[0].set_xlabel(x_col)
    ax[0].set_ylabel(y_col)

    ax[1].loglog(x, y)
    ax[1].set_title('Log-Log Plot')
    ax[1].set_xlabel(x_col)
    ax[1].set_ylabel(y_col)

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
one_var = ["B_c","B_r","batch_size", "block_dim_y","emb_dim","num_heads"] # , "seq_length"]


df = pd.read_csv(f"timing_csvs/grid_search.csv")

df = df[df['time'] >= 0]
group_plot_and_save(df, "B_r", "time", "B_c", "timing_plots/grid_B_r.png")
group_plot_and_save(df, "B_c", "time", "B_r", "timing_plots/grid_B_c.png")

for var in one_var:
    df = pd.read_csv(f"timing_csvs/{var}.csv")
    print(df)
    df = df[df['time'] >= 0]
    plot_and_save(df, var, 'time', f"timing_plots/{var}.png")


