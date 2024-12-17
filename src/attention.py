import torch
import os
from naive_attention import manual_attention
import pandas as pd
import subprocess
import sys
import numpy as np

def write_matrix(matrix, file_name):
    with open(file_name, "w") as file:
        for row in matrix:
            file.write(" ".join(f"{x:.30f}" for x in row) + "\n")
    return matrix


def create_random(batch_size, num_heads,seq_len, emb_dim, seed=0):
    np.random.seed(seed)
    matrix = np.random.randn(batch_size, num_heads, seq_len, emb_dim)
    return matrix.reshape((batch_size * num_heads * seq_len, emb_dim))
    # return (np.random.randn(seq_len, emb_dim)[np.newaxis, np.newaxis, :, :]+ np.zeros((batch_size, num_heads, seq_len, emb_dim))).reshape((batch_size * num_heads * seq_len, emb_dim))


def read_matrix(file_name, batch_size, num_heads,seq_len, emb_dim):
    return np.loadtxt(file_name, dtype=float).reshape(batch_size, num_heads,seq_len, emb_dim)


def run_from_frame(df, row, warmups=1, repeats=8, check=False):
    row = df.iloc[row]

    batch_size, num_heads, seq_len, emb_dim = row["batch_size"], row["num_heads"], row["seq_len"], row["emb_dim"]
    
    if check:
        q_matrix = write_matrix(create_random(batch_size, num_heads,seq_len, emb_dim), files["q_file"])
        k_matrix = write_matrix(create_random(batch_size, num_heads,seq_len, emb_dim), files["k_file"])
        v_matrix = write_matrix(create_random(batch_size, num_heads,seq_len, emb_dim), files["v_file"])

    B_c, B_r = row["B_c"], row["B_r"]
    block_dim_y = row["block_dim_y"]
    
    if not(seq_len % B_r == 0 and emb_dim % B_c == 0):
        print("Block size doesn't divide")
        return -1
    if not((B_r * emb_dim + 2 * B_c * emb_dim + B_r * B_c) * 4 <= 98304):
        print("Using too much shmem")
        return -2

    result = subprocess.run(
        [
            "./build/attention",
            str(repeats),
            str(warmups),
            "1" if check else "0",
            files["q_file"] if check else "none",
            files["k_file"] if check else "none",
            files["v_file"] if check else "none",
            str(output_file) if check else "none",
            str(batch_size),
            str(num_heads),
            str(seq_len),
            str(emb_dim),
            str(B_c),
            str(B_r),
            str(block_dim_y),
            "1" if row["use_parallel"] else "0",
        ],
        capture_output=True,
        text=True,
    )

    print(result)
    
    if check:             
        output = read_matrix(output_file, batch_size, num_heads, seq_len, emb_dim)
        q_matrix = q_matrix.reshape(batch_size, num_heads, seq_len, emb_dim)
        k_matrix = k_matrix.reshape(batch_size, num_heads, seq_len, emb_dim)
        v_matrix = v_matrix.reshape(batch_size, num_heads, seq_len, emb_dim)
                
        expected_output = manual_attention(q_matrix, k_matrix, v_matrix)
        print(output.shape, expected_output.shape)
        # print(expected_output)
        error = np.abs(expected_output - output)

        print("max error", error.max())
        print("mean error", error.mean())
        return result.stdout, error.max(), error.mean()
    
    if result.stdout:
        return float(result.stdout)/1_000_000
    else:
        return -3

# timing_csv_names = ["timing_csvs/emb_dim.csv"]
timing_csv_names = ["timing_csvs/B_c.csv","timing_csvs/B_r.csv",
                    "timing_csvs/batch_size.csv", "timing_csvs/block_dim_y.csv",
                    "timing_csvs/emb_dim.csv","timing_csvs/num_heads.csv",
                    "timing_csvs/parallel_vs_serial_timing.csv", "timing_csvs/seq_length.csv"]

timing_csv_names = ["timing_csvs/grid_search.csv"]

files = {
    "q_file": "q_matrix.txt",
    "k_file": "k_matrix.txt",
    "v_file": "v_matrix.txt",
}
output_file = "output.txt"
check = False

for file_name in timing_csv_names:
    df = pd.read_csv(file_name)

    if check:
        global_max_error = 0
        for row in range(len(df)):
            time, error_max, error_mean = run_from_frame(df, row, check=True)
            global_max_error = max(global_max_error, error_max)
        print(global_max_error)
    else:
        df["time"] = df.index.to_series().apply(lambda row: run_from_frame(df, row))
        df.to_csv(file_name,index=False)
