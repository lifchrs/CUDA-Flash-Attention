# from torch.utils.cpp_extension import load
import torch
import os
from naive_attention import manual_attention
import pandas as pd
import subprocess
import sys
import numpy as np

df = pd.read_csv("timings.csv")

# print(df)


def write_matrix(matrix, file_name):
    with open(file_name, "w") as file:
        for row in matrix:
            file.write(" ".join(f"{x:.30f}" for x in row) + "\n")
    return matrix


def create_random(batch_size, num_heads,seq_len, emb_dim, seed=0):
    np.random.seed(seed)
    # matrix = np.random.randn(batch_size, num_heads, seq_len, emb_dim)
    # return matrix.reshape((batch_size * num_heads * seq_len, emb_dim))
    return (np.random.randn(seq_len, emb_dim)[np.newaxis, np.newaxis, :, :]+ np.zeros((batch_size, num_heads, seq_len, emb_dim))).reshape((batch_size * num_heads * seq_len, emb_dim))


def read_matrix(file_name, batch_size, num_heads,seq_len, emb_dim):
    return np.loadtxt(file_name, dtype=float).reshape(batch_size, num_heads,seq_len, emb_dim)


def run_from_frame(df, row, warmups=3, repeats=2, print_o_matrix=False, check=False):
    row = df.iloc[row]

    batch_size, num_heads, seq_len, emb_dim = row["batch_size"], row["num_heads"], row["seq_len"], row["emb_dim"]
    
    q_matrix = write_matrix(create_random(batch_size, num_heads,seq_len, emb_dim), files["q_file"])
    k_matrix = write_matrix(create_random(batch_size, num_heads,seq_len, emb_dim), files["k_file"])
    v_matrix = write_matrix(create_random(batch_size, num_heads,seq_len, emb_dim), files["v_file"])

    B_c, B_r = row["B_c"], row["B_r"]
    block_dim_y = row["block_dim_y"]

    result = subprocess.run(
        [
            "./build/attention",
            str(repeats),
            str(warmups),
            "1" if print_o_matrix else "0",
            files["q_file"],
            files["k_file"],
            files["v_file"],
            str(output_file),
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
    
    print(output_file, "done \n\n")
    print(result)
    print("after res\n")
    
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
    return result.stdout


files = {
    "q_file": "q_matrix.txt",
    "k_file": "k_matrix.txt",
    "v_file": "v_matrix.txt",
}
output_file = "output.txt"
run_from_frame(df, 0, check=True)


# N = 32*4
# d = 100
# q_matrix = write_matrix(create_random(N, d), files["q_file"])
# k_matrix = write_matrix(create_random(N, d), files["k_file"])
# v_matrix = write_matrix(create_random(N, d), files["v_file"])

# B_c = 16
# B_r = 16
# grid_dim_x = 1
# grid_dim_y = 1
# grid_dim_z = 1
# block_dim_x = 16
# block_dim_y = 16
# block_dim_z = 16
# cmd = f'./build/attention {files["q_file"]} {files["k_file"]} {files["v_file"]} {B_c} {B_r} {grid_dim_x} {grid_dim_y} {grid_dim_z} {block_dim_x} {block_dim_y} {block_dim_z} > {output_file}'

# os.system(cmd)

# output = read_matrix(output_file)
# # q_read_matrix(files['q_file'])
# expected_output = manual_attention(q_matrix, k_matrix, v_matrix)
# error = expected_output - output

# print("max error", error.max())
# print("mean error", error.mean())