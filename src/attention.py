# from torch.utils.cpp_extension import load
import torch
import os
import naive_attention
import pandas as pd
import subprocess
import sys

df = pd.read_csv("timings.csv")

# print(df)


def write_matrix(matrix, file_name):
    with open(file_name, "w") as file:
        for row in matrix:
            file.write(" ".join(f"{x:.9f}" for x in row) + "\n")
    return matrix


def create_random(batch_size, num_heads,seq_len, emb_dim, seed=0):
    np.random.seed(seed)
    matrix = np.random.uniform(size=(batch_size, num_heads,seq_len, emb_dim))
    return matrix.reshape((batch_size * num_heads * seq_len, emb_dim))
    # return [[1.0,2.0],[3.0,4.0]]


def read_matrix(file_name, batch_size, num_heads,seq_len, emb_dim):
    return np.loadtxt(file_name, dtype=float).reshape(batch_size, num_heads,seq_len, emb_dim)


def run_from_frame(df, row, warmups=2, repeats=2, print_o_matrix=False, check=False):
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
            if print_o_matrix "1" else "0",
            files["q_file"],
            files["k_file"],
            files["v_file"],
            output_file,
            batch_size,
            num_heads,
            seq_len,
            emb_dim
            str(B_c),
            str(B_r),
            str(block_dim_y),
            if row["use_parallel"] "1" else "0",
        ],
        capture_output=True,
        text=True,
    )
    
    if check:             
        output = read_matrix(output_file, batch_size, num_heads,seq_len, emb_dim)
        expected_output = manual_attention(q_matrix, k_matrix, v_matrix)
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