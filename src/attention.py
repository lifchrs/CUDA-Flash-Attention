# from torch.utils.cpp_extension import load
import torch
import os
from check import *
import naive_attention
import pandas as pd
import subprocess
import sys

df = pd.read_csv("timings.csv")

# print(df)


def run_from_frame(df, row, warmups=2, repeats=2, print_o_matrix=0, check=False):
    row = df.iloc[row]

    # print("there is the row")
    # print(row)

    seq_len, emb_dim = row["seq_len"], row["emb_dim"]
    
    q_matrix = write_matrix(create_random(seq_len, emb_dim), files["q_file"])
    k_matrix = write_matrix(create_random(seq_len, emb_dim), files["k_file"])
    v_matrix = write_matrix(create_random(seq_len, emb_dim), files["v_file"])

    B_c, B_r = row["B_c"], row["B_r"]
    grid_dim_x, grid_dim_y, grid_dim_z = (
        row["grid_dim_x"],
        row["grid_dim_y"],
        row["grid_dim_z"],
    )
    block_dim_x, block_dim_y, block_dim_z = (
        row["block_dim_x"],
        row["block_dim_y"],
        row["block_dim_y"],
    )

    result = subprocess.run(
        [
            "./build/attention",
            files["q_file"],
            files["k_file"],
            files["v_file"],
            str(B_c),
            str(B_r),
            str(grid_dim_x),
            str(grid_dim_y),
            str(grid_dim_z),
            str(block_dim_x),
            str(block_dim_y),
            str(block_dim_z),
            output_file,
            str(print_o_matrix),
            str(repeats),
            str(warmups),
        ],
        capture_output=True,
        text=True,
    )

    print(result)
    
    if check:             
        output = read_matrix(output_file)
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



sys.exit()

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