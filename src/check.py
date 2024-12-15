import numpy as np
from naive_attention import manual_attention
import torch


def write_matrix(matrix, file_name):
    with open(file_name, "w") as file:
        for row in matrix:
            file.write(" ".join(f"{x:.9f}" for x in row) + "\n")
    return matrix


def create_random(batch_size, num_heads,seq_len, emb_dim, seed=0):
    np.random.seed(seed)
    matrix = np.random.uniform(size=(batch_size, num_heads,seq_len, emb_dim))
    return matrix
    # return [[1.0,2.0],[3.0,4.0]]


def read_matrix(file_name, batch_size, num_heads,seq_len, emb_dim):
    return np.loadtxt(file_name, dtype=float)


def check_attention(q_file, k_file, v_file, result_file, attn_opp=manual_attention):
    q = read_matrix(q_file)
    k = read_matrix(k_file)
    v = read_matrix(v_file)
    res = read_matrix(result_file)

    try:
        return np.max(np.abs(attn_opp(q, k, v) - res))
    except Exception:
        return float("inf")


q = read_matrix("12x10.txt")
k = read_matrix("12x10.txt")
v = read_matrix("12x10.txt")
write_matrix(manual_attention(q, k, v), "ans.txt")
res = check_attention("12x10.txt", "12x10.txt", "12x10.txt", "ans.txt")
print(res)
