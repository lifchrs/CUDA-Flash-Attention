import numpy as np
from naive_attention import manual_attention
import torch


def write_matrix(matrix, file_name):
    with open(file_name, "w") as file:
        for row in matrix:
            file.write(" ".join(map(str, row)) + "\n")
    return matrix


def create_random(width, height, seed=0):
    np.random.seed(seed)
    matrix = np.random.uniform(size=(height, width))
    # return matrix
    return [[1,2],[3,4]]

def write_random(width, height, seed=0):
    np.random.seed(seed)
    matrix = np.random.uniform(size=(height, width))
    write_matrix(matrix, f"{height}x{width}.txt")


def read_matrix(file_name):
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
