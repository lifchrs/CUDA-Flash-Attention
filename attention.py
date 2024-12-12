from torch.utils.cpp_extension import load
import torch
import os
import check
import naive_attention
# Load the CUDA kernel as a python module
minimal_attn = load(
    name="flash_attention", sources=["main.cpp", "flash_attention.cu"], extra_cuda_cflags=["-O2"]
)

seq_len, embd = (4096, 512)


q = torch.randn(seq_len, embd).cuda()
k = torch.randn(seq_len, embd).cuda()
v = torch.randn(seq_len, embd).cuda()

B_c, B_r = (32, 32)

grid_dim = (seq_len, 1, 1)
block_dim = (B_c, 1, 1)
with torch.autograd.profiler.profile(use_cuda=True) as prof:
    files = {
        'q_file': 'q_matrix.txt',
        'k_file': 'k_matrix.txt',
        'v_file': 'v_matrix.txt',
    }
    
    q_matrix = write_matrix(write_random(32*4, 32*4), files['q_file'])
    k_matrix = write_matrix(write_random(32*4, 32*4), files['k_file'])
    v_matrix = write_matrix(write_random(32*4, 32*4), files['v_file'])
    
    B_c = 32
    B_r = 32
    grid_dim_x = 1
    grid_dim_y = 1
    grid_dim_z = 1
    block_dim_x = 16
    block_dim_y = 16
    block_dim_z = 16
    output_file = 'output.txt'
    cmd = f'./main {q_file} {k_file} {v_file} {B_c} {B_r} {grid_dim_x} {grid_dim_y} {grid_dim_z} {block_dim_x} {block_dim_y} {block_dim_z} > {output_file}'
    
    os.system(cmd) 
    output = read_matrix(output_file)
    q_read_matrix(files['q_file'])
    expected_output = manual_attention(q_matrix, k_matrix, v_matrix)
    print(output - expected_output)
    # minimal_result = minimal_attn.forward(q, k, v, B_c, B_r, *grid_dim, *block_dim)
