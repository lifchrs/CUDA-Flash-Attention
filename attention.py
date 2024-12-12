from torch.utils.cpp_extension import load
import torch

# Load the CUDA kernel as a python module
minimal_attn = load(
    name="minimal_attn", sources=["main.cpp", "flash.cu"], extra_cuda_cflags=["-O2"]
)

seq_len, embd = (4096, 512)


q = torch.randn(seq_len, embd).cuda()
k = torch.randn(seq_len, embd).cuda()
v = torch.randn(seq_len, embd).cuda()

B_c, B_r = (32, 32)

grid_dim = (seq_len, 1, 1)
block_dim = (B_c, 1, 1)
with torch.autograd.profiler.profile(use_cuda=True) as prof:
    minimal_result = minimal_attn.forward(
        q,
        k,
        v,
        B_c,
        B_r,
    )
