import torch
import numpy as np
import time

def create_random(batch_size, num_heads,seq_len, emb_dim, seed=0):
    matrix = np.random.randn(batch_size, num_heads, seq_len, emb_dim)
    return matrix.reshape((batch_size * num_heads * seq_len, emb_dim))

dev = torch.device("cuda")
m = torch.tensor(create_random(2, 2, 4096, 64))

def scaled_time_taken(m):
        start_time = time.time()
        m = m.to(dev)
        torch.nn.functional.scaled_dot_product_attention(m,m,m)
        print(time.time() - start_time )
        m = m.to(torch.device('cpu'))
        
def time_taken(m):
        start_time = time.time()
        m = m.to(dev)
        torch.nn.functional.scaled_dot_product_attention(m,m,m)
        print(time.time() - start_time )
        m = m.to(torch.device('cpu'))
        



scaled_time_taken(m)
time_taken(m)