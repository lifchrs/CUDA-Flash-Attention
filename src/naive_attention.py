import torch
import torch.nn.functional as F
import math
import numpy as np


def manual_attention(query, key, value):
    query = torch.Tensor(query)
    key = torch.Tensor(key)
    value = torch.Tensor(value)
    scaling_factor = 1.0 / math.sqrt(key.size(-1))
    attention_scores = torch.matmul(query, key.transpose(-2, -1)) * scaling_factor
    attention_probs = F.softmax(attention_scores, dim=-1)
    attention_output = torch.matmul(attention_probs, value)

    return np.array(attention_output)
