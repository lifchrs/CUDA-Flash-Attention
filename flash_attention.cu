#include <cuda.h>

using matrix = float*;

__global__ void serial_flash_attn_kernel(
    const int N,      // sequence length
    const int d,      // hidden dimension
    matrix Q,         // query matrix
    matrix K,         // key matrix
    matrix V,         // value matrix
    const int B_c,    // column block size
    const int B_r,    // row block size
    const int T_c,    // number of column tiles
    const int T_r,    // number of row tiles
    matrix O          // output matrix
) {
    const float scale = 1.0f / sqrt(static_cast<float>(d));
    const int block_sz = d * B_c;
    
    // Allocate shared memory for tiles and intermediate results
    extern __shared__ float sram[];
    matrix Q_i = sram;                    // Query tile
    matrix K_j = sram + block_sz;         // Key tile
    matrix V_j = sram + 2 * block_sz;     // Value tile
    matrix S = sram + 3 * block_sz;       // Score matrix
    
    for (int j = 0; j < T_c; j++) {
        // Load K_j and V_j tiles to shared memory
        for (int x = 0; x < block_sz; x++) {
            K_j[x] = K[x + block_sz * j];
            V_j[x] = V[x + block_sz * j];
        }
        for (int i = 0; i < T_r; i++) {
            // Load Q_i tile to shared memory
            for (int x = 0; x < block_sz; x++) {
                Q_i[x] = Q[x + block_sz * i];
            }
            
            // Compute attention scores (Q_i * K_j^T)
            for (int q_idx = 0; q_idx < B_r; q_idx++) {
                for (int k_idx = 0; k_idx < B_c; k_idx++) {
                    float score = 0.0f;
                    for (int h = 0; h < d; h++) {
                        score += Q_i[q_idx * d + h] * K_j[k_idx * d + h];
                    }
                    S[q_idx * B_c + k_idx] = score * scale;
                }
            }
            
            // Apply softmax row-wise
            for (int q_idx = 0; q_idx < B_r; q_idx++) {
                float max_val = S[q_idx * B_c];
                for (int k_idx = 1; k_idx < B_c; k_idx++) {
                    max_val = max(max_val, S[q_idx * B_c + k_idx]);
                }
                
                // Compute exp and sum
                float sum = 0.0f;
                for (int k_idx = 0; k_idx < B_c; k_idx++) {
                    S[q_idx * B_c + k_idx] = __expf(S[q_idx * B_c + k_idx] - max_val);
                    sum += S[q_idx * B_c + k_idx];
                }
                
                // Normalize
                for (int k_idx = 0; k_idx < B_c; k_idx++) {
                    S[q_idx * B_c + k_idx] /= sum;
                }
            }
            
            // Compute output (S * V_j)
            for (int q_idx = 0; q_idx < B_r; q_idx++) {
                for (int h = 0; h < d; h++) {
                    float sum = 0.0f;
                    for (int k_idx = 0; k_idx < B_c; k_idx++) {
                        sum += S[q_idx * B_c + k_idx] * V_j[k_idx * d + h];
                    }
                    // Accumulate to global memory
                    O[(i * B_r + q_idx) * d + h] += sum;
                }
            }
        }
    }
}