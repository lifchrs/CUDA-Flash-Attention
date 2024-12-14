#include <cuda.h>
#include <cuda_runtime.h>
#include <cmath>    // for sqrt
#include <stdio.h>  // for printf
using matrix = float *;

// __global__ 

void serial_flash_attn_kernel(
    const int N,   // sequence length
    const int d,   // hidden dimension
    matrix Q,      // query matrix
    matrix K,      // key matrix
    matrix V,      // value matrix
    const int B_c, // column block size
    const int B_r, // row block size
    const int T_c, // number of column tiles
    const int T_r, // number of row tiles
    matrix O,      // output matrix
    float* l,      // normalizing factors
    float* m       // max values for stability
)
{
    const float scale = 1.0f / sqrt(static_cast<float>(d));
    
    // Correct shared memory allocation
    // extern __shared__ float sram[];
    float sram[B_r*d*4];
    matrix Q_i = sram;                      // Query tile: B_r × d
    matrix K_j = &sram[B_r * d];           // Key tile: B_c × d
    matrix V_j = &sram[B_r * d + B_c * d]; // Value tile: B_c × d
    matrix S = &sram[B_r * d + 2 * B_c * d]; // Score matrix: B_r × B_c

    for (int j = 0; j < T_c; j++) {
        // Load K_j and V_j tiles
        for (int k_idx = 0; k_idx < B_c; k_idx++) {
            for (int h = 0; h < d; h++) {
                K_j[k_idx * d + h] = K[j * B_c * d + k_idx * d + h];
                V_j[k_idx * d + h] = V[j * B_c * d + k_idx * d + h];
            }
        }

        for (int i = 0; i < T_r; i++) {
            // Load Q_i tile
            for (int q_idx = 0; q_idx < B_r; q_idx++) {
                for (int h = 0; h < d; h++) {
                    Q_i[q_idx * d + h] = Q[i * B_r * d + q_idx * d + h];
                    // fprintf(stderr, "copied %f over for q\n", Q[i * B_r * d + q_idx * d + h]);
                }
            }

            // Compute attention scores (Q_i * K_j^T)
            for (int q_idx = 0; q_idx < B_r; q_idx++) {
                for (int k_idx = 0; k_idx < B_c; k_idx++) {
                    float score = 0.0f;
                    for (int h = 0; h < d; h++) {
                        score += Q_i[q_idx * d + h] * K_j[k_idx * d + h];
                    }
                    S[q_idx * B_c + k_idx] = score;// * scale;
                    // fprintf(stderr, "computerd score %f \n", S[q_idx * B_c + k_idx]);
                }
            }

            // Apply softmax row-wise and update output
            for (int q_idx = 0; q_idx < B_r; q_idx++) {
                // Find max for numerical stability
                float max_val = S[q_idx * B_c];
                for (int k_idx = 1; k_idx < B_c; k_idx++) {
                    max_val = max(max_val, S[q_idx * B_c + k_idx]);
                }

                // Compute exp and sum
                float sum = 0.0f;
                for (int k_idx = 0; k_idx < B_c; k_idx++) {
                    S[q_idx * B_c + k_idx] = expf(S[q_idx * B_c + k_idx] - max_val);
                    sum += S[q_idx * B_c + k_idx];
                }

                // Get previous m and l values
                const int row_idx = i * B_r + q_idx;
                float prev_m = m[row_idx];
                float prev_l = l[row_idx];
                float new_m = max(prev_m, max_val);
                float new_l = prev_l * expf(prev_m - new_m) + sum * expf(max_val - new_m);

                // Update output with flash attention formula
                for (int h = 0; h < d; h++) {
                    float pv = 0.0f;  // P_ij * V_j
                    for (int k_idx = 0; k_idx < B_c; k_idx++) {
                        pv += S[q_idx * B_c + k_idx] * V_j[k_idx * d + h];
                    }
                    
                    const int out_idx = row_idx * d + h;

                    fprintf(stderr, "out ind %d \n", out_idx);
                    O[out_idx] = (1.0f / new_l) * (
                        prev_l * expf(prev_m - new_m) * O[out_idx] +
                        expf(max_val - new_m) * pv
                    );
                }

                // Update m and l
                m[row_idx] = new_m;
                l[row_idx] = new_l;
            }
        }
    }
}

void forward(
    float* Q, float* K, float* V,
    const int B_c, const int B_r,
    const int grid_dim_x, const int grid_dim_y, const int grid_dim_z,
    const int block_dim_x, const int block_dim_y, const int block_dim_z,
    const int N, const int d,
    const int T_c, const int T_r,
    float* O, float* l, float* m
)
{
    dim3 grid_dim(grid_dim_x, grid_dim_y, grid_dim_z);
    dim3 block_dim(block_dim_x, block_dim_y, block_dim_z);
    
    // Correct shared memory size calculation
    const int sram_size = (B_r * d + 2 * B_c * d + B_r * B_c) * sizeof(float);

    // serial_flash_attn_kernel<<<grid_dim, block_dim, sram_size>>>(
    //     N, d, Q, K, V, B_c, B_r, T_c, T_r, O, l, m
    // );

    serial_flash_attn_kernel(
        N, d, Q, K, V, B_c, B_r, T_c, T_r, O, l, m
    );
    for(int i = 0; i < 2; i++) for(int j =0 ; j < 2; j++) fprintf(stderr, "%f ", Q[i*2+j]);
    fprintf(stderr, "\n");

    for(int i = 0; i < 2; i++) for(int j =0 ; j < 2; j++) fprintf(stderr, "%f ", K[i*2+j]);
    fprintf(stderr, "\n");

    for(int i = 0; i < 2; i++) for(int j =0 ; j < 2; j++) fprintf(stderr, "%f ", V[i*2+j]);
    fprintf(stderr, "\n");


    for(int i = 0; i < 2; i++) for(int j =0 ; j < 2; j++) fprintf(stderr, "%f ", O[i*2+j]);
    fprintf(stderr, "\n");
}