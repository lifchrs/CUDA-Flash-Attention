#include <cuda.h>
#include <cuda_runtime.h>
#include <cmath>    // for sqrt
#include <stdio.h>  // for printf
using matrix = float *;

#define CUDA_CHECK(call) { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error in %s at line %d: %s\n", \
                __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(1); \
    } \
}
__global__ 
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
    extern __shared__ float sram[];
    // float sram[(B_r * d + 2 * B_c * d + B_r * B_c)];
    matrix Q_i = sram;                      // Query tile: B_r × d
    matrix K_j = &sram[B_r * d];           // Key tile: B_c × d
    matrix V_j = &sram[B_r * d + B_c * d]; // Value tile: B_c × d
    matrix S = &sram[B_r * d + 2 * B_c * d]; // Score matrix: B_r × B_c

    for (int j = 0; j < T_c; j++) {
        for (int k_idx = 0; k_idx < B_c; k_idx++) {
            for (int h = 0; h < d; h++) {
                K_j[k_idx * d + h] = K[j * B_c * d + k_idx * d + h];
                V_j[k_idx * d + h] = V[j * B_c * d + k_idx * d + h];
            }
        }

        for (int i = 0; i < T_r; i++) {
            for (int q_idx = 0; q_idx < B_r; q_idx++) {
                for (int h = 0; h < d; h++) {
                    Q_i[q_idx * d + h] = Q[i * B_r * d + q_idx * d + h];
                }
            }

            for (int q_idx = 0; q_idx < B_r; q_idx++) {
                for (int k_idx = 0; k_idx < B_c; k_idx++) {
                    float score = 0.0f;
                    for (int h = 0; h < d; h++) {
                        score += Q_i[q_idx * d + h] * K_j[k_idx * d + h];
                    }
                    S[q_idx * B_c + k_idx] = score * scale;
                }
            }

            for (int q_idx = 0; q_idx < B_r; q_idx++) {
                float max_val = S[q_idx * B_c];
                for (int k_idx = 1; k_idx < B_c; k_idx++) {
                    max_val = max(max_val, S[q_idx * B_c + k_idx]);
                }

                float sum = 0.0f;
                for (int k_idx = 0; k_idx < B_c; k_idx++) {
                    S[q_idx * B_c + k_idx] = expf(S[q_idx * B_c + k_idx] - max_val);
                    sum += S[q_idx * B_c + k_idx];
                }

                const int row_idx = i * B_r + q_idx;
                float prev_m = m[row_idx];
                float prev_l = l[row_idx];
                float new_m = max(prev_m, max_val);
                float new_l = prev_l * expf(prev_m - new_m) + sum * expf(max_val - new_m);

                for (int h = 0; h < d; h++) {
                    float pv = 0.0f;  // P_ij * V_j
                    for (int k_idx = 0; k_idx < B_c; k_idx++) {
                        pv += S[q_idx * B_c + k_idx] * V_j[k_idx * d + h];
                    }
                    
                    const int out_idx = row_idx * d + h;

                    // fprintf(stderr, "out ind %d \n", out_idx);
                    O[out_idx] = (1.0f / new_l) * (
                        prev_l * expf(prev_m - new_m) * O[out_idx] +
                        expf(max_val - new_m) * pv
                    );
                    // fprintf(stderr, "O[%d] = %f\n", out_idx, O[out_idx]);

                }

                m[row_idx] = new_m;
                l[row_idx] = new_l;
            }
        }
    }
}

void forward(
    float* Q_h, float* K_h, float* V_h,
    const int B_c, const int B_r,
    const int grid_dim_x, const int grid_dim_y, const int grid_dim_z,
    const int block_dim_x, const int block_dim_y, const int block_dim_z,
    const int N, const int d,
    const int T_c, const int T_r,
    float* O_h, float* l_h, float* m_h
)
{
    dim3 grid_dim(grid_dim_x, grid_dim_y, grid_dim_z);
    dim3 block_dim(block_dim_x, block_dim_y, block_dim_z);

    const int sram_size = (B_r * d + 2 * B_c * d + B_r * B_c) * sizeof(float);
    const int matrix_size = N*d*sizeof(float);
    const int vector_size = N*sizeof(float);
    float *Q_d, *K_d, *V_d, *O_d, *l_d, *m_d;
    cudaMalloc(&Q_d, matrix_size);
    cudaMalloc(&K_d, matrix_size);
    cudaMalloc(&V_d, matrix_size);
    cudaMalloc(&O_d, matrix_size);
    cudaMalloc(&l_d, vector_size);
    cudaMalloc(&m_d, vector_size);

    cudaMemcpy(Q_d, Q_h, matrix_size, cudaMemcpyHostToDevice);
    cudaMemcpy(K_d, K_h, matrix_size, cudaMemcpyHostToDevice);
    cudaMemcpy(V_d, V_h, matrix_size, cudaMemcpyHostToDevice);
    cudaMemcpy(O_d, O_h, matrix_size, cudaMemcpyHostToDevice);
    cudaMemcpy(l_d, l_h, vector_size, cudaMemcpyHostToDevice);
    cudaMemcpy(m_d, m_h, vector_size, cudaMemcpyHostToDevice);
    fprintf(stderr, "shared memory: %d bytes\n", sram_size);
    serial_flash_attn_kernel<<<grid_dim, block_dim, sram_size>>>(
        N, d, Q_d, K_d, V_d, B_c, B_r, T_c, T_r, O_d, l_d, m_d
    );

    CUDA_CHECK(cudaGetLastError());

    cudaDeviceSynchronize();

    // copy results back to host
    cudaMemcpy(O_h, O_d, matrix_size, cudaMemcpyDeviceToHost);
    cudaMemcpy(l_h, l_d, vector_size, cudaMemcpyDeviceToHost);
    cudaMemcpy(m_h, m_d, vector_size, cudaMemcpyDeviceToHost);
    
    // Free device memory
    cudaFree(Q_d);
    cudaFree(K_d);
    cudaFree(V_d);
    cudaFree(O_d);
    cudaFree(l_d);
    cudaFree(m_d);
}