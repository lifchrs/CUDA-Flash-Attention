#include <cuda.h>
#include <cuda_runtime.h>
#include <cmath>   // for sqrt
#include <stdio.h> // for printf
using matrix = float *;

#define CUDA_CHECK(call)                                          \
    {                                                             \
        cudaError_t err = call;                                   \
        if (err != cudaSuccess)                                   \
        {                                                         \
            fprintf(stderr, "CUDA error in %s at line %d: %s\n",  \
                    __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(1);                                              \
        }                                                         \
    }

__global__ void serial_flash_attn_kernel(
    const int batch_size, // Batch size
    const int num_heads,  // Number of heads
    const int N,          // sequence length
    const int d,          // hidden dimension
    matrix Q,             // query matrix
    matrix K,             // key matrix
    matrix V,             // value matrix
    const int B_c,        // column block size
    const int B_r,        // row block size
    const int T_c,        // number of column tiles
    const int T_r,        // number of row tiles
    matrix O,             // output matrix
    float *l,             // normalizing factors
    float *m              // max values for stability
)
{
    const float scale = 1.0f / sqrt(static_cast<float>(d));

    // Correct shared memory allocation
    extern __shared__ float sram[];
    matrix Q_i = sram;                       // Query tile: B_r × d
    matrix K_j = &sram[B_r * d];             // Key tile: B_c × d
    matrix V_j = &sram[B_r * d + B_c * d];   // Value tile: B_c × d
    matrix S = &sram[B_r * d + 2 * B_c * d]; // Score matrix: B_r × B_c

    for (int batch = 0; batch < batch_size; batch++)
    {
        for (int head = 0; head < num_heads; head++)
        {
            const int matrix_head_batch_offset = (num_heads * batch + head) * (N * d);
            const int vector_head_batch_offset = (num_heads * batch + head) * N;

            for (int j = 0; j < T_c; j++)
            {
                for (int k_idx = 0; k_idx < B_c; k_idx++)
                {
                    for (int h = 0; h < d; h++)
                    {
                        K_j[k_idx * d + h] = K[matrix_head_batch_offset + j * B_c * d + k_idx * d + h];
                        V_j[k_idx * d + h] = V[matrix_head_batch_offset + j * B_c * d + k_idx * d + h];
                    }
                }

                for (int i = 0; i < T_r; i++)
                {
                    for (int q_idx = 0; q_idx < B_r; q_idx++)
                    {
                        for (int h = 0; h < d; h++)
                        {
                            Q_i[q_idx * d + h] = Q[matrix_head_batch_offset + i * B_r * d + q_idx * d + h];
                        }
                    }

                    for (int q_idx = 0; q_idx < B_r; q_idx++)
                    {
                        for (int k_idx = 0; k_idx < B_c; k_idx++)
                        {
                            float score = 0.0f;
                            for (int h = 0; h < d; h++)
                            {
                                score += Q_i[q_idx * d + h] * K_j[k_idx * d + h];
                            }
                            S[q_idx * B_c + k_idx] = score * scale;
                        }
                    }

                    // for (int ch = 0; ch < B_r * B_c; ch++)
                    // {
                    //     printf("%d ", S[ch]);
                    // }
                    // printf("\n\n\n\n");

                    for (int q_idx = 0; q_idx < B_r; q_idx++)
                    {
                        float max_val = S[q_idx * B_c];
                        for (int k_idx = 1; k_idx < B_c; k_idx++)
                        {
                            max_val = max(max_val, S[q_idx * B_c + k_idx]);
                        }

                        float sum = 0.0f;
                        for (int k_idx = 0; k_idx < B_c; k_idx++)
                        {
                            S[q_idx * B_c + k_idx] = expf(S[q_idx * B_c + k_idx] - max_val);
                            sum += S[q_idx * B_c + k_idx];
                        }

                        const int row_idx = i * B_r + q_idx;
                        float prev_m = m[vector_head_batch_offset + row_idx];
                        float prev_l = l[vector_head_batch_offset + row_idx];
                        float new_m = max(prev_m, max_val);
                        float new_l = prev_l * expf(prev_m - new_m) + sum * expf(max_val - new_m);

                        for (int h = 0; h < d; h++)
                        {
                            float pv = 0.0f; // P_ij * V_j
                            for (int k_idx = 0; k_idx < B_c; k_idx++)
                            {
                                pv += S[q_idx * B_c + k_idx] * V_j[k_idx * d + h];
                            }

                            const int out_idx = matrix_head_batch_offset + row_idx * d + h;

                            // fprintf(stderr, "out ind %d \n", out_idx);
                            O[out_idx] = (1.0f / new_l) * (prev_l * expf(prev_m - new_m) * O[out_idx] +
                                                           expf(max_val - new_m) * pv);
                            // fprintf(stderr, "O[%d] = %f\n", out_idx, O[out_idx]);
                        }

                        m[vector_head_batch_offset + row_idx] = new_m;
                        l[vector_head_batch_offset + row_idx] = new_l;
                    }
                }
            }
        }
    }
}

void forward_serial(
    float *Q_h,           // Query matrix
    float *K_h,           // Key matrix
    float *V_h,           // Value matrix
    float *O_h,           // Output matrix
    float *m_h,           // Running maximum values
    float *l_h,           // Running sum of exponentials
    const int B_c,        // Column block size
    const int B_r,        // Row block size
    const int batch_size, // Batch size
    const int num_heads,  // Number of heads
    const int seq_len,    // Sequence Length
    const int d,          // Hidden dimension
    const int block_dim_y // Unused
)
{
    CUDA_CHECK(cudaFuncSetAttribute(serial_flash_attn_kernel,
                                    cudaFuncAttributeMaxDynamicSharedMemorySize,
                                    98304));
    dim3 grid_dim(1, 1, 1);
    dim3 block_dim(1, 1, 1);

    const int sram_size = (B_r * d + 2 * B_c * d + B_r * B_c) * sizeof(float);
    const int matrix_size = batch_size * num_heads * seq_len * d * sizeof(float);
    const int vector_size = batch_size * num_heads * seq_len * sizeof(float);
    const int T_c = (seq_len + B_c - 1) / B_c;
    const int T_r = (seq_len + B_r - 1) / B_r;

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
    // fprintf(stderr, "shared memory: %d bytes\n", sram_size);
    serial_flash_attn_kernel<<<grid_dim, block_dim, sram_size>>>(
        batch_size, num_heads, seq_len, d, Q_d, K_d, V_d, B_c, B_r, T_c, T_r, O_d, l_d, m_d);

    CUDA_CHECK(cudaGetLastError());

    cudaDeviceSynchronize();

    // copy results back to host
    cudaMemcpy(O_h, O_d, matrix_size, cudaMemcpyDeviceToHost);
    // cudaMemcpy(l_h, l_d, vector_size, cudaMemcpyDeviceToHost);
    // cudaMemcpy(m_h, m_d, vector_size, cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(K_d);
    cudaFree(V_d);
    cudaFree(Q_d);
    cudaFree(O_d);
    cudaFree(l_d);
    cudaFree(m_d);
}

__global__ void parallel_flash_attn_kernel(
    const int batch_size, // Batch size
    const int num_heads,  // Number of heads
    const int N,          // sequence length
    const int d,          // hidden dimension
    matrix Q,             // query matrix
    matrix K,             // key matrix
    matrix V,             // value matrix
    const int B_c,        // column block size
    const int B_r,        // row block size
    const int T_c,        // number of column tiles
    const int T_r,        // number of row tiles
    matrix O,             // output matrix
    float *l,             // normalizing factors
    float *m              // max values for stability
)
{
    // printf("|| %d %d %d %d %d %d||", threadIdx.x, threadIdx.y, threadIdx.z, blockIdx.x, blockIdx.y, blockIdx.z);
    const float scale = 1.0f / sqrt(static_cast<float>(d));

    // Correct shared memory allocation
    extern __shared__ float sram[];
    matrix Q_i = sram;                       // Query tile: B_r × d
    matrix K_j = &sram[B_r * d];             // Key tile: B_c × d
    matrix V_j = &sram[B_r * d + B_c * d];   // Value tile: B_c × d
    matrix S = &sram[B_r * d + 2 * B_c * d]; // Score matrix: B_r × B_c

    const int batch_idx = blockIdx.x;
    const int head_idx = blockIdx.y;

    const int matrix_head_batch_offset = (num_heads * batch_idx + head_idx) * (N * d);
    const int vector_head_batch_offset = (num_heads * batch_idx + head_idx) * N;

    for (int j = 0; j < T_c; j++)
    {

        for (int k_idx = threadIdx.x; k_idx < B_c; k_idx += blockDim.x) // Can be parallelized
        {
            for (int h = 0; h < d; h++)
            {
                K_j[k_idx * d + h] = K[matrix_head_batch_offset + j * B_c * d + k_idx * d + h];
                V_j[k_idx * d + h] = V[matrix_head_batch_offset + j * B_c * d + k_idx * d + h];
            }
        }

        // threads needs to be synchronized within block since
        // each thread multiplies a row of Q with a tile of K and V
        // and we need to make sure that the tiles of K and V are in sram
        __syncthreads();

        // printf("%d %d %d %d \n", threadIdx.x, threadIdx.y, T_r, B_r);
        for (int i = 0; i < T_r; i++)
        // for (int i = 0; i < T_r; i ++)
        {
            const int tile_row_idx = threadIdx.x;

            for (int h = 0; h < d; h++)
            {
                Q_i[tile_row_idx * d + h] = Q[matrix_head_batch_offset + i * B_r * d + tile_row_idx * d + h];
            }
            for (int k_idx = 0; k_idx < B_c; k_idx++)
            {
                float score = 0.0f;
                for (int h = 0; h < d; h++)
                {
                    score += Q_i[tile_row_idx * d + h] * K_j[k_idx * d + h];
                }
                S[tile_row_idx * B_c + k_idx] = score * scale;
            }
            float max_val = S[tile_row_idx * B_c];
            for (int k_idx = 1; k_idx < B_c; k_idx++)
            {
                max_val = max(max_val, S[tile_row_idx * B_c + k_idx]);
            }

            float sum = 0.0f;
            for (int k_idx = 0; k_idx < B_c; k_idx++)
            {
                S[tile_row_idx * B_c + k_idx] = expf(S[tile_row_idx * B_c + k_idx] - max_val);
                sum += S[tile_row_idx * B_c + k_idx];
            }

            const int output_row_idx = i * B_r + tile_row_idx;
            float prev_m = m[vector_head_batch_offset + output_row_idx];
            float prev_l = l[vector_head_batch_offset + output_row_idx];
            float new_m = max(prev_m, max_val);
            float new_l = prev_l * expf(prev_m - new_m) + sum * expf(max_val - new_m);

            for (int k = 0; k < d; k++)
            {
                float pv = 0.0f;
                for (int k_idx = 0; k_idx < B_c; k_idx++)
                {
                    pv += S[tile_row_idx * B_c + k_idx] * V_j[k_idx * d + k];
                }

                const int out_idx = matrix_head_batch_offset + output_row_idx * d + k;

                O[out_idx] = (1.0f / new_l) * (prev_l * expf(prev_m - new_m) * O[out_idx] +
                                               expf(max_val - new_m) * pv);
            }

            m[vector_head_batch_offset + output_row_idx] = new_m;
            l[vector_head_batch_offset + output_row_idx] = new_l;
            // __syncthreads();
        }
    }
}

void forward_parallel(
    float *Q_h,           // Query matrix
    float *K_h,           // Key matrix
    float *V_h,           // Value matrix
    float *O_h,           // Output matrix
    float *m_h,           // Running maximum values
    float *l_h,           // Running sum of exponentials
    const int B_c,        // Column block size
    const int B_r,        // Row block size
    const int batch_size, // Batch size
    const int num_heads,  // Number of heads
    const int seq_len,    // Sequence Length
    const int d,          // Hidden dimension
    const int block_dim_y // Inversely proportional to how many tiles a thread attends to
)
{
    CUDA_CHECK(cudaFuncSetAttribute(serial_flash_attn_kernel,
                                    cudaFuncAttributeMaxDynamicSharedMemorySize,
                                    98304));
    dim3 grid_dim(batch_size, num_heads);
    dim3 block_dim(B_r, block_dim_y);
    // block_dim = dim3(B_r, 1);

    const int sram_size = (B_r * d + 2 * B_c * d + B_r * B_c) * sizeof(float);
    const int matrix_size = batch_size * num_heads * seq_len * d * sizeof(float);
    const int vector_size = batch_size * num_heads * seq_len * sizeof(float);
    const int T_c = (seq_len + B_c - 1) / B_c;
    const int T_r = (seq_len + B_r - 1) / B_r;

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
    // fprintf(stderr, "shared memory: %d bytes\n", sram_size);
    parallel_flash_attn_kernel<<<grid_dim, block_dim, sram_size>>>(
        batch_size, num_heads, seq_len, d, Q_d, K_d, V_d, B_c, B_r, T_c, T_r, O_d, l_d, m_d);

    CUDA_CHECK(cudaGetLastError());

    cudaDeviceSynchronize();

    // copy results back to host
    cudaMemcpy(O_h, O_d, matrix_size, cudaMemcpyDeviceToHost);
    // cudaMemcpy(l_h, l_d, vector_size, cudaMemcpyDeviceToHost);
    // cudaMemcpy(m_h, m_d, vector_size, cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(K_d);
    cudaFree(V_d);
    cudaFree(Q_d);
    cudaFree(O_d);
    cudaFree(l_d);
    cudaFree(m_d);
}