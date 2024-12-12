#include <cuda.h>

using matrix = float *;

__global__ void serial_flash_attn_kernel(
    const int N,   // sequence length
    const int d,   // hidden dimension
    matrix Q,      // query matrix
    matrix K,      // key matrix
    matrix V,      // value matrix
    const int B_c, // column block size
    const int B_r, // row block size
    const int T_c, // number of column tiles
    const int T_r, // number of row tiles
    matrix O       // output matrix
)
{
    const float scale = 1.0f / sqrt(static_cast<float>(d));
    const int block_sz = d * B_c;

    // Allocate shared memory for tiles and intermediate results
    __shared__ float sram[4 * block_sz];
    matrix Q_i = sram;                // Query tile
    matrix K_j = sram + block_sz;     // Key tile
    matrix V_j = sram + 2 * block_sz; // Value tile
    matrix S = sram + 3 * block_sz;   // Score matrix

    for (int j = 0; j < T_c; j++)
    {
        // Load K_j and V_j tiles to shared memory
        for (int x = 0; x < block_sz; x++)
        {
            K_j[x] = K[x + block_sz * j];
            V_j[x] = V[x + block_sz * j];
        }
        for (int i = 0; i < T_r; i++)
        {
            // Load Q_i tile to shared memory
            for (int x = 0; x < block_sz; x++)
            {
                Q_i[x] = Q[x + block_sz * i];
            }

            // Compute attention scores (Q_i * K_j^T)
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

            // Apply softmax row-wise
            for (int q_idx = 0; q_idx < B_r; q_idx++)
            {
                // Find max for numerical stability
                float max_val = S[q_idx * B_c];
                for (int k_idx = 1; k_idx < B_c; k_idx++)
                {
                    max_val = max(max_val, S[q_idx * B_c + k_idx]);
                }

                // Compute exp and sum
                float sum = 0.0f;
                for (int k_idx = 0; k_idx < B_c; k_idx++)
                {
                    S[q_idx * B_c + k_idx] = __expf(S[q_idx * B_c + k_idx] - max_val);
                    sum += S[q_idx * B_c + k_idx];
                }

                // Normalize
                for (int k_idx = 0; k_idx < B_c; k_idx++)
                {
                    S[q_idx * B_c + k_idx] /= sum;
                }
            }

            // Compute output (S * V_j)
            for (int q_idx = 0; q_idx < B_r; q_idx++)
            {
                for (int h = 0; h < d; h++)
                {
                    float sum = 0.0f;
                    for (int k_idx = 0; k_idx < B_c; k_idx++)
                    {
                        sum += S[q_idx * B_c + k_idx] * V_j[k_idx * d + h];
                    }
                    // Accumulate to global memory
                    O[(i * B_r + q_idx) * d + h] += sum;
                }
            }
        }
    }
}

torch::Tensor forward(torch::Tensor Q, torch::Tensor K, torch::Tensor V, const int B_c, const int B_r, const int grid_dim_x, const int grid_dim_y, const int grid_dim_z, const int block_dim_x, const int block_dim_y, const int block_dim_z, const int sram_size)
{

    const int B = Q.size(0);
    const int nh = Q.size(1);
    const int N = Q.size(2);
    const int d = Q.size(3);

    const int Tc = ceil((float)N / B_c);
    const int Tr = ceil((float)N / B_r);
    const float softmax_scale = 1.0 / sqrt(d);

    // Initialize O, l, m to HBM
    auto O = torch::zeros_like(Q);
    auto l = torch::zeros({B, nh, N});
    auto m = torch::full({B, nh, N}, -INFINITY);
    torch::Device device(torch::kCUDA);
    l = l.to(device);
    m = m.to(device);

    // Calculate SRAM size needed per block
    const int sram_size = (3 * B_c * d * sizeof(float)) + (B_c * B_r * sizeof(float));
    int max_sram_size;
    cudaDeviceGetAttribute(&max_sram_size, cudaDevAttrMaxSharedMemoryPerBlock, 0);
    printf("Max shared memory: %d, requested shared memory: %d \\n", max_sram_size, sram_size);

    dim3 grid_dim(grid_dim_x, grid_dim_y, grid_dim_z);     // batch_size x num_heads
    dim3 block_dim(block_dim_x, block_dim_y, block_dim_z); // B_c threads per block

    forward_kernel<<<grid_dim, block_dim, sram_size>>>(
        Q.data_ptr<float>(), K.data_ptr<float>(), V.data_ptr<float>(),
        N, d, Tc, Tr, B_c, B_r, softmax_scale,
        l.data_ptr<float>(), m.data_ptr<float>(), O.data_ptr<float>());
    return O;
}