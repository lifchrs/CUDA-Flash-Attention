#ifndef FLASH_ATTENTION_CUH
#define FLASH_ATTENTION_CUH

void forward(
    float* Q,          // Query matrix
    float* K,          // Key matrix
    float* V,          // Value matrix
    const int B_c,     // Column block size
    const int B_r,     // Row block size
    const int grid_dim_x,  // Grid dimension x
    const int grid_dim_y,  // Grid dimension y
    const int grid_dim_z,  // Grid dimension z
    const int block_dim_x, // Block dimension x
    const int block_dim_y, // Block dimension y
    const int block_dim_z, // Block dimension z
    const int N,          // Sequence length
    const int d,          // Hidden dimension
    const int T_c,        // Number of column tiles
    const int T_r,        // Number of row tiles
    float* O,          // Output matrix
    float* l,          // Running sum of exponentials
    float* m           // Running maximum values
);

#endif