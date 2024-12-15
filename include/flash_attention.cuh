#ifndef FLASH_ATTENTION_CUH
#define FLASH_ATTENTION_CUH

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
    const int block_dim_y // Inversely proportional to how many tiles a thread attends to
);

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
);
#endif