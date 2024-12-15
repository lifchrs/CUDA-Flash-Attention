#ifndef FLASH_ATTENTION_CUH
#define FLASH_ATTENTION_CUH

void forward_serial(
    float* Q,             // Query matrix
    float* K,             // Key matrix
    float* V,             // Value matrix
    float* O,             // Output matrix
    float* m,             // Running maximum values
    float* l,             // Running sum of exponentials
    const int B_c,        // Column block size
    const int B_r,        // Row block size
    const int num_heads,  // Number of heads
    const int batch_size, // Batch size
    const int seq_len,    // Sequence Length
    const int d          // Hidden dimension
);

void forward_parallel(
    float* Q,             // Query matrix
    float* K,             // Key matrix
    float* V,             // Value matrix
    float* O,             // Output matrix
    float* m,             // Running maximum values
    float* l,             // Running sum of exponentials
    const int B_c,        // Column block size
    const int B_r,        // Row block size
    const int num_heads,  // Number of heads
    const int batch_size, // Batch size
    const int seq_len,    // Sequence Length
    const int d          // Hidden dimension
);
#endif