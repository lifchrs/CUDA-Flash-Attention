#ifndef FLASH_ATTENTION_CUH
#define FLASH_ATTENTION_CUH

void forward(
    float* Q,           // Query matrix
    float* K,           // Key matrix
    float* V,           // Value matrix
    float* O,           // Output buffer
    int batch_size,     // Batch size 
    int num_heads,      // Number of attention heads
    int seq_len,        // Sequence length
    int head_dim,       // Head dimension
    int B_c,           // Block size for columns
    int B_r,           // Block size for rows
    int T_c,           // Tile size for columns
    int T_r            // Tile size for rows
);


#endif