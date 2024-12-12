#include <torch/extension.h>

torch::Tensor forward(torch::Tensor q, torch::Tensor k, torch::Tensor v, const int B_c, const int B_r, const int grid_dim_x, const int grid_dim_y, const int grid_dim_z, const int block_dim_x, const int block_dim_y, const int block_dim_z, const int sram_size);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("forward", torch::wrap_pybind_function(forward), "forward");
}