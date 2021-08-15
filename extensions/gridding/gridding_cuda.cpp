/*
 * @Author: Haozhe Xie
 * @Date:   2019-11-13 10:52:53
 * @Last Modified by:   Haozhe Xie
 * @Last Modified time: 2020-06-17 14:52:32
 * @Email:  cshzxie@gmail.com
 */

#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>

// NOTE: AT_ASSERT has become AT_CHECK on master after 0.4.
#define CHECK_CUDA(x) AT_ASSERTM(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) \
  AT_ASSERTM(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) \
  CHECK_CUDA(x);       \
  CHECK_CONTIGUOUS(x)

std::vector<torch::Tensor> gridding_cuda_forward(float min_x,
                                                 float max_x,
                                                 float min_y,
                                                 float max_y,
                                                 float min_z,
                                                 float max_z,
                                                 torch::Tensor ptcloud,
                                                 cudaStream_t stream);

torch::Tensor gridding_cuda_backward(torch::Tensor grid_pt_weights,
                                     torch::Tensor grid_pt_indexes,
                                     torch::Tensor grad_grid,
                                     cudaStream_t stream);

torch::Tensor gridding_reverse_cuda_forward(int scale,
                                            torch::Tensor grid,
                                            cudaStream_t stream);

torch::Tensor gridding_reverse_cuda_backward(torch::Tensor ptcloud,
                                             torch::Tensor grid,
                                             torch::Tensor grad_ptcloud,
                                             cudaStream_t stream);

std::vector<torch::Tensor> gridding_forward(float min_x,
                                            float max_x,
                                            float min_y,
                                            float max_y,
                                            float min_z,
                                            float max_z,
                                            torch::Tensor ptcloud) {
  CHECK_INPUT(ptcloud);

  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  return gridding_cuda_forward(min_x, max_x, min_y, max_y, min_z, max_z,
                               ptcloud, stream);
}

torch::Tensor gridding_backward(torch::Tensor grid_pt_weights,
                                torch::Tensor grid_pt_indexes,
                                torch::Tensor grad_grid) {
  CHECK_INPUT(grid_pt_weights);
  CHECK_INPUT(grid_pt_indexes);
  CHECK_INPUT(grad_grid);

  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  return gridding_cuda_backward(grid_pt_weights, grid_pt_indexes, grad_grid,
                                stream);
}

torch::Tensor gridding_reverse_forward(int scale, torch::Tensor grid) {
  CHECK_INPUT(grid);

  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  return gridding_reverse_cuda_forward(scale, grid, stream);
}

torch::Tensor gridding_reverse_backward(torch::Tensor ptcloud,
                                        torch::Tensor grid,
                                        torch::Tensor grad_ptcloud) {
  CHECK_INPUT(ptcloud);
  CHECK_INPUT(grid);
  CHECK_INPUT(grad_ptcloud);

  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  return gridding_reverse_cuda_backward(ptcloud, grid, grad_ptcloud, stream);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &gridding_forward, "Gridding forward (CUDA)");
  m.def("backward", &gridding_backward, "Gridding backward (CUDA)");
  m.def("rev_forward", &gridding_reverse_forward,
        "Gridding Reverse forward (CUDA)");
  m.def("rev_backward", &gridding_reverse_backward,
        "Gridding Reverse backward (CUDA)");
}
