/*
 * @Author: Haozhe Xie
 * @Date:   2019-12-19 17:04:38
 * @Last Modified by:   Haozhe Xie
 * @Last Modified time: 2020-06-17 14:50:22
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

std::vector<torch::Tensor> cubic_feature_sampling_cuda_forward(
  int scale,
  int neighborhood_size,
  torch::Tensor ptcloud,
  torch::Tensor cubic_features,
  cudaStream_t stream);

std::vector<torch::Tensor> cubic_feature_sampling_cuda_backward(
  int scale,
  int neighborhood_size,
  torch::Tensor grad_point_features,
  torch::Tensor grid_pt_indexes,
  cudaStream_t stream);

std::vector<torch::Tensor> cubic_feature_sampling_forward(
  int scale,
  int neighborhood_size,
  torch::Tensor ptcloud,
  torch::Tensor cubic_features) {
  CHECK_INPUT(ptcloud);
  CHECK_INPUT(cubic_features);

  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  return cubic_feature_sampling_cuda_forward(scale, neighborhood_size, ptcloud,
                                             cubic_features, stream);
}

std::vector<torch::Tensor> cubic_feature_sampling_backward(
  int scale,
  int neighborhood_size,
  torch::Tensor grad_point_features,
  torch::Tensor grid_pt_indexes) {
  CHECK_INPUT(grad_point_features);
  CHECK_INPUT(grid_pt_indexes);

  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  return cubic_feature_sampling_cuda_backward(
    scale, neighborhood_size, grad_point_features, grid_pt_indexes, stream);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &cubic_feature_sampling_forward,
        "Cubic Feature Sampling forward (CUDA)");
  m.def("backward", &cubic_feature_sampling_backward,
        "Cubic Feature Sampling backward (CUDA)");
}