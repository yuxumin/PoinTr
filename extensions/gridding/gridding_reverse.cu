/*
 * @Author: Haozhe Xie
 * @Date:   2019-11-21 16:42:18
 * @Last Modified by:   Haozhe Xie
 * @Last Modified time: 2020-06-17 15:00:21
 * @Email:  cshzxie@gmail.com
 */

#include <bits/stdc++.h>
#include <torch/extension.h>
#include <cstdio>
#include <cstdlib>

#define CUDA_NUM_THREADS 512
#define EPS 1e-6

// Computer the number of threads needed in GPU
inline int get_n_threads(int n) {
  const int pow_2 = std::log(static_cast<float>(n)) / std::log(2.0);
  return max(min(1 << pow_2, CUDA_NUM_THREADS), 1);
}

__device__ int compute_index(int offset_x,
                             int offset_y,
                             int offset_z,
                             int scale) {
  return offset_x * scale * scale + offset_y * scale + offset_z;
}

__global__ void gridding_reverse_kernel(int scale,
                                        int n_pts,
                                        const float *__restrict__ grid,
                                        float *__restrict__ ptcloud) {
  int batch_index = blockIdx.x;
  int index       = threadIdx.x;
  int stride      = blockDim.x;

  ptcloud += batch_index * n_pts * 3;
  grid += batch_index * n_pts;

  for (int j = index; j < n_pts; j += stride) {
    int sqr_scale = scale * scale;
    int x_offset  = j / sqr_scale;
    int y_offset  = j % sqr_scale / scale;
    int z_offset  = j % sqr_scale % scale;
    if (x_offset == 0 || y_offset == 0 || z_offset == 0) {
      continue;
    }

    // assert j == compute_index(x_offset, y_offset, z_offset, scale)
    float weights[8] = {
      grid[compute_index(x_offset - 1, y_offset - 1, z_offset - 1, scale)],
      grid[compute_index(x_offset - 1, y_offset - 1, z_offset, scale)],
      grid[compute_index(x_offset - 1, y_offset, z_offset - 1, scale)],
      grid[compute_index(x_offset - 1, y_offset, z_offset, scale)],
      grid[compute_index(x_offset, y_offset - 1, z_offset - 1, scale)],
      grid[compute_index(x_offset, y_offset - 1, z_offset, scale)],
      grid[compute_index(x_offset, y_offset, z_offset - 1, scale)],
      grid[j]};

    float weights_sum = 0;
    for (size_t i = 0; i < 8; ++i) {
      weights_sum += weights[i];
    }
    if (weights_sum < EPS) {
      continue;
    }
    for (size_t i = 0; i < 8; ++i) {
      weights[i] /= weights_sum;
    }

    x_offset -= scale / 2;
    y_offset -= scale / 2;
    z_offset -= scale / 2;

    // clang-format off
    ptcloud[j * 3 + 0] = weights[0] * (x_offset - 1) +
                         weights[1] * (x_offset - 1) +
                         weights[2] * (x_offset - 1) +
                         weights[3] * (x_offset - 1) +
                         weights[4] * x_offset +
                         weights[5] * x_offset +
                         weights[6] * x_offset +
                         weights[7] * x_offset;
    ptcloud[j * 3 + 1] = weights[0] * (y_offset - 1) +
                         weights[1] * (y_offset - 1) +
                         weights[2] * y_offset +
                         weights[3] * y_offset +
                         weights[4] * (y_offset - 1) +
                         weights[5] * (y_offset - 1) +
                         weights[6] * y_offset +
                         weights[7] * y_offset;
    ptcloud[j * 3 + 2] = weights[0] * (z_offset - 1) +
                         weights[1] * z_offset +
                         weights[2] * (z_offset - 1) +
                         weights[3] * z_offset +
                         weights[4] * (z_offset - 1) +
                         weights[5] * z_offset +
                         weights[6] * (z_offset - 1) +
                         weights[7] * z_offset;
    // clang-format on
  }
}

torch::Tensor gridding_reverse_cuda_forward(int scale,
                                            torch::Tensor grid,
                                            cudaStream_t stream) {
  int batch_size = grid.size(0);
  int n_pts      = scale * scale * scale;

  torch::Tensor ptcloud =
    torch::zeros({batch_size, n_pts, 3}, torch::CUDA(torch::kFloat));

  gridding_reverse_kernel<<<batch_size, get_n_threads(n_pts), 0, stream>>>(
    scale, n_pts, grid.data_ptr<float>(), ptcloud.data_ptr<float>());

  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    printf("Error in gridding_cuda_forward: %s\n", cudaGetErrorString(err));
  }
  return ptcloud;
}

__global__ void gridding_reverse_grad_kernel(
  int scale,
  int n_pts,
  const float *__restrict__ ptcloud,
  const float *__restrict__ grid,
  const float *__restrict__ grad_ptcloud,
  float *__restrict__ grad_grid) {
  int batch_index = blockIdx.x;
  int index       = threadIdx.x;
  int stride      = blockDim.x;

  ptcloud += batch_index * n_pts * 3;
  grid += batch_index * n_pts;
  grad_ptcloud += batch_index * n_pts * 3;
  grad_grid += batch_index * n_pts;

  for (int j = index; j < n_pts; j += stride) {
    int sqr_scale = scale * scale;
    int x_offset  = j / sqr_scale;
    int y_offset  = j % sqr_scale / scale;
    int z_offset  = j % sqr_scale % scale;
    if (x_offset == 0 || y_offset == 0 || z_offset == 0) {
      continue;
    }

    int gvtx_indexes[8] = {
      compute_index(x_offset - 1, y_offset - 1, z_offset - 1, scale),
      compute_index(x_offset - 1, y_offset - 1, z_offset, scale),
      compute_index(x_offset - 1, y_offset, z_offset - 1, scale),
      compute_index(x_offset - 1, y_offset, z_offset, scale),
      compute_index(x_offset, y_offset - 1, z_offset - 1, scale),
      compute_index(x_offset, y_offset - 1, z_offset, scale),
      compute_index(x_offset, y_offset, z_offset - 1, scale),
      j};
    float weights[8] = {grid[gvtx_indexes[0]], grid[gvtx_indexes[1]],
                        grid[gvtx_indexes[2]], grid[gvtx_indexes[3]],
                        grid[gvtx_indexes[4]], grid[gvtx_indexes[5]],
                        grid[gvtx_indexes[6]], grid[gvtx_indexes[7]]};

    float weights_sum = 0;
    for (size_t i = 0; i < 8; ++i) {
      weights_sum += weights[i];
    }

    if (weights_sum < EPS) {
      continue;
    }
    for (size_t i = 0; i < 8; ++i) {
      weights[i] /= weights_sum;
    }

    x_offset -= scale / 2;
    y_offset -= scale / 2;
    z_offset -= scale / 2;

    // clang-format off
    atomicAdd(&(grad_grid[gvtx_indexes[0]]),
                grad_ptcloud[j * 3 + 0] * ((x_offset - 1) - ptcloud[j * 3 + 0]) / weights_sum +
                grad_ptcloud[j * 3 + 1] * ((y_offset - 1) - ptcloud[j * 3 + 1]) / weights_sum +
                grad_ptcloud[j * 3 + 2] * ((z_offset - 1) - ptcloud[j * 3 + 2]) / weights_sum);
    atomicAdd(&(grad_grid[gvtx_indexes[1]]),
                grad_ptcloud[j * 3 + 0] * ((x_offset - 1) - ptcloud[j * 3 + 0]) / weights_sum +
                grad_ptcloud[j * 3 + 1] * ((y_offset - 1) - ptcloud[j * 3 + 1]) / weights_sum +
                grad_ptcloud[j * 3 + 2] * (z_offset - ptcloud[j * 3 + 2]) / weights_sum);
    atomicAdd(&(grad_grid[gvtx_indexes[2]]),
                grad_ptcloud[j * 3 + 0] * ((x_offset - 1) - ptcloud[j * 3 + 0]) / weights_sum +
                grad_ptcloud[j * 3 + 1] * (y_offset - ptcloud[j * 3 + 1]) / weights_sum +
                grad_ptcloud[j * 3 + 2] * ((z_offset - 1) - ptcloud[j * 3 + 2]) / weights_sum);
    atomicAdd(&(grad_grid[gvtx_indexes[3]]),
                grad_ptcloud[j * 3 + 0] * ((x_offset - 1) - ptcloud[j * 3 + 0]) / weights_sum +
                grad_ptcloud[j * 3 + 1] * (y_offset - ptcloud[j * 3 + 1]) / weights_sum +
                grad_ptcloud[j * 3 + 2] * (z_offset - ptcloud[j * 3 + 2]) / weights_sum);
    atomicAdd(&(grad_grid[gvtx_indexes[4]]),
                grad_ptcloud[j * 3 + 0] * (x_offset - ptcloud[j * 3 + 0]) / weights_sum +
                grad_ptcloud[j * 3 + 1] * ((y_offset - 1) - ptcloud[j * 3 + 1]) / weights_sum +
                grad_ptcloud[j * 3 + 2] * ((z_offset - 1) - ptcloud[j * 3 + 2]) / weights_sum);
    atomicAdd(&(grad_grid[gvtx_indexes[5]]),
                grad_ptcloud[j * 3 + 0] * (x_offset - ptcloud[j * 3 + 0]) / weights_sum +
                grad_ptcloud[j * 3 + 1] * ((y_offset - 1) - ptcloud[j * 3 + 1]) / weights_sum +
                grad_ptcloud[j * 3 + 2] * (z_offset - ptcloud[j * 3 + 2]) / weights_sum);
    atomicAdd(&(grad_grid[gvtx_indexes[6]]),
                grad_ptcloud[j * 3 + 0] * (x_offset - ptcloud[j * 3 + 0]) / weights_sum +
                grad_ptcloud[j * 3 + 1] * (y_offset - ptcloud[j * 3 + 1]) / weights_sum +
                grad_ptcloud[j * 3 + 2] * ((z_offset - 1) - ptcloud[j * 3 + 2]) / weights_sum);
    atomicAdd(&(grad_grid[gvtx_indexes[7]]),
                grad_ptcloud[j * 3 + 0] * (x_offset - ptcloud[j * 3 + 0]) / weights_sum +
                grad_ptcloud[j * 3 + 1] * (y_offset - ptcloud[j * 3 + 1]) / weights_sum +
                grad_ptcloud[j * 3 + 2] * (z_offset - ptcloud[j * 3 + 2]) / weights_sum);
    // clang-format on
  }
}

torch::Tensor gridding_reverse_cuda_backward(torch::Tensor ptcloud,
                                             torch::Tensor grid,
                                             torch::Tensor grad_ptcloud,
                                             cudaStream_t stream) {
  int batch_size = ptcloud.size(0);
  int n_pts      = ptcloud.size(1);
  int scale      = static_cast<int>(std::cbrt(n_pts));

  torch::Tensor grad_grid =
    torch::zeros({batch_size, n_pts}, torch::CUDA(torch::kFloat));

  gridding_reverse_grad_kernel<<<batch_size, get_n_threads(n_pts), 0, stream>>>(
    scale, n_pts, ptcloud.data_ptr<float>(), grid.data_ptr<float>(),
    grad_ptcloud.data_ptr<float>(), grad_grid.data_ptr<float>());

  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    printf("Error in gridding_cuda_forward: %s\n", cudaGetErrorString(err));
  }
  return grad_grid;
}