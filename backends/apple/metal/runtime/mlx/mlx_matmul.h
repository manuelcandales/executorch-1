/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

/*
 * MLX Metal Matmul Integration for ExecuTorch
 * 
 * This file adapts MLX's optimized Metal matrix multiplication kernels
 * to work with ExecuTorch's Metal backend infrastructure.
 * 
 * Original MLX code: Copyright © 2023-2024 Apple Inc.
 */

#pragma once

#include <executorch/runtime/core/exec_aten/exec_aten.h>
#include <executorch/backends/apple/metal/runtime/shims/et_metal.h>

namespace executorch {
namespace backends {
namespace metal {
namespace mlx {

using executorch::runtime::etensor::Tensor;

// MLX GEMM Parameters (adapted from mlx/backend/metal/kernels/steel/gemm/params.h)
struct GEMMParams {
  const int M;
  const int N;
  const int K;

  const int lda;
  const int ldb;
  const int ldd;

  const int tiles_n;
  const int tiles_m;

  const int64_t batch_stride_a;
  const int64_t batch_stride_b;
  const int64_t batch_stride_d;

  const int swizzle_log;
  const int gemm_k_iterations_aligned;

  const int batch_ndim;
};

struct GEMMSpiltKParams {
  const int M;
  const int N;
  const int K;

  const int lda;
  const int ldb;
  const int ldc;

  const int tiles_n;
  const int tiles_m;

  const int split_k_partitions;
  const int split_k_partition_stride;
  const int split_k_partition_size;

  const int gemm_k_iterations_aligned;
};

/**
 * MLX-based matrix multiplication implementation for ExecuTorch
 * 
 * This function uses MLX's optimized Metal kernels for matrix multiplication,
 * providing better performance than MPS in many cases.
 * 
 * @param out Output tensor (M x N)
 * @param a Input tensor A (M x K)
 * @param b Input tensor B (K x N)
 * @param stream ExecuTorch Metal stream for command encoding
 * @return true if successful, false otherwise
 */
bool metal_mm_out(
    Tensor* out,
    const Tensor* a,
    const Tensor* b,
    ETMetalStream* stream);

/**
 * Initialize MLX matmul kernels
 * Compiles and caches the Metal shader library
 */
void init_mlx_matmul_kernels();

} // namespace mlx
} // namespace metal
} // namespace backends
} // namespace executorch

