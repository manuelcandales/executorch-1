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
 * Adapted from: mlx/backend/metal/matmul.cpp
 */

#import <Metal/Metal.h>
#import <Foundation/Foundation.h>
#include <sstream>
#include <algorithm>
#include <executorch/runtime/platform/log.h>
#include <executorch/backends/apple/metal/runtime/mlx/mlx_matmul.h>
#include <executorch/backends/apple/metal/runtime/shims/et_metal.h>

namespace executorch {
namespace backends {
namespace metal {
namespace mlx {

using executorch::runtime::etensor::Tensor;

// Global shader library for MLX kernels
static std::unique_ptr<ETMetalShaderLibrary> g_mlx_shader_library;

// Forward declarations for helper functions
static std::string get_mlx_kernel_source();
static std::string type_to_name(const Tensor* t);
static bool check_transpose(const Tensor* t, int64_t& leading_dim);
static MTL::Size get_block_dims(int x, int y, int z);

// FIXME: MLX uses a sophisticated device architecture detection system
// For now, we'll use simple heuristics based on problem size
static char get_device_arch() {
  // TODO: Implement proper device architecture detection
  // 'd' = large device (M1 Max, M2 Max, etc.)
  // 'c' = medium device (M1, M2, M3)
  // 'g'/'p' = small device (A-series)
  return 'c'; // Default to medium for now
}

/**
 * Get Metal type name string from ExecuTorch tensor
 * Adapted from MLX's type_to_name function
 */
static std::string type_to_name(const Tensor* t) {
  auto dtype = static_cast<int32_t>(t->scalar_type());
  // FIXME: Need proper dtype mapping from ExecuTorch to Metal type names
  // For now, support float32 and bfloat16
  if (dtype == 0) { // Assume 0 is float32
    return "float32";
  } else if (dtype == 15) { // Assume 15 is bfloat16  
    return "bfloat16";
  }
  return "float32"; // Default fallback
}

/**
 * Check if tensor is transposed and get leading dimension
 * Adapted from MLX's check_transpose function
 */
static bool check_transpose(const Tensor* t, int64_t& leading_dim) {
  auto ndim = t->dim();
  auto stx = t->strides()[ndim - 2];
  auto sty = t->strides()[ndim - 1];
  
  if (sty == 1) {
    // Row-major (contiguous)
    leading_dim = stx;
    return false;
  } else if (stx == 1) {
    // Column-major (transposed)
    leading_dim = sty;
    return true;
  } else {
    // Non-contiguous - would need copy in full implementation
    // FIXME: For now, assume row-major and use shape-based leading dim
    ET_LOG(Warning, "MLX matmul: Non-contiguous tensor detected, assuming row-major layout");
    leading_dim = t->sizes()[ndim - 1];
    return false;
  }
}

/**
 * Get thread group dimensions
 * Helper for kernel dispatch
 */
static MTL::Size get_block_dims(int x, int y, int z) {
  // Simple heuristic for now
  // TODO: Implement MLX's more sophisticated block size selection
  int bx = std::min(32, x);
  int by = std::min(32, y);
  int bz = std::min(1, z);
  return MTL::Size(bx, by, bz);
}

/**
 * Steel GEMM kernel dispatch
 * Adapted from MLX's steel_matmul_regular_axpby function
 * 
 * This implements MLX's optimized GEMM using "Steel" kernels
 * FIXME: Simplified version - full MLX implementation has many more optimizations
 */
static void steel_matmul_regular(
    ETMetalStream* stream,
    const Tensor* a,
    const Tensor* b,
    Tensor* out,
    int M,
    int N,
    int K,
    int batch_size_out,
    int64_t lda,
    int64_t ldb,
    bool transpose_a,
    bool transpose_b,
    int64_t A_batch_stride,
    int64_t B_batch_stride) {
  
  @autoreleasepool {
    // Determine dispatch kernel parameters
    // Adapted from MLX's GEMM_TPARAM_MACRO
    int bm = 64, bn = 64, bk = 16;
    int wm = 2, wn = 2;
    
    char devc = get_device_arch();
    
    // FIXME: Simplified heuristics - MLX has complex device-specific tuning
    if (devc == 'd') {
      // Large device
      size_t problem_size = static_cast<size_t>(batch_size_out) * M * N;
      if (problem_size >= (1ul << 20)) {
        if (2 * std::max(M, N) > K) {
          bm = 64; bn = 64; bk = 16;
          wm = 1; wn = 2;
        } else if (!transpose_a && transpose_b) {
          bm = 64; bn = 32; bk = 32;
          wm = 2; wn = 2;
        }
      }
    } else {
      // Medium/small device
      bm = 64; bn = 64; bk = 16;
      wm = 2; wn = 2;
    }
    
    // Prepare kernel name
    std::ostringstream kname;
    kname << "steel_gemm_fused_"
          << (transpose_a ? 't' : 'n')
          << (transpose_b ? 't' : 'n')
          << "_" << type_to_name(a)
          << "_" << type_to_name(out)
          << "_bm" << bm << "_bn" << bn << "_bk" << bk
          << "_wm" << wm << "_wn" << wn;
    
    std::string base_name = kname.str();
    
    const bool has_batch = (batch_size_out > 1);
    const bool align_M = (M % bm) == 0;
    const bool align_N = (N % bn) == 0;
    const bool align_K = (K % bk) == 0;
    
    // FIXME: MLX uses function constants for specialization
    // ExecuTorch doesn't have direct equivalent - using full kernel name instead
    kname << "_has_batch_" << (has_batch ? 't' : 'f')
          << "_use_out_source_" << 'f'
          << "_do_axpby_" << 'f'
          << "_align_M_" << (align_M ? 't' : 'f')
          << "_align_N_" << (align_N ? 't' : 'f')
          << "_align_K_" << (align_K ? 't' : 'f');
    
    std::string hash_name = kname.str();
    
    ET_LOG(Debug, "MLX matmul: Using kernel %s", hash_name.c_str());
    
    // Get kernel function
    // FIXME: Need to compile MLX kernels into ExecuTorch shader library
    // For now, this will fail - need to add kernel compilation support
    auto kernel_func = g_mlx_shader_library->getKernelFunction(hash_name);
    if (!kernel_func) {
      ET_LOG(Error, "MLX matmul: Failed to get kernel function %s", hash_name.c_str());
      throw std::runtime_error("Failed to get MLX matmul kernel");
    }
    
    // Get Metal buffers from tensor pointers
    // FIXME: Need proper buffer resolution from ExecuTorch tensor system
    id<MTLBuffer> a_buffer = nil;
    id<MTLBuffer> b_buffer = nil;
    id<MTLBuffer> out_buffer = nil;
    
    // TODO: Get buffers from ptr_to_mtl_buffer map
    extern std::unordered_map<void*, id<MTLBuffer>> ptr_to_mtl_buffer;
    
    auto a_it = ptr_to_mtl_buffer.find(const_cast<void*>(a->const_data_ptr()));
    auto b_it = ptr_to_mtl_buffer.find(const_cast<void*>(b->const_data_ptr()));
    auto out_it = ptr_to_mtl_buffer.find(out->mutable_data_ptr());
    
    if (a_it == ptr_to_mtl_buffer.end() ||
        b_it == ptr_to_mtl_buffer.end() ||
        out_it == ptr_to_mtl_buffer.end()) {
      ET_LOG(Error, "MLX matmul: Failed to find Metal buffers for tensors");
      throw std::runtime_error("Metal buffer lookup failed");
    }
    
    a_buffer = a_it->second;
    b_buffer = b_it->second;
    out_buffer = out_it->second;
    
    // Prepare GEMM params
    int tn = (N + bn - 1) / bn;
    int tm = (M + bm - 1) / bm;
    int swizzle_log = 0; // FIXME: MLX has swizzle tuning
    
    GEMMParams params{
      /* M = */ M,
      /* N = */ N,
      /* K = */ K,
      /* lda = */ static_cast<int>(lda),
      /* ldb = */ static_cast<int>(ldb),
      /* ldd = */ N,
      /* tiles_n = */ tn,
      /* tiles_m = */ tm,
      /* batch_stride_a = */ A_batch_stride,
      /* batch_stride_b = */ B_batch_stride,
      /* batch_stride_d = */ static_cast<int64_t>(M) * N,
      /* swizzle_log = */ swizzle_log,
      /* gemm_k_iterations_aligned = */ (K / bk),
      /* batch_ndim = */ (has_batch ? 1 : 0)
    };
    
    // Prepare grid
    int tile = 1 << swizzle_log;
    tm = (tm + tile - 1) / tile;
    tn = tn * tile;
    
    MTL::Size group_dims = MTL::Size(32, wn, wm);
    MTL::Size grid_dims = MTL::Size(tn, tm, batch_size_out);
    
    // Encode kernel
    kernel_func->startEncoding();
    
    // FIXME: MLX uses CommandEncoder::set_input_array which handles buffer+offset
    // ExecuTorch's setArg is simpler - need to adapt
    auto enc = stream->commandEncoder();
    [enc setBuffer:a_buffer offset:0 atIndex:0];
    [enc setBuffer:b_buffer offset:0 atIndex:1];
    [enc setBuffer:out_buffer offset:0 atIndex:3];
    [enc setBytes:&params length:sizeof(GEMMParams) atIndex:4];
    
    // FIXME: MLX also sets batch shape/strides for batched operations
    // Skipping for now - would need vectors
    
    // Dispatch
    [enc dispatchThreadgroups:MTLSizeMake(grid_dims.width, grid_dims.height, grid_dims.depth)
        threadsPerThreadgroup:MTLSizeMake(group_dims.width, group_dims.height, group_dims.depth)];
    
    ET_LOG(Debug, "MLX matmul: Dispatched kernel with grid [%lu,%lu,%lu] group [%lu,%lu,%lu]",
           grid_dims.width, grid_dims.height, grid_dims.depth,
           group_dims.width, group_dims.height, group_dims.depth);
  }
}

/**
 * Main MLX matmul entry point
 * Adapted from MLX's Matmul::eval_gpu
 */
bool metal_mm_out(
    Tensor* out,
    const Tensor* a,
    const Tensor* b,
    ETMetalStream* stream) {
  
  ET_LOG(Debug, "MLX matmul: Starting with a=%p, b=%p, out=%p",
         a, b, out);
  
  try {
    // Validate inputs
    if (!a || !b || !out || !stream) {
      ET_LOG(Error, "MLX matmul: null inputs");
      return false;
    }
    
    // FIXME: MLX handles empty inputs - skipping for now
    if (a->numel() == 0 || b->numel() == 0) {
      ET_LOG(Error, "MLX matmul: empty input tensors not supported yet");
      return false;
    }
    
    // Get dimensions
    int M = a->sizes()[a->dim() - 2];
    int N = b->sizes()[b->dim() - 1];
    int K = a->sizes()[a->dim() - 1];
    
    ET_LOG(Debug, "MLX matmul: M=%d, N=%d, K=%d", M, N, K);
    
    // Check transpose and get leading dimensions
    int64_t lda, ldb;
    bool a_transposed = check_transpose(a, lda);
    bool b_transposed = check_transpose(b, ldb);
    
    ET_LOG(Debug, "MLX matmul: a_transposed=%d, b_transposed=%d, lda=%lld, ldb=%lld",
           a_transposed, b_transposed, lda, ldb);
    
    // Determine batch size
    // FIXME: MLX has sophisticated batch dimension collapsing
    // For now, support only simple 2D matmul
    int batch_size_out = 1;
    if (a->dim() > 2 || b->dim() > 2) {
      ET_LOG(Warning, "MLX matmul: Batched matmul not fully supported yet, treating as unbatched");
      // Could calculate batch_size_out = out->numel() / (M * N)
      // but would need proper batch stride calculation
    }
    
    // FIXME: MLX has GEMV specialization for matrix-vector cases
    // Skipping for now - always use GEMM
    if (M == 1 || N == 1) {
      ET_LOG(Warning, "MLX matmul: GEMV specialization not implemented, using GEMM");
    }
    
    // Dispatch to steel matmul
    steel_matmul_regular(
        stream,
        a,
        b,
        out,
        M,
        N,
        K,
        batch_size_out,
        lda,
        ldb,
        a_transposed,
        b_transposed,
        M * K,  // A_batch_stride (FIXME: simplified)
        K * N); // B_batch_stride (FIXME: simplified)
    
    ET_LOG(Debug, "MLX matmul: Completed successfully");
    return true;
    
  } catch (const std::exception& e) {
    ET_LOG(Error, "MLX matmul exception: %s", e.what());
    return false;
  } catch (...) {
    ET_LOG(Error, "MLX matmul: unknown exception");
    return false;
  }
}

/**
 * Get MLX kernel source code
 * FIXME: For now, return placeholder
 * In full implementation, would embed or load MLX .metal files
 */
static std::string get_mlx_kernel_source() {
  // TODO: Embed or load the actual MLX kernel source
  // This would include:
  // - steel/gemm/kernels/steel_gemm_fused.metal
  // - steel/gemm/kernels/steel_gemm_splitk.metal
  // - gemv.metal (for matrix-vector cases)
  // - All necessary headers
  
  std::ostringstream source;
  source << "// MLX Metal Kernels\n";
  source << "// FIXME: Placeholder - need to embed actual kernel source\n";
  source << "#include <metal_stdlib>\n";
  source << "using namespace metal;\n";
  
  // In a real implementation, we would read the .metal files and concatenate them
  // For now, return minimal placeholder
  return source.str();
}

/**
 * Initialize MLX matmul kernels
 * Compiles the Metal shader library
 */
void init_mlx_matmul_kernels() {
  ET_LOG(Info, "Initializing MLX matmul kernels");
  
  try {
    // FIXME: Need to properly load and compile MLX kernel source
    // For now, create placeholder library
    // In full implementation:
    // 1. Load all .metal files from mlx/kernels/ directory
    // 2. Concatenate with proper includes
    // 3. Compile with ExecuTorch's shader library
    
    std::string kernel_source = get_mlx_kernel_source();
    g_mlx_shader_library = std::make_unique<ETMetalShaderLibrary>(kernel_source);
    
    ET_LOG(Info, "MLX matmul kernels initialized");
    
  } catch (const std::exception& e) {
    ET_LOG(Error, "Failed to initialize MLX matmul kernels: %s", e.what());
    throw;
  }
}

} // namespace mlx
} // namespace metal
} // namespace backends
} // namespace executorch

