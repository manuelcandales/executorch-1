/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include "types.h"

namespace executorch {
namespace backends {
namespace aoti {

// Forward declarations to match PyTorch's AOTI MPS interface
struct AOTIMetalKernelFunctionOpaque;
using AOTIMetalKernelFunctionHandle = AOTIMetalKernelFunctionOpaque*;

struct AOTIMetalShaderLibraryOpaque;
using AOTIMetalShaderLibraryHandle = AOTIMetalShaderLibraryOpaque*;

// Match PyTorch's AtenTensorHandle definition
struct AtenTensorOpaque;
using AtenTensorHandle = AtenTensorOpaque*;

#ifdef __cplusplus
extern "C" {
#endif

// MetalShaderLibrary functions
AOTITorchError aoti_torch_mps_create_shader_library(
    const char* metal_shader_source,
    AOTIMetalShaderLibraryHandle* library_handle);

AOTITorchError aoti_torch_mps_delete_shader_library(
    AOTIMetalShaderLibraryHandle library_handle);

AOTITorchError aoti_torch_mps_get_kernel_function(
    AOTIMetalShaderLibraryHandle library_handle,
    const char* kernel_name,
    AOTIMetalKernelFunctionHandle* function_handle);

// MetalKernelFunction functions - individual call versions
AOTITorchError aoti_torch_mps_start_encoding(
    AOTIMetalKernelFunctionHandle func);

AOTITorchError aoti_torch_mps_set_arg_tensor(
    AOTIMetalKernelFunctionHandle func,
    unsigned idx,
    AtenTensorHandle tensor);

AOTITorchError aoti_torch_mps_set_arg_int(
    AOTIMetalKernelFunctionHandle func,
    unsigned idx,
    int64_t val);

// Pure C dispatch functions - single value versions
AOTITorchError aoti_torch_mps_dispatch_single(
    AOTIMetalKernelFunctionHandle func,
    uint64_t length);

AOTITorchError aoti_torch_mps_dispatch_single_with_group_size(
    AOTIMetalKernelFunctionHandle func,
    uint64_t length,
    uint64_t group_size);

// Pure C dispatch functions - array versions
AOTITorchError aoti_torch_mps_dispatch_array(
    AOTIMetalKernelFunctionHandle func,
    const uint64_t* length,
    size_t length_size);

AOTITorchError aoti_torch_mps_dispatch_array_with_group_size(
    AOTIMetalKernelFunctionHandle func,
    const uint64_t* length,
    size_t length_size,
    const uint64_t* group_size,
    size_t group_size_size);

// Memory management functions
AOTITorchError aoti_torch_mps_malloc(void** buffer, size_t num_bytes);

AOTITorchError aoti_torch_mps_free(void* ptr);

AOTITorchError aoti_torch_mps_memcpy(
    void* buffer,
    size_t constant_offset,
    size_t bytes_read,
    size_t data_size,
    uint8_t* constants_start);

AOTITorchError aoti_torch_mps_copy_buffer(
    void* src_buffer,
    void* dst_buffer,
    size_t data_size,
    size_t src_offset,
    size_t dst_offset);
// Stream management with PyTorch-style synchronization
AOTITorchError aoti_torch_mps_synchronize_stream();
AOTITorchError aoti_torch_mps_synchronize_stream_with_type(int sync_type);

#ifdef __cplusplus
} // extern "C"

// C++ only functions that can use std::function
#include <functional>

AOTITorchError aoti_torch_mps_run_command_block(
    AOTIMetalKernelFunctionHandle func,
    std::function<void(AOTIMetalKernelFunctionHandle)> command_block);

// Legacy operation-specific functions (preserved for backward compatibility)
AOTITorchError aoti_torch_mps_addmm_out(
    AtenTensorHandle out,
    AtenTensorHandle self,
    AtenTensorHandle mat1,
    AtenTensorHandle mat2,
    double beta,
    double alpha);

AOTITorchError aoti_torch_mps_mm_out(
    AtenTensorHandle out,
    AtenTensorHandle self,
    AtenTensorHandle mat2);

AOTITorchError aoti_torch_mps_convolution(
    AtenTensorHandle input,
    AtenTensorHandle weight,
    AtenTensorHandle* bias,
    const int64_t* stride,
    int64_t stride_len_,
    const int64_t* padding,
    int64_t padding_len_,
    const int64_t* dilation,
    int64_t dilation_len_,
    int32_t transposed,
    const int64_t* output_padding,
    int64_t output_padding_len_,
    int64_t groups,
    AtenTensorHandle* ret0);

#endif

} // namespace aoti
} // namespace backends
} // namespace executorch
