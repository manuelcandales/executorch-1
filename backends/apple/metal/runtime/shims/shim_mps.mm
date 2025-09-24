/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#import <Metal/Metal.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>
#import <Foundation/Foundation.h>
#include <executorch/runtime/platform/log.h>
#include <executorch/runtime/core/exec_aten/exec_aten.h>
#include "shim_mps.h"
#include "et_metal.h"
#include "utils.h"
#include "memory.h"
#include <functional>

namespace executorch {
namespace backends {
namespace aoti {

extern "C" {

// MetalShaderLibrary functions
AOTITorchError aoti_torch_mps_create_shader_library(
    const char* metal_shader_source,
    AOTIMetalShaderLibraryHandle* library_handle) {

    if (!metal_shader_source || !library_handle) {
        ET_LOG(Error, "aoti_torch_mps_create_shader_library: null arguments");
        return Error::InvalidArgument;
    }

    @autoreleasepool {
        try {
            auto library = std::make_unique<ETMetalShaderLibrary>(std::string(metal_shader_source));
            auto* raw_library = library.get();

            // Store the unique_ptr to keep the object alive
            storeLibraryHandle(raw_library, std::move(library));

            // Return raw pointer to match existing API
            *library_handle = reinterpret_cast<AOTIMetalShaderLibraryHandle>(raw_library);

            ET_LOG(Debug, "aoti_torch_mps_create_shader_library: Created shader library %p", raw_library);
            return Error::Ok;

        } catch (const std::exception& e) {
            ET_LOG(Error, "aoti_torch_mps_create_shader_library exception: %s", e.what());
            return Error::Internal;
        } catch (...) {
            ET_LOG(Error, "aoti_torch_mps_create_shader_library: unknown exception");
            return Error::Internal;
        }
    }
}

AOTITorchError aoti_torch_mps_delete_shader_library(
    AOTIMetalShaderLibraryHandle library_handle) {

    if (!library_handle) {
        ET_LOG(Error, "aoti_torch_mps_delete_shader_library: null library handle");
        return Error::InvalidArgument;
    }

    try {
        auto* library = reinterpret_cast<ETMetalShaderLibrary*>(library_handle);
        if (removeLibraryHandle(library)) {
            ET_LOG(Debug, "aoti_torch_mps_delete_shader_library: Deleted shader library %p", library);
        } else {
            ET_LOG(Error, "aoti_torch_mps_delete_shader_library: Library not found in storage");
            return Error::InvalidArgument;
        }

        return Error::Ok;

    } catch (const std::exception& e) {
        ET_LOG(Error, "aoti_torch_mps_delete_shader_library exception: %s", e.what());
        return Error::Internal;
    } catch (...) {
        ET_LOG(Error, "aoti_torch_mps_delete_shader_library: unknown exception");
        return Error::Internal;
    }
}

AOTITorchError aoti_torch_mps_get_kernel_function(
    AOTIMetalShaderLibraryHandle library_handle,
    const char* kernel_name,
    AOTIMetalKernelFunctionHandle* function_handle) {

    if (!library_handle || !kernel_name || !function_handle) {
        ET_LOG(Error, "aoti_torch_mps_get_kernel_function: null arguments");
        return Error::InvalidArgument;
    }

    try {
        auto* library = reinterpret_cast<ETMetalShaderLibrary*>(library_handle);
        auto function_shared_ptr = library->getKernelFunction(std::string(kernel_name));
        if (!function_shared_ptr) {
            ET_LOG(Error, "aoti_torch_mps_get_kernel_function: Failed to get kernel function '%s'", kernel_name);
            return Error::Internal;
        }

        auto* raw_function = function_shared_ptr.get();

        // Store the shared_ptr to keep the object alive
        storeFunctionHandle(raw_function, function_shared_ptr);

        // Return raw pointer to match existing API
        *function_handle = reinterpret_cast<AOTIMetalKernelFunctionHandle>(raw_function);

        ET_LOG(Debug, "aoti_torch_mps_get_kernel_function: Got kernel function '%s' -> %p", kernel_name, raw_function);
        return Error::Ok;

    } catch (const std::exception& e) {
        ET_LOG(Error, "aoti_torch_mps_get_kernel_function exception: %s", e.what());
        return Error::Internal;
    } catch (...) {
        ET_LOG(Error, "aoti_torch_mps_get_kernel_function: unknown exception");
        return Error::Internal;
    }
}

AOTITorchError aoti_torch_mps_start_encoding(
    AOTIMetalKernelFunctionHandle func) {

    if (!func) {
        ET_LOG(Error, "aoti_torch_mps_start_encoding: null function handle");
        return Error::InvalidArgument;
    }

    try {
        auto* function = reinterpret_cast<ETMetalKernelFunction*>(func);
        function->startEncoding();

        ET_LOG(Debug, "aoti_torch_mps_start_encoding: Started encoding for function %p", function);
        return Error::Ok;

    } catch (const std::exception& e) {
        ET_LOG(Error, "aoti_torch_mps_start_encoding exception: %s", e.what());
        return Error::Internal;
    } catch (...) {
        ET_LOG(Error, "aoti_torch_mps_start_encoding: unknown exception");
        return Error::Internal;
    }
}

AOTITorchError aoti_torch_mps_set_arg_tensor(
    AOTIMetalKernelFunctionHandle func,
    unsigned idx,
    AtenTensorHandle tensor) {

    if (!func || !tensor) {
        ET_LOG(Error, "aoti_torch_mps_set_arg_tensor: null function handle or tensor");
        return Error::InvalidArgument;
    }

    @autoreleasepool {
        try {
            auto* function = reinterpret_cast<ETMetalKernelFunction*>(func);
            auto* et_tensor = reinterpret_cast<executorch::runtime::etensor::Tensor*>(tensor);

            function->setArg(idx, *et_tensor);

            ET_LOG(Debug, "aoti_torch_mps_set_arg_tensor: Set tensor argument at index %u", idx);
            return Error::Ok;

        } catch (const std::exception& e) {
            ET_LOG(Error, "aoti_torch_mps_set_arg_tensor exception: %s", e.what());
            return Error::Internal;
        } catch (...) {
            ET_LOG(Error, "aoti_torch_mps_set_arg_tensor: unknown exception");
            return Error::Internal;
        }
    }
}

AOTITorchError aoti_torch_mps_set_arg_int(
    AOTIMetalKernelFunctionHandle func,
    unsigned idx,
    int64_t val) {

    if (!func) {
        ET_LOG(Error, "aoti_torch_mps_set_arg_int: null function handle");
        return Error::InvalidArgument;
    }

    try {
        auto* function = reinterpret_cast<ETMetalKernelFunction*>(func);
        function->setArg(idx, val);

        ET_LOG(Debug, "aoti_torch_mps_set_arg_int: Set int64_t value %lld at index %u", val, idx);
        return Error::Ok;

    } catch (const std::exception& e) {
        ET_LOG(Error, "aoti_torch_mps_set_arg_int exception: %s", e.what());
        return Error::Internal;
    } catch (...) {
        ET_LOG(Error, "aoti_torch_mps_set_arg_int: unknown exception");
        return Error::Internal;
    }
}

// Pure C dispatch functions - single value versions
AOTITorchError aoti_torch_mps_dispatch_single(
    AOTIMetalKernelFunctionHandle func,
    uint64_t length) {

    if (!func) {
        ET_LOG(Error, "aoti_torch_mps_dispatch_single: null function handle");
        return Error::InvalidArgument;
    }

    try {
        auto* function = reinterpret_cast<ETMetalKernelFunction*>(func);
        function->dispatchSingle(length);

        ET_LOG(Debug, "aoti_torch_mps_dispatch_single: Dispatched function %p with length %llu", function, length);
        return Error::Ok;

    } catch (const std::exception& e) {
        ET_LOG(Error, "aoti_torch_mps_dispatch_single exception: %s", e.what());
        return Error::Internal;
    } catch (...) {
        ET_LOG(Error, "aoti_torch_mps_dispatch_single: unknown exception");
        return Error::Internal;
    }
}

AOTITorchError aoti_torch_mps_dispatch_single_with_group_size(
    AOTIMetalKernelFunctionHandle func,
    uint64_t length,
    uint64_t group_size) {

    if (!func) {
        ET_LOG(Error, "aoti_torch_mps_dispatch_single_with_group_size: null function handle");
        return Error::InvalidArgument;
    }

    try {
        auto* function = reinterpret_cast<ETMetalKernelFunction*>(func);
        function->dispatchSingleWithGroupSize(length, group_size);

        ET_LOG(Debug, "aoti_torch_mps_dispatch_single_with_group_size: Dispatched function %p with length %llu, group size %llu", function, length, group_size);
        return Error::Ok;

    } catch (const std::exception& e) {
        ET_LOG(Error, "aoti_torch_mps_dispatch_single_with_group_size exception: %s", e.what());
        return Error::Internal;
    } catch (...) {
        ET_LOG(Error, "aoti_torch_mps_dispatch_single_with_group_size: unknown exception");
        return Error::Internal;
    }
}

// Pure C dispatch functions - array versions
AOTITorchError aoti_torch_mps_dispatch_array(
    AOTIMetalKernelFunctionHandle func,
    const uint64_t* length,
    size_t length_size) {

    if (!func) {
        ET_LOG(Error, "aoti_torch_mps_dispatch_array: null function handle");
        return Error::InvalidArgument;
    }

    try {
        auto* function = reinterpret_cast<ETMetalKernelFunction*>(func);
        function->dispatchArray(length, length_size);

        ET_LOG(Debug, "aoti_torch_mps_dispatch_array: Dispatched function %p with %zu dimensions", function, length_size);
        return Error::Ok;

    } catch (const std::exception& e) {
        ET_LOG(Error, "aoti_torch_mps_dispatch_array exception: %s", e.what());
        return Error::Internal;
    } catch (...) {
        ET_LOG(Error, "aoti_torch_mps_dispatch_array: unknown exception");
        return Error::Internal;
    }
}

AOTITorchError aoti_torch_mps_dispatch_array_with_group_size(
    AOTIMetalKernelFunctionHandle func,
    const uint64_t* length,
    size_t length_size,
    const uint64_t* group_size,
    size_t group_size_size) {

    if (!func) {
        ET_LOG(Error, "aoti_torch_mps_dispatch_array_with_group_size: null function handle");
        return Error::InvalidArgument;
    }

    try {
        auto* function = reinterpret_cast<ETMetalKernelFunction*>(func);
        function->dispatchArrayWithGroupSize(length, length_size, group_size, group_size_size);

        ET_LOG(Debug, "aoti_torch_mps_dispatch_array_with_group_size: Dispatched function %p with %zu dimensions", function, length_size);
        return Error::Ok;

    } catch (const std::exception& e) {
        ET_LOG(Error, "aoti_torch_mps_dispatch_array_with_group_size exception: %s", e.what());
        return Error::Internal;
    } catch (...) {
        ET_LOG(Error, "aoti_torch_mps_dispatch_array_with_group_size: unknown exception");
        return Error::Internal;
    }
}

AOTITorchError aoti_torch_mps_malloc(void** buffer, size_t num_bytes) {
    if (num_bytes == 0) {
        *buffer = nullptr;
        return Error::Ok;
    }

    if (!buffer) {
        ET_LOG(Error, "aoti_torch_mps_malloc: null buffer pointer");
        return Error::InvalidArgument;
    }

    @autoreleasepool {
        try {
            id<MTLDevice> device = get_metal_device();
            if (!device) {
                ET_LOG(Error, "aoti_torch_mps_malloc: Failed to get Metal device");
                return Error::Internal;
            }

            id<MTLBuffer> metal_buffer = [device newBufferWithLength:num_bytes
                                                             options:MTLResourceCPUCacheModeWriteCombined | MTLResourceStorageModeShared];
            if (!metal_buffer) {
                ET_LOG(Error, "aoti_torch_mps_malloc: Failed to allocate Metal buffer of size %zu", num_bytes);
                return Error::Internal;
            }

            *buffer = (void*)metal_buffer;
            ET_LOG(Debug, "aoti_torch_mps_malloc: Allocated Metal buffer %p of size %zu", metal_buffer, num_bytes);
            return Error::Ok;

        } catch (const std::exception& e) {
            ET_LOG(Error, "aoti_torch_mps_malloc exception: %s", e.what());
            return Error::Internal;
        } catch (...) {
            ET_LOG(Error, "aoti_torch_mps_malloc: unknown exception");
            return Error::Internal;
        }
    }
}

AOTITorchError aoti_torch_mps_free(void* ptr) {
    if (!ptr) {
        return Error::Ok;  // Nothing to free
    }

    @autoreleasepool {
        try {
            auto metal_buffer = (id<MTLBuffer>)ptr;
            [metal_buffer release];

            ET_LOG(Debug, "aoti_torch_mps_free: Freed Metal buffer %p", ptr);
            return Error::Ok;

        } catch (const std::exception& e) {
            ET_LOG(Error, "aoti_torch_mps_free exception: %s", e.what());
            return Error::Internal;
        } catch (...) {
            ET_LOG(Error, "aoti_torch_mps_free: unknown exception");
            return Error::Internal;
        }
    }
}

AOTITorchError aoti_torch_mps_memcpy(
    void* buffer,
    size_t constant_offset,
    size_t bytes_read,
    size_t data_size,
    uint8_t* constants_start) {

    if (!buffer || !constants_start) {
        ET_LOG(Error, "aoti_torch_mps_memcpy: null buffer or constants_start");
        return Error::InvalidArgument;
    }

    @autoreleasepool {
        try {
            auto metal_buffer = (id<MTLBuffer>)buffer;
            auto buffer_pointer = static_cast<uint8_t*>([metal_buffer contents]);

            if (!buffer_pointer) {
                ET_LOG(Error, "aoti_torch_mps_memcpy: Failed to get buffer contents");
                return Error::Internal;
            }

            memcpy(buffer_pointer + constant_offset, constants_start + bytes_read, data_size);

            ET_LOG(Debug, "aoti_torch_mps_memcpy: Copied %zu bytes from offset %zu to buffer offset %zu",
                   data_size, bytes_read, constant_offset);
            return Error::Ok;

        } catch (const std::exception& e) {
            ET_LOG(Error, "aoti_torch_mps_memcpy exception: %s", e.what());
            return Error::Internal;
        } catch (...) {
            ET_LOG(Error, "aoti_torch_mps_memcpy: unknown exception");
            return Error::Internal;
        }
    }
}

AOTITorchError aoti_torch_mps_copy_buffer(
    void* src_buffer,
    void* dst_buffer,
    size_t data_size,
    size_t src_offset,
    size_t dst_offset) {

    if (!src_buffer || !dst_buffer) {
        ET_LOG(Error, "aoti_torch_mps_copy_buffer: null buffer");
        return Error::InvalidArgument;
    }

    @autoreleasepool {
        try {
            auto src_mtl_buffer = (id<MTLBuffer>)src_buffer;
            auto dst_mtl_buffer = (id<MTLBuffer>)dst_buffer;

            uint8_t* src_contents = static_cast<uint8_t*>([src_mtl_buffer contents]);
            uint8_t* dst_contents = static_cast<uint8_t*>([dst_mtl_buffer contents]);

            if (!src_contents || !dst_contents) {
                ET_LOG(Error, "aoti_torch_mps_copy_buffer: Failed to get buffer contents");
                return Error::Internal;
            }

            memcpy(dst_contents + dst_offset, src_contents + src_offset, data_size);

            ET_LOG(Debug, "aoti_torch_mps_copy_buffer: Copied %zu bytes from src+%zu to dst+%zu",
                   data_size, src_offset, dst_offset);
            return Error::Ok;

        } catch (const std::exception& e) {
            ET_LOG(Error, "aoti_torch_mps_copy_buffer exception: %s", e.what());
            return Error::Internal;
        } catch (...) {
            ET_LOG(Error, "aoti_torch_mps_copy_buffer: unknown exception");
            return Error::Internal;
        }
    }
}

AOTITorchError aoti_torch_mps_synchronize_stream() {
    @autoreleasepool {
        try {
            // Use the ETMetalStream for proper synchronization
            ETMetalStream* stream = getCurrentMetalStream();
            stream->synchronize(SyncType::COMMIT_AND_WAIT);

            ET_LOG(Debug, "aoti_torch_mps_synchronize_stream: Stream synchronized with COMMIT_AND_WAIT");
            return Error::Ok;

        } catch (const std::exception& e) {
            ET_LOG(Error, "aoti_torch_mps_synchronize_stream exception: %s", e.what());
            return Error::Internal;
        } catch (...) {
            ET_LOG(Error, "aoti_torch_mps_synchronize_stream: unknown exception");
            return Error::Internal;
        }
    }
}

// Synchronization function with SyncType options
AOTITorchError aoti_torch_mps_synchronize_stream_with_type(int sync_type) {
    @autoreleasepool {
        try {
            ETMetalStream* stream = getCurrentMetalStream();
            SyncType syncTypeEnum = static_cast<SyncType>(sync_type);
            stream->synchronize(syncTypeEnum);

            ET_LOG(Debug, "aoti_torch_mps_synchronize_stream_with_type: Stream synchronized with SyncType %d", sync_type);
            return Error::Ok;

        } catch (const std::exception& e) {
            ET_LOG(Error, "aoti_torch_mps_synchronize_stream_with_type exception: %s", e.what());
            return Error::Internal;
        } catch (...) {
            ET_LOG(Error, "aoti_torch_mps_synchronize_stream_with_type: unknown exception");
            return Error::Internal;
        }
    }
}

} // extern "C"

// C++ only functions
AOTITorchError aoti_torch_mps_run_command_block(
    AOTIMetalKernelFunctionHandle func,
    std::function<void(AOTIMetalKernelFunctionHandle)> command_block) {

    if (!func) {
        ET_LOG(Error, "aoti_torch_mps_run_command_block: null function handle");
        return Error::InvalidArgument;
    }

    try {
        auto* function = reinterpret_cast<ETMetalKernelFunction*>(func);
        function->runCommandBlock([&command_block, func]() {
            command_block(func);
        });

        ET_LOG(Debug, "aoti_torch_mps_run_command_block: Executed command block for function %p", function);
        return Error::Ok;

    } catch (const std::exception& e) {
        ET_LOG(Error, "aoti_torch_mps_run_command_block exception: %s", e.what());
        return Error::Internal;
    } catch (...) {
        ET_LOG(Error, "aoti_torch_mps_run_command_block: unknown exception");
        return Error::Internal;
    }
}

AOTITorchError aoti_torch_mps_addmm_out(
    AtenTensorHandle out,
    AtenTensorHandle self,
    AtenTensorHandle mat1,
    AtenTensorHandle mat2,
    double beta,
    double alpha) {

  ET_LOG(Debug, "aoti_torch_mps_addmm_out: Starting with out=%p, self=%p, mat1=%p, mat2=%p, beta=%f, alpha=%f",
         out, self, mat1, mat2, beta, alpha);

  if (!out || !self || !mat1 || !mat2) {
    ET_LOG(Error, "aoti_torch_mps_addmm_out: null tensor handles");
    return Error::InvalidArgument;
  }

  @autoreleasepool {
    try {
      // Convert AtenTensorHandle to ExecutorTorch tensors
      auto out_tensor = reinterpret_cast<executorch::runtime::etensor::Tensor*>(out);
      auto self_tensor = reinterpret_cast<executorch::runtime::etensor::Tensor*>(self);
      auto mat1_tensor = reinterpret_cast<executorch::runtime::etensor::Tensor*>(mat1);
      auto mat2_tensor = reinterpret_cast<executorch::runtime::etensor::Tensor*>(mat2);

      ET_LOG(Debug, "aoti_torch_mps_addmm_out: Converted tensor handles to ET tensors");

      // For now, just zero out the output tensor to get the right shape
      // TODO: Implement actual matrix multiplication: out = beta * self + alpha * (mat1 @ mat2)

      // Get output data pointer and size
      float* out_data = static_cast<float*>(out_tensor->mutable_data_ptr());
      size_t out_numel = out_tensor->numel();

      if (!out_data) {
        ET_LOG(Error, "aoti_torch_mps_addmm_out: null output data pointer");
        return Error::InvalidArgument;
      }

      // Zero out the output tensor
      std::memset(out_data, 0, out_numel * sizeof(float));

      ET_LOG(Debug, "aoti_torch_mps_addmm_out: Zeroed output tensor with %zu elements", out_numel);
      return Error::Ok;

    } catch (const std::exception& e) {
      ET_LOG(Error, "aoti_torch_mps_addmm_out exception: %s", e.what());
      return Error::Internal;
    } catch (...) {
      ET_LOG(Error, "aoti_torch_mps_addmm_out: unknown exception");
      return Error::Internal;
    }
  }
}

AOTITorchError aoti_torch_mps_mm_out(
    AtenTensorHandle out,
    AtenTensorHandle self,
    AtenTensorHandle mat2) {
    ET_LOG(Error, "aoti_torch_mps_mm_out: Legacy operation not supported in ExecuTorch");
    return Error::NotImplemented;
}

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
    AtenTensorHandle* ret0) {
    ET_LOG(Error, "aoti_torch_mps_convolution: Legacy operation not supported in ExecuTorch");
    return Error::NotImplemented;
}

} // namespace aoti
} // namespace backends
} // namespace executorch
