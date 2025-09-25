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
#include <unordered_map>

namespace executorch {
namespace backends {
namespace aoti {

// Declare the global mapping from et_metal.mm
extern std::unordered_map<void*, id<MTLBuffer>> ptr_to_mtl_buffer;

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
    AOTITensorHandle tensor) {

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

AOTITorchError aoti_torch_mps_mm_out(
    AOTITensorHandle out,
    AOTITensorHandle self,
    AOTITensorHandle mat2) {
  ET_LOG(Debug, "aoti_torch_mps_mm_out: Starting with out=%p, self=%p, mat2=%p",
         out, self, mat2);

  if (!out || !self || !mat2) {
    ET_LOG(Error, "aoti_torch_mps_mm_out: null tensor handles");
    return Error::InvalidArgument;
  }

  @autoreleasepool {
    try {
      // Convert AOTITensorHandle to ExecutorTorch tensors
      auto out_tensor = reinterpret_cast<executorch::runtime::etensor::Tensor*>(out);
      auto self_tensor = reinterpret_cast<executorch::runtime::etensor::Tensor*>(self);
      auto mat2_tensor = reinterpret_cast<executorch::runtime::etensor::Tensor*>(mat2);

      ET_LOG(Debug, "aoti_torch_mps_mm_out: Converted tensor handles to ET tensors");

      // Validate tensor dimensions
      if (self_tensor->dim() != 2 || mat2_tensor->dim() != 2) {
        ET_LOG(Error, "aoti_torch_mps_mm_out: tensors must be 2-D, got %d and %d",
               (int)self_tensor->dim(), (int)mat2_tensor->dim());
        return Error::InvalidArgument;
      }

      int64_t M = self_tensor->sizes()[0];  // rows of self
      int64_t K = self_tensor->sizes()[1];  // cols of self / rows of mat2
      int64_t N = mat2_tensor->sizes()[1];  // cols of mat2

      // Check matrix multiplication compatibility
      if (self_tensor->sizes()[1] != mat2_tensor->sizes()[0]) {
        ET_LOG(Error, "aoti_torch_mps_mm_out: incompatible matrix sizes for mm (%dx%d and %dx%d)",
               (int)M, (int)K, (int)mat2_tensor->sizes()[0], (int)N);
        return Error::InvalidArgument;
      }

      // Log tensor shapes for debugging
      ET_LOG(Debug, "aoti_torch_mps_mm_out: self shape: [%d, %d], mat2 shape: [%d, %d], out shape: [%d, %d]",
             (int)M, (int)K, (int)mat2_tensor->sizes()[0], (int)N,
             out_tensor->dim() > 0 ? (int)out_tensor->sizes()[0] : 0,
             out_tensor->dim() > 1 ? (int)out_tensor->sizes()[1] : 0);

      // Get Metal device and stream
      ETMetalStream* stream = getCurrentMetalStream();
      id<MTLDevice> device = get_metal_device();
      if (!device) {
        ET_LOG(Error, "aoti_torch_mps_mm_out: Failed to get Metal device");
        return Error::Internal;
      }

      // Get Metal buffers from tensors using the global mapping
      void* self_data_ptr = self_tensor->mutable_data_ptr();
      void* mat2_data_ptr = mat2_tensor->mutable_data_ptr();
      void* out_data_ptr = out_tensor->mutable_data_ptr();

      id<MTLBuffer> self_buffer = nullptr;
      id<MTLBuffer> mat2_buffer = nullptr;
      id<MTLBuffer> out_buffer = nullptr;

      // Look up Metal buffers from the global mapping
      auto self_it = ptr_to_mtl_buffer.find(self_data_ptr);
      auto mat2_it = ptr_to_mtl_buffer.find(mat2_data_ptr);
      auto out_it = ptr_to_mtl_buffer.find(out_data_ptr);

      if (self_it != ptr_to_mtl_buffer.end()) {
        self_buffer = self_it->second;
      }
      if (mat2_it != ptr_to_mtl_buffer.end()) {
        mat2_buffer = mat2_it->second;
      }
      if (out_it != ptr_to_mtl_buffer.end()) {
        out_buffer = out_it->second;
      }

      // If buffers are not in Metal memory, create temporary Metal buffers
      if (!self_buffer) {
        size_t self_size = self_tensor->numel() * sizeof(float);
        self_buffer = [device newBufferWithBytes:self_data_ptr
                                          length:self_size
                                         options:MTLResourceStorageModeShared];
        if (!self_buffer) {
          ET_LOG(Error, "aoti_torch_mps_mm_out: Failed to create Metal buffer for self tensor");
          return Error::Internal;
        }
      }

      if (!mat2_buffer) {
        size_t mat2_size = mat2_tensor->numel() * sizeof(float);
        mat2_buffer = [device newBufferWithBytes:mat2_data_ptr
                                          length:mat2_size
                                         options:MTLResourceStorageModeShared];
        if (!mat2_buffer) {
          ET_LOG(Error, "aoti_torch_mps_mm_out: Failed to create Metal buffer for mat2 tensor");
          return Error::Internal;
        }
      }

      if (!out_buffer) {
        size_t out_size = out_tensor->numel() * sizeof(float);
        out_buffer = [device newBufferWithBytes:out_data_ptr
                                         length:out_size
                                        options:MTLResourceStorageModeShared];
        if (!out_buffer) {
          ET_LOG(Error, "aoti_torch_mps_mm_out: Failed to create Metal buffer for out tensor");
          return Error::Internal;
        }
      }

      // End any existing kernel coalescing to ensure a clean state for MPS
      stream->endKernelCoalescing();

      // Get command buffer from stream (stream manages lifecycle)
      id<MTLCommandBuffer> commandBuffer = stream->commandBuffer();

      // Create matrix descriptors for the multiplication
      MPSMatrixDescriptor* selfDesc = [MPSMatrixDescriptor matrixDescriptorWithRows:M
                                                                            columns:K
                                                                           matrices:1
                                                                           rowBytes:K * sizeof(float)
                                                                        matrixBytes:M * K * sizeof(float)
                                                                           dataType:MPSDataTypeFloat32];

      MPSMatrixDescriptor* mat2Desc = [MPSMatrixDescriptor matrixDescriptorWithRows:K
                                                                            columns:N
                                                                           matrices:1
                                                                           rowBytes:N * sizeof(float)
                                                                        matrixBytes:K * N * sizeof(float)
                                                                           dataType:MPSDataTypeFloat32];

      MPSMatrixDescriptor* outDesc = [MPSMatrixDescriptor matrixDescriptorWithRows:M
                                                                           columns:N
                                                                          matrices:1
                                                                          rowBytes:N * sizeof(float)
                                                                       matrixBytes:M * N * sizeof(float)
                                                                          dataType:MPSDataTypeFloat32];

      MPSMatrix* selfMatrix = [[MPSMatrix alloc] initWithBuffer:self_buffer
                                                         offset:0
                                                     descriptor:selfDesc];

      MPSMatrix* mat2Matrix = [[MPSMatrix alloc] initWithBuffer:mat2_buffer
                                                         offset:0
                                                     descriptor:mat2Desc];

      MPSMatrix* outMatrix = [[MPSMatrix alloc] initWithBuffer:out_buffer
                                                        offset:0
                                                    descriptor:outDesc];

      // Create matrix multiplication kernel
      MPSMatrixMultiplication* matmul = [[MPSMatrixMultiplication alloc] initWithDevice:device
                                                                         transposeLeft:NO
                                                                        transposeRight:NO
                                                                            resultRows:M
                                                                         resultColumns:N
                                                                      interiorColumns:K
                                                                                 alpha:1.0
                                                                                  beta:0.0];

      // Encode the matrix multiplication (stream will handle commit/synchronization)
      [matmul encodeToCommandBuffer:commandBuffer
                         leftMatrix:selfMatrix
                        rightMatrix:mat2Matrix
                       resultMatrix:outMatrix];

      // Clean up MPS objects
      [selfMatrix release];
      [mat2Matrix release];
      [outMatrix release];
      [matmul release];

      ET_LOG(Debug, "aoti_torch_mps_mm_out: Matrix multiplication completed successfully");
      return Error::Ok;

    } catch (const std::exception& e) {
      ET_LOG(Error, "aoti_torch_mps_mm_out exception: %s", e.what());
      return Error::Internal;
    } catch (...) {
      ET_LOG(Error, "aoti_torch_mps_mm_out: unknown exception");
      return Error::Internal;
    }
  }
}

AOTITorchError aoti_torch_mps_addmm_out(
    AOTITensorHandle out,
    AOTITensorHandle self,
    AOTITensorHandle mat1,
    AOTITensorHandle mat2,
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
      // Convert AOTITensorHandle to ExecutorTorch tensors
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

AOTITorchError aoti_torch_mps_convolution(
    AOTITensorHandle input,
    AOTITensorHandle weight,
    AOTITensorHandle* bias,
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
    AOTITensorHandle* ret0) {
  ET_LOG(Debug, "aoti_torch_mps_convolution: Starting with input=%p, weight=%p, bias=%p, groups=%lld, transposed=%d",
         input, weight, bias, groups, transposed);

  if (!input || !weight || !ret0) {
    ET_LOG(Error, "aoti_torch_mps_convolution: null required handles (input, weight, or ret0)");
    return Error::InvalidArgument;
  }

  @autoreleasepool {
    try {
      // Convert AOTITensorHandle to ExecutorTorch tensors
      auto input_tensor = reinterpret_cast<executorch::runtime::etensor::Tensor*>(input);
      auto weight_tensor = reinterpret_cast<executorch::runtime::etensor::Tensor*>(weight);

      // bias can be null for convolutions without bias
      executorch::runtime::etensor::Tensor* bias_tensor = nullptr;
      if (bias && *bias) {
        bias_tensor = reinterpret_cast<executorch::runtime::etensor::Tensor*>(*bias);
        ET_LOG(Debug, "aoti_torch_mps_convolution: Has bias tensor");
      } else {
        ET_LOG(Debug, "aoti_torch_mps_convolution: No bias tensor");
      }

      ET_LOG(Debug, "aoti_torch_mps_convolution: Converted tensor handles to ET tensors");

      // Log tensor shapes for debugging
      ET_LOG(Debug, "aoti_torch_mps_convolution: input shape: [%d, %d, %d, %d]",
             input_tensor->dim() > 0 ? (int)input_tensor->sizes()[0] : 0,
             input_tensor->dim() > 1 ? (int)input_tensor->sizes()[1] : 0,
             input_tensor->dim() > 2 ? (int)input_tensor->sizes()[2] : 0,
             input_tensor->dim() > 3 ? (int)input_tensor->sizes()[3] : 0);

      ET_LOG(Debug, "aoti_torch_mps_convolution: weight shape: [%d, %d, %d, %d]",
             weight_tensor->dim() > 0 ? (int)weight_tensor->sizes()[0] : 0,
             weight_tensor->dim() > 1 ? (int)weight_tensor->sizes()[1] : 0,
             weight_tensor->dim() > 2 ? (int)weight_tensor->sizes()[2] : 0,
             weight_tensor->dim() > 3 ? (int)weight_tensor->sizes()[3] : 0);

      // Log convolution parameters
      if (stride && stride_len_ >= 2) {
        ET_LOG(Debug, "aoti_torch_mps_convolution: stride: [%lld, %lld]", stride[0], stride[1]);
      }
      if (padding && padding_len_ >= 2) {
        ET_LOG(Debug, "aoti_torch_mps_convolution: padding: [%lld, %lld]", padding[0], padding[1]);
      }
      if (dilation && dilation_len_ >= 2) {
        ET_LOG(Debug, "aoti_torch_mps_convolution: dilation: [%lld, %lld]", dilation[0], dilation[1]);
      }
      if (output_padding && output_padding_len_ >= 2) {
        ET_LOG(Debug, "aoti_torch_mps_convolution: output_padding: [%lld, %lld]", output_padding[0], output_padding[1]);
      }

      // Calculate output dimensions
      // For now, we'll create a zero-filled tensor with the expected output shape
      // TODO: Implement actual 2D convolution using MetalPerformanceShaders or custom Metal kernels

      // Get input dimensions (assuming NCHW format)
      int64_t N = input_tensor->sizes()[0];  // batch size
      int64_t C_in = input_tensor->sizes()[1];  // input channels
      int64_t H_in = input_tensor->sizes()[2];  // input height
      int64_t W_in = input_tensor->sizes()[3];  // input width

      // Get weight dimensions (assuming OIHW format for weight)
      int64_t C_out = weight_tensor->sizes()[0];  // output channels
      int64_t kernel_h = weight_tensor->sizes()[2];  // kernel height
      int64_t kernel_w = weight_tensor->sizes()[3];  // kernel width

      // Calculate output dimensions
      int64_t stride_h = stride && stride_len_ > 0 ? stride[0] : 1;
      int64_t stride_w = stride && stride_len_ > 1 ? stride[1] : 1;
      int64_t pad_h = padding && padding_len_ > 0 ? padding[0] : 0;
      int64_t pad_w = padding && padding_len_ > 1 ? padding[1] : 0;
      int64_t dil_h = dilation && dilation_len_ > 0 ? dilation[0] : 1;
      int64_t dil_w = dilation && dilation_len_ > 1 ? dilation[1] : 1;

      int64_t H_out = (H_in + 2 * pad_h - dil_h * (kernel_h - 1) - 1) / stride_h + 1;
      int64_t W_out = (W_in + 2 * pad_w - dil_w * (kernel_w - 1) - 1) / stride_w + 1;

      ET_LOG(Debug, "aoti_torch_mps_convolution: Calculated output shape: [%lld, %lld, %lld, %lld]", N, C_out, H_out, W_out);

      // Validate output dimensions are positive
      if (N <= 0 || C_out <= 0 || H_out <= 0 || W_out <= 0) {
        ET_LOG(Error, "aoti_torch_mps_convolution: Invalid output dimensions N=%lld, C_out=%lld, H_out=%lld, W_out=%lld",
               N, C_out, H_out, W_out);
        return Error::InvalidArgument;
      }

      // Create output tensor with calculated dimensions
      std::vector<int32_t> output_sizes = {(int32_t)N, (int32_t)C_out, (int32_t)H_out, (int32_t)W_out};

      // Calculate expected number of elements
      size_t expected_numel = N * C_out * H_out * W_out;
      ET_LOG(Debug, "aoti_torch_mps_convolution: Expected output tensor numel = %zu", expected_numel);

      // Log the sizes vector for debugging
      ET_LOG(Debug, "aoti_torch_mps_convolution: output_sizes vector: [%d, %d, %d, %d]",
             output_sizes[0], output_sizes[1], output_sizes[2], output_sizes[3]);

      // Allocate memory for the tensor data
      size_t tensor_size_bytes = expected_numel * sizeof(float);
      void* tensor_data = std::malloc(tensor_size_bytes);
      if (!tensor_data) {
        ET_LOG(Error, "aoti_torch_mps_convolution: Failed to allocate %zu bytes for tensor", tensor_size_bytes);
        return Error::Internal;
      }

      // Zero out the allocated memory
      std::memset(tensor_data, 0, tensor_size_bytes);

      // Create tensor using aoti_torch_create_tensor_from_blob_v2 to ensure we have control over the memory
      // Convert sizes vector to int64_t array
      std::vector<int64_t> output_sizes_int64 = {N, C_out, H_out, W_out};

      // Calculate default strides for a contiguous tensor (NCHW format)
      std::vector<int64_t> output_strides = {
          C_out * H_out * W_out,  // Stride for N dimension
          H_out * W_out,          // Stride for C dimension
          W_out,                  // Stride for H dimension
          1                       // Stride for W dimension
      };

      AOTITensorHandle output_tensor_handle = nullptr;

      AOTITorchError create_result = aoti_torch_create_tensor_from_blob_v2(
          tensor_data,
          4,  // ndim
          output_sizes_int64.data(),
          output_strides.data(),
          0,  // storage_offset
          static_cast<int32_t>(SupportedDTypes::FLOAT32),  // dtype
          0,  // device_type (CPU)
          0,  // device_index
          &output_tensor_handle,
          0,  // layout (strided)
          nullptr,  // opaque_metadata
          0   // opaque_metadata_size
      );

      if (create_result != Error::Ok || !output_tensor_handle) {
        ET_LOG(Error, "aoti_torch_mps_convolution: Failed to create output tensor, error code: %d", static_cast<int>(create_result));
        std::free(tensor_data);  // Free the allocated memory on failure
        return Error::Internal;
      }

      // Verify the tensor was created with the correct size
      auto* et_tensor = reinterpret_cast<executorch::runtime::etensor::Tensor*>(output_tensor_handle);
      size_t actual_numel = et_tensor->numel();
      ET_LOG(Debug, "aoti_torch_mps_convolution: Created tensor with actual numel = %zu", actual_numel);

      if (actual_numel != expected_numel) {
        ET_LOG(Error, "aoti_torch_mps_convolution: Tensor size mismatch. Expected %zu, got %zu", expected_numel, actual_numel);
        std::free(tensor_data);  // Free the allocated memory on failure
        return Error::Internal;
      }

      // Store the tensor handle - mark that we own the memory since we manually allocated it with malloc
      *ret0 = output_tensor_handle;
      is_tensor_own_memory[et_tensor] = true;  // We allocated the memory manually

      ET_LOG(Debug, "aoti_torch_mps_convolution: Created zero-filled output tensor with %zu elements", actual_numel);
      return Error::Ok;

    } catch (const std::exception& e) {
      ET_LOG(Error, "aoti_torch_mps_convolution exception: %s", e.what());
      return Error::Internal;
    } catch (...) {
      ET_LOG(Error, "aoti_torch_mps_convolution: unknown exception");
      return Error::Internal;
    }
  }
}

} // namespace aoti
} // namespace backends
} // namespace executorch
