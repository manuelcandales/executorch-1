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
#include "metal_helper.h"
#include "utils.h"
#include "memory.h"
#include <unordered_map>
#include <memory>
#include <string>
#include <functional>

// ExecuTorch-specific C++ headers (if c10 headers aren't available, we'll provide minimal compatibility)
#ifndef C10_UTIL_ARRAYREF_H_
// Minimal ArrayRef compatibility for ExecuTorch
namespace c10 {
template<typename T>
class ArrayRef {
public:
    ArrayRef() : data_(nullptr), size_(0) {}
    ArrayRef(const T* data, size_t size) : data_(data), size_(size) {}
    ArrayRef(const std::vector<T>& vec) : data_(vec.data()), size_(vec.size()) {}

    const T* data() const { return data_; }
    size_t size() const { return size_; }
    const T& operator[](size_t i) const { return data_[i]; }

private:
    const T* data_;
    size_t size_;
};

template<typename T>
class OptionalArrayRef {
public:
    OptionalArrayRef() : has_value_(false) {}
    OptionalArrayRef(std::nullopt_t) : has_value_(false) {}
    OptionalArrayRef(ArrayRef<T> ref) : has_value_(true), value_(ref) {}

    bool has_value() const { return has_value_; }
    const ArrayRef<T>& value() const { return value_; }

private:
    bool has_value_;
    ArrayRef<T> value_;
};
}
#else
#include <c10/util/ArrayRef.h>
#include <c10/util/OptionalArrayRef.h>
#endif

namespace executorch {
namespace backends {
namespace aoti {

// ExecuTorch MetalShaderLibrary equivalent - simplified version of PyTorch's implementation
class ETMetalShaderLibrary {
public:
    ETMetalShaderLibrary(const std::string& source) : shaderSource_(source) {
        compileLibrary();
    }

    ~ETMetalShaderLibrary() {
        if (library_) {
            [library_ release];
            library_ = nil;
        }

        // Clean up pipeline states
        for (auto& pair : pipelineStates_) {
            [pair.second.first release];  // MTLComputePipelineState
            [pair.second.second release]; // MTLFunction
        }
        pipelineStates_.clear();
    }

    std::shared_ptr<class ETMetalKernelFunction> getKernelFunction(const std::string& name);

private:
    void compileLibrary() {
        @autoreleasepool {
            id<MTLDevice> device = get_metal_device();
            if (!device) {
                ET_LOG(Error, "ETMetalShaderLibrary: Failed to get Metal device");
                return;
            }

            NSString* sourceString = [NSString stringWithUTF8String:shaderSource_.c_str()];
            NSError* error = nil;

            library_ = [device newLibraryWithSource:sourceString options:nil error:&error];
            if (!library_ || error) {
                ET_LOG(Error, "ETMetalShaderLibrary: Failed to compile shader library: %s",
                       error ? [[error localizedDescription] UTF8String] : "unknown error");
                return;
            }

            [library_ retain];
            ET_LOG(Debug, "ETMetalShaderLibrary: Successfully compiled shader library");
        }
    }

    std::pair<id<MTLComputePipelineState>, id<MTLFunction>> getLibraryPipelineState(const std::string& functionName) {
        auto it = pipelineStates_.find(functionName);
        if (it != pipelineStates_.end()) {
            return it->second;
        }

        @autoreleasepool {
            if (!library_) {
                ET_LOG(Error, "ETMetalShaderLibrary: Library not compiled");
                return {nil, nil};
            }

            id<MTLDevice> device = get_metal_device();
            if (!device) {
                ET_LOG(Error, "ETMetalShaderLibrary: Failed to get Metal device");
                return {nil, nil};
            }

            NSString* funcName = [NSString stringWithUTF8String:functionName.c_str()];
            id<MTLFunction> function = [library_ newFunctionWithName:funcName];
            if (!function) {
                ET_LOG(Error, "ETMetalShaderLibrary: Failed to get function '%s'", functionName.c_str());
                return {nil, nil};
            }

            NSError* error = nil;
            id<MTLComputePipelineState> pipelineState = [device newComputePipelineStateWithFunction:function error:&error];
            if (!pipelineState || error) {
                ET_LOG(Error, "ETMetalShaderLibrary: Failed to create pipeline state for '%s': %s",
                       functionName.c_str(), error ? [[error localizedDescription] UTF8String] : "unknown error");
                [function release];
                return {nil, nil};
            }

            // Retain objects and store in cache
            [pipelineState retain];
            [function retain];
            pipelineStates_[functionName] = {pipelineState, function};

            ET_LOG(Debug, "ETMetalShaderLibrary: Created pipeline state for function '%s'", functionName.c_str());
            return {pipelineState, function};
        }
    }

    friend class ETMetalKernelFunction;

    std::string shaderSource_;
    id<MTLLibrary> library_ = nil;
    std::unordered_map<std::string, std::pair<id<MTLComputePipelineState>, id<MTLFunction>>> pipelineStates_;
};

// ExecuTorch MetalKernelFunction equivalent - simplified version of PyTorch's implementation
class ETMetalKernelFunction {
public:
    ETMetalKernelFunction(id<MTLComputePipelineState> cps, id<MTLFunction> func)
        : cps_(cps), func_(func), encoder_(nil) {
        if (cps_) [cps_ retain];
        if (func_) [func_ retain];
    }

    ~ETMetalKernelFunction() {
        if (encoder_) {
            [encoder_ release];
            encoder_ = nil;
        }
        if (cps_) {
            [cps_ release];
            cps_ = nil;
        }
        if (func_) {
            [func_ release];
            func_ = nil;
        }
    }

    void startEncoding() {
        @autoreleasepool {
            if (encoder_) {
                [encoder_ release];
                encoder_ = nil;
            }

            id<MTLCommandQueue> commandQueue = get_metal_command_queue();
            if (!commandQueue) {
                ET_LOG(Error, "ETMetalKernelFunction: Failed to get command queue");
                return;
            }

            id<MTLCommandBuffer> commandBuffer = [commandQueue commandBuffer];
            if (!commandBuffer) {
                ET_LOG(Error, "ETMetalKernelFunction: Failed to create command buffer");
                return;
            }

            encoder_ = [commandBuffer computeCommandEncoder];
            if (!encoder_) {
                ET_LOG(Error, "ETMetalKernelFunction: Failed to create compute command encoder");
                return;
            }

            [encoder_ retain];
            [encoder_ setComputePipelineState:cps_];

            ET_LOG(Debug, "ETMetalKernelFunction: Started encoding");
        }
    }

    void setArg(unsigned idx, const executorch::runtime::etensor::Tensor& tensor) {
        if (!encoder_) {
            ET_LOG(Error, "ETMetalKernelFunction::setArg: No active encoder");
            return;
        }

        void* data_ptr = tensor.mutable_data_ptr();

        // Check if this is a Metal buffer or CPU tensor
        auto it = ptr_to_mtl_buffer.find(data_ptr);
        if (it != ptr_to_mtl_buffer.end()) {
            // This tensor data lives in a Metal buffer
            id<MTLBuffer> mtlBuffer = it->second;
            [encoder_ setBuffer:mtlBuffer offset:0 atIndex:idx];
            ET_LOG(Debug, "ETMetalKernelFunction::setArg: Set Metal buffer at index %u", idx);
        } else {
            // CPU tensor - use setBytes for small data or create temporary buffer for large data
            size_t totalSize = tensor.numel() * tensor.element_size();
            if (totalSize <= 4096) {
                [encoder_ setBytes:data_ptr length:totalSize atIndex:idx];
                ET_LOG(Debug, "ETMetalKernelFunction::setArg: Set CPU tensor via setBytes at index %u", idx);
            } else {
                // Create temporary buffer for large data
                id<MTLDevice> device = get_metal_device();
                id<MTLBuffer> tempBuffer = [device newBufferWithBytes:data_ptr length:totalSize options:MTLResourceStorageModeShared];
                [encoder_ setBuffer:tempBuffer offset:0 atIndex:idx];
                ET_LOG(Debug, "ETMetalKernelFunction::setArg: Set large CPU tensor via temporary buffer at index %u", idx);
            }
        }
    }

    void setArg(unsigned idx, int64_t val) {
        if (!encoder_) {
            ET_LOG(Error, "ETMetalKernelFunction::setArg: No active encoder");
            return;
        }

        [encoder_ setBytes:&val length:sizeof(int64_t) atIndex:idx];
        ET_LOG(Debug, "ETMetalKernelFunction::setArg: Set int64_t value %lld at index %u", val, idx);
    }

    void setArg(unsigned idx, const void* ptr, uint64_t size) {
        if (!encoder_) {
            ET_LOG(Error, "ETMetalKernelFunction::setArg: No active encoder");
            return;
        }

        [encoder_ setBytes:ptr length:size atIndex:idx];
        ET_LOG(Debug, "ETMetalKernelFunction::setArg: Set bytes at index %u, size %llu", idx, size);
    }

    void dispatch(uint64_t length, std::optional<uint64_t> groupSize = std::nullopt) {
        if (!encoder_) {
            ET_LOG(Error, "ETMetalKernelFunction::dispatch: No active encoder");
            return;
        }

        const auto maxThreadsPerGroup = [cps_ maxTotalThreadsPerThreadgroup];
        uint64_t actualGroupSize = groupSize.value_or(std::min(maxThreadsPerGroup, length));

        auto size = MTLSizeMake(length, 1, 1);
        auto threadGroupSize = MTLSizeMake(actualGroupSize, 1, 1);

        [encoder_ dispatchThreads:size threadsPerThreadgroup:threadGroupSize];
        ET_LOG(Debug, "ETMetalKernelFunction::dispatch: Dispatched with length %llu, group size %llu", length, actualGroupSize);
    }

    void dispatch(c10::ArrayRef<uint64_t> length, c10::OptionalArrayRef<uint64_t> groupSize = std::nullopt) {
        if (!encoder_) {
            ET_LOG(Error, "ETMetalKernelFunction::dispatch: No active encoder");
            return;
        }

        if (length.size() == 0) {
            ET_LOG(Error, "ETMetalKernelFunction::dispatch: Empty length array");
            return;
        }

        const auto maxThreadsPerGroup = [cps_ maxTotalThreadsPerThreadgroup];

        // Handle 1D, 2D, and 3D dispatch
        MTLSize size, threadGroupSize;

        if (length.size() == 1) {
            size = MTLSizeMake(length[0], 1, 1);
            uint64_t actualGroupSize = maxThreadsPerGroup;
            if (groupSize.has_value() && groupSize.value().size() > 0) {
                actualGroupSize = std::min(maxThreadsPerGroup, groupSize.value()[0]);
            }
            threadGroupSize = MTLSizeMake(actualGroupSize, 1, 1);
        } else if (length.size() == 2) {
            size = MTLSizeMake(length[0], length[1], 1);
            uint64_t groupX = std::min(static_cast<uint64_t>(32), length[0]);
            uint64_t groupY = maxThreadsPerGroup / groupX;
            if (groupSize.has_value() && groupSize.value().size() >= 2) {
                groupX = std::min(static_cast<uint64_t>(groupSize.value()[0]), length[0]);
                groupY = std::min(static_cast<uint64_t>(groupSize.value()[1]), length[1]);
            }
            threadGroupSize = MTLSizeMake(groupX, groupY, 1);
        } else { // 3D or higher - treat as 3D
            size = MTLSizeMake(length[0], length[1], length.size() > 2 ? length[2] : 1);
            uint64_t groupX = std::min(static_cast<uint64_t>(8), length[0]);
            uint64_t groupY = std::min(static_cast<uint64_t>(8), length[1]);
            uint64_t groupZ = maxThreadsPerGroup / (groupX * groupY);
            if (groupSize.has_value() && groupSize.value().size() >= 3) {
                groupX = std::min(static_cast<uint64_t>(groupSize.value()[0]), length[0]);
                groupY = std::min(static_cast<uint64_t>(groupSize.value()[1]), length[1]);
                groupZ = std::min(static_cast<uint64_t>(groupSize.value()[2]), length.size() > 2 ? length[2] : 1);
            }
            threadGroupSize = MTLSizeMake(groupX, groupY, groupZ);
        }

        [encoder_ dispatchThreads:size threadsPerThreadgroup:threadGroupSize];
        ET_LOG(Debug, "ETMetalKernelFunction::dispatch: Dispatched %luD with size [%lu, %lu, %lu], group [%lu, %lu, %lu]",
               length.size(), size.width, size.height, size.depth,
               threadGroupSize.width, threadGroupSize.height, threadGroupSize.depth);
    }

    void runCommandBlock(std::function<void(void)> f) {
        @autoreleasepool {
            if (!encoder_) {
                ET_LOG(Error, "ETMetalKernelFunction::runCommandBlock: No active encoder");
                return;
            }

            // Execute the command block
            f();

            // End encoding and commit
            [encoder_ endEncoding];

            id<MTLCommandBuffer> commandBuffer = [encoder_ commandBuffer];
            [commandBuffer commit];
            [commandBuffer waitUntilCompleted];

            ET_LOG(Debug, "ETMetalKernelFunction::runCommandBlock: Executed command block");
        }
    }

    // Accessor for encoder (needed by shim functions)
    id<MTLComputeCommandEncoder> getEncoder() const { return encoder_; }

private:
    id<MTLComputePipelineState> cps_;
    id<MTLFunction> func_;
    id<MTLComputeCommandEncoder> encoder_;
};

std::shared_ptr<ETMetalKernelFunction> ETMetalShaderLibrary::getKernelFunction(const std::string& name) {
    auto pipelineStatePair = getLibraryPipelineState(name);
    if (!pipelineStatePair.first || !pipelineStatePair.second) {
        ET_LOG(Error, "ETMetalShaderLibrary::getKernelFunction: Failed to get pipeline state for '%s'", name.c_str());
        return nullptr;
    }

    return std::make_shared<ETMetalKernelFunction>(pipelineStatePair.first, pipelineStatePair.second);
}

// Global storage to keep shared_ptr alive while raw pointers are used (matching PyTorch pattern)
static std::unordered_map<ETMetalKernelFunction*, std::shared_ptr<ETMetalKernelFunction>> function_storage;
static std::unordered_map<ETMetalShaderLibrary*, std::unique_ptr<ETMetalShaderLibrary>> library_storage;

// We need to match PyTorch's MetalKernelFunction structure to extract the encoder
// This is based on PyTorch's ATen/native/mps/MetalShaderLibrary.h
namespace {
  // Match the actual PyTorch MetalKernelFunction structure
  // From MetalShaderLibrary.h: cps, func, encoder (in that order)
  struct MetalKernelFunctionShim {
    id<MTLComputePipelineState> cps;    // First member
    id<MTLFunction> func;               // Second member
    id<MTLComputeCommandEncoder> encoder;  // Third member (what we need)
  };
}

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
            library_storage[raw_library] = std::move(library);

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
        auto it = library_storage.find(library);
        if (it != library_storage.end()) {
            library_storage.erase(it);
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
        function_storage[raw_function] = function_shared_ptr;

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

// MetalKernelFunction functions
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

            // For simplicity, use CPU-side memory copy
            // In a full implementation, you might want to use GPU-side copy via Metal command encoder
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
            // For ExecuTorch, we'll use simple synchronous execution
            // In a full implementation, you might want to implement proper stream synchronization
            id<MTLCommandQueue> commandQueue = get_metal_command_queue();
            if (!commandQueue) {
                ET_LOG(Error, "aoti_torch_mps_synchronize_stream: Failed to get command queue");
                return Error::Internal;
            }

            // Create a dummy command buffer and wait for it to complete to ensure synchronization
            id<MTLCommandBuffer> commandBuffer = [commandQueue commandBuffer];
            [commandBuffer commit];
            [commandBuffer waitUntilCompleted];

            ET_LOG(Debug, "aoti_torch_mps_synchronize_stream: Stream synchronized");
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
      ET_LOG(Debug, "aoti_torch_mps_set_arg_tensor: Starting with func=%p, idx=%u, tensor=%p", func, idx, tensor);

      // Cast the opaque handle to our shim structure to access the encoder
      auto kernelFunc = reinterpret_cast<MetalKernelFunctionShim*>(func);
      ET_LOG(Debug, "aoti_torch_mps_set_arg_tensor: Cast to kernelFunc=%p", kernelFunc);

      id<MTLComputeCommandEncoder> encoder = kernelFunc->encoder;
      ET_LOG(Debug, "aoti_torch_mps_set_arg_tensor: Retrieved encoder=%p", encoder);

      if (!encoder) {
        ET_LOG(Error, "aoti_torch_mps_set_arg_tensor: null encoder");
        return Error::InvalidArgument;
      }

      // Convert the AtenTensorHandle to our ExecutorTorch tensor
      // In our case, AtenTensorHandle is just a pointer to our ExecutorTorch tensor
      auto et_tensor = reinterpret_cast<executorch::runtime::etensor::Tensor*>(tensor);
      ET_LOG(Debug, "aoti_torch_mps_set_arg_tensor: Cast to et_tensor=%p", et_tensor);

      // Get the data pointer
      void* data_ptr = et_tensor->mutable_data_ptr();
      ET_LOG(Debug, "aoti_torch_mps_set_arg_tensor: Retrieved data_ptr=%p", data_ptr);

      // Check if this is a Metal buffer or CPU tensor
      auto it = ptr_to_mtl_buffer.find(data_ptr);
      if (it != ptr_to_mtl_buffer.end()) {
        // This tensor data lives in a Metal buffer - use setBuffer
        id<MTLBuffer> mtlBuffer = it->second;
        ET_LOG(Debug, "aoti_torch_mps_set_arg_tensor: Retrieved mtlBuffer=%p", mtlBuffer);

        if (!mtlBuffer) {
          ET_LOG(Error, "aoti_torch_mps_set_arg_tensor: null MTLBuffer");
          return Error::Internal;
        }

        // Set the Metal buffer directly on the encoder
        // ExecutorTorch tensors don't have storage_offset, so we assume offset 0
        // This is fine because ExecutorTorch tensors are typically not views
        size_t offset = 0;
        ET_LOG(Debug, "aoti_torch_mps_set_arg_tensor: About to call setBuffer with idx=%u, offset=%zu", idx, offset);

        [encoder setBuffer:mtlBuffer offset:offset atIndex:idx];

        // Also log the buffer contents for debugging (first few bytes)
        void* bufferContents = [mtlBuffer contents];
        if (bufferContents) {
          float* floatData = (float*)bufferContents;
          ET_LOG(Debug, "aoti_torch_mps_set_arg_tensor: Buffer contents at idx %u: [%.3f, %.3f, %.3f, ...]",
                 idx, floatData[0], floatData[1], floatData[2]);
        }

        ET_LOG(Debug, "aoti_torch_mps_set_arg_tensor: Successfully set Metal buffer at index %u with offset %zu",
               idx, offset);

      } else {
        // This is a CPU tensor - handle as bytes or buffer depending on size
        int dims = et_tensor->dim();
        size_t numel = et_tensor->numel();
        size_t element_size = et_tensor->element_size();
        size_t total_size = numel * element_size;

        ET_LOG(Debug, "aoti_torch_mps_set_arg_tensor: CPU tensor with dims=%d, numel=%zu, element_size=%zu, total_size=%zu",
               dims, numel, element_size, total_size);

        // Metal has a limit of 4096 bytes for setBytes
        // For larger data, we need to create a temporary buffer
        if (total_size <= 4096) {
          // Small data - use setBytes
          ET_LOG(Debug, "aoti_torch_mps_set_arg_tensor: Using setBytes for small tensor (size=%zu)", total_size);
          [encoder setBytes:data_ptr length:total_size atIndex:idx];
        } else {
          // Large data - create a temporary Metal buffer
          ET_LOG(Debug, "aoti_torch_mps_set_arg_tensor: Creating temporary buffer for large tensor (size=%zu)", total_size);

          // Get Metal device
          id<MTLDevice> device = get_metal_device();
          if (!device) {
            ET_LOG(Error, "aoti_torch_mps_set_arg_tensor: Failed to get Metal device for large tensor");
            return Error::Internal;
          }

          // Create temporary buffer
          id<MTLBuffer> tempBuffer = [device newBufferWithBytes:data_ptr
                                                         length:total_size
                                                        options:MTLResourceStorageModeShared];
          if (!tempBuffer) {
            ET_LOG(Error, "aoti_torch_mps_set_arg_tensor: Failed to create temporary buffer");
            return Error::Internal;
          }

          // Use setBuffer instead of setBytes
          [encoder setBuffer:tempBuffer offset:0 atIndex:idx];

          ET_LOG(Debug, "aoti_torch_mps_set_arg_tensor: Successfully set large CPU tensor as buffer at index %u", idx);
        }

        ET_LOG(Debug, "aoti_torch_mps_set_arg_tensor: Successfully set CPU tensor at index %u with size %zu",
               idx, total_size);
      }

      ET_LOG(Debug, "aoti_torch_mps_set_arg_tensor: Completed successfully");
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

  @autoreleasepool {
    try {
      // Cast the opaque handle to our shim structure to access the encoder
      auto kernelFunc = reinterpret_cast<MetalKernelFunctionShim*>(func);
      id<MTLComputeCommandEncoder> encoder = kernelFunc->encoder;

      if (!encoder) {
        ET_LOG(Error, "aoti_torch_mps_set_arg_int: null encoder");
        return Error::InvalidArgument;
      }

      // Set the integer value as bytes
      [encoder setBytes:&val length:sizeof(int64_t) atIndex:idx];

      ET_LOG(Debug, "aoti_torch_mps_set_arg_int: set int64_t value %lld at index %u", val, idx);

      return Error::Ok;

    } catch (const std::exception& e) {
      ET_LOG(Error, "aoti_torch_mps_set_arg_int exception: %s", e.what());
      return Error::Internal;
    } catch (...) {
      ET_LOG(Error, "aoti_torch_mps_set_arg_int: unknown exception");
      return Error::Internal;
    }
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

  ET_LOG(Debug, "aoti_torch_mps_mm_out: Starting with out=%p, self=%p, mat2=%p",
         out, self, mat2);

  if (!out || !self || !mat2) {
    ET_LOG(Error, "aoti_torch_mps_mm_out: null tensor handles");
    return Error::InvalidArgument;
  }

  @autoreleasepool {
    try {
      // Convert AtenTensorHandle to ExecutorTorch tensors
      auto out_tensor = reinterpret_cast<executorch::runtime::etensor::Tensor*>(out);
      auto self_tensor = reinterpret_cast<executorch::runtime::etensor::Tensor*>(self);
      auto mat2_tensor = reinterpret_cast<executorch::runtime::etensor::Tensor*>(mat2);

      ET_LOG(Debug, "aoti_torch_mps_mm_out: Converted tensor handles to ET tensors");

      // Log tensor shapes for debugging
      ET_LOG(Debug, "aoti_torch_mps_mm_out: self shape: [%d, %d], mat2 shape: [%d, %d], out shape: [%d, %d]",
             self_tensor->dim() > 0 ? (int)self_tensor->sizes()[0] : 0,
             self_tensor->dim() > 1 ? (int)self_tensor->sizes()[1] : 0,
             mat2_tensor->dim() > 0 ? (int)mat2_tensor->sizes()[0] : 0,
             mat2_tensor->dim() > 1 ? (int)mat2_tensor->sizes()[1] : 0,
             out_tensor->dim() > 0 ? (int)out_tensor->sizes()[0] : 0,
             out_tensor->dim() > 1 ? (int)out_tensor->sizes()[1] : 0);

      // For now, just zero out the output tensor to get the right shape
      // TODO: Implement actual matrix multiplication: out = self @ mat2

      // Get output data pointer and size
      float* out_data = static_cast<float*>(out_tensor->mutable_data_ptr());
      size_t out_numel = out_tensor->numel();

      if (!out_data) {
        ET_LOG(Error, "aoti_torch_mps_mm_out: null output data pointer");
        return Error::InvalidArgument;
      }

      // Zero out the output tensor
      std::memset(out_data, 0, out_numel * sizeof(float));

      ET_LOG(Debug, "aoti_torch_mps_mm_out: Zeroed output tensor with %zu elements", out_numel);
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

  ET_LOG(Debug, "aoti_torch_mps_convolution: Starting with input=%p, weight=%p, bias=%p, groups=%lld, transposed=%d",
         input, weight, bias, groups, transposed);

  if (!input || !weight || !ret0) {
    ET_LOG(Error, "aoti_torch_mps_convolution: null required handles (input, weight, or ret0)");
    return Error::InvalidArgument;
  }

  @autoreleasepool {
    try {
      // Convert AtenTensorHandle to ExecutorTorch tensors
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

      AtenTensorHandle output_tensor_handle = nullptr;

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

} // extern "C"

// C++ only functions that can use std::function and C++ types
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

// Single dispatch function that handles both cases
AOTITorchError aoti_torch_mps_dispatch(
    AOTIMetalKernelFunctionHandle func,
    c10::ArrayRef<uint64_t> length,
    c10::OptionalArrayRef<uint64_t> groupSize) {

    if (!func) {
        ET_LOG(Error, "aoti_torch_mps_dispatch: null function handle");
        return Error::InvalidArgument;
    }

    try {
        auto* function = reinterpret_cast<ETMetalKernelFunction*>(func);
        function->dispatch(length, groupSize);

        ET_LOG(Debug, "aoti_torch_mps_dispatch: Dispatched function %p with %zu dimensions", function, length.size());
        return Error::Ok;

    } catch (const std::exception& e) {
        ET_LOG(Error, "aoti_torch_mps_dispatch exception: %s", e.what());
        return Error::Internal;
    } catch (...) {
        ET_LOG(Error, "aoti_torch_mps_dispatch: unknown exception");
        return Error::Internal;
    }
}

} // namespace aoti
} // namespace backends
} // namespace executorch
