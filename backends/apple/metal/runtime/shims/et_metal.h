/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#import <Metal/Metal.h>
#import <Foundation/Foundation.h>
#include <unordered_map>
#include <memory>
#include <string>
#include <functional>
#include <vector>
#include <dispatch/dispatch.h>

namespace executorch {
namespace runtime {
namespace etensor {
class Tensor;
}
}
}

namespace executorch {
namespace backends {
namespace aoti {

// Forward declarations
class ETMetalKernelFunction;
class ETMetalStream;

// =======================
// SyncType - Metal synchronization options
// =======================
enum class SyncType {
    NONE,                // no commit to command buffer
    COMMIT,              // commit and flush the command buffer
    COMMIT_AND_WAIT,     // flush and wait for command buffer execution to finish
    COMMIT_AND_CONTINUE, // commit and continue with a new underlying command buffer
    COMMIT_ADAPTIVE,     // commit adaptively based on available memory
};

// =======================
// ETMetalShaderLibrary - ExecuTorch Metal shader library management
// =======================
class ETMetalShaderLibrary {
public:
    ETMetalShaderLibrary(const std::string& source);
    ~ETMetalShaderLibrary();

    std::shared_ptr<ETMetalKernelFunction> getKernelFunction(const std::string& name);

private:
    void compileLibrary();
    std::pair<id<MTLComputePipelineState>, id<MTLFunction>> getLibraryPipelineState(const std::string& functionName);

    friend class ETMetalKernelFunction;

    std::string shaderSource_;
    id<MTLLibrary> library_ = nil;
    std::unordered_map<std::string, std::pair<id<MTLComputePipelineState>, id<MTLFunction>>> pipelineStates_;
};

// =======================
// ETMetalKernelFunction - ExecuTorch Metal kernel function execution
// =======================
class ETMetalKernelFunction {
public:
    ETMetalKernelFunction(id<MTLComputePipelineState> cps, id<MTLFunction> func);
    ~ETMetalKernelFunction();

    void startEncoding();
    void setArg(unsigned idx, const executorch::runtime::etensor::Tensor& tensor);
    void setArg(unsigned idx, int64_t val);

    void dispatchSingle(uint64_t length);
    void dispatchSingleWithGroupSize(uint64_t length, uint64_t group_size);
    void dispatchArray(const uint64_t* length, size_t length_size);
    void dispatchArrayWithGroupSize(const uint64_t* length, size_t length_size,
                                   const uint64_t* group_size, size_t group_size_size);

    void runCommandBlock(std::function<void(void)> f);
    void endEncoding();

private:
    id<MTLComputePipelineState> cps_;
    id<MTLFunction> func_;
    id<MTLComputeCommandEncoder> encoder_;
};

// =======================
// ETMetalStream - Metal command buffer and synchronization management
// =======================
class ETMetalStream {
public:
    ETMetalStream();
    ~ETMetalStream();

    // Get the default stream (singleton)
    static ETMetalStream* getDefaultStream();

    // Device and queue access
    id<MTLDevice> device() const { return device_; }
    id<MTLCommandQueue> commandQueue() const { return commandQueue_; }
    dispatch_queue_t queue() const { return serialQueue_; }

    // Command buffer management with lazy creation
    id<MTLCommandBuffer> commandBuffer();
    id<MTLComputeCommandEncoder> commandEncoder();

    // Synchronization methods
    void synchronize(SyncType syncType = SyncType::COMMIT_AND_WAIT);
    void endKernelCoalescing();

    // Command buffer lifecycle management
    void commitCommandBuffer(id<MTLCommandBuffer> commandBuffer);
    void flush();
    bool isEmpty() const;

    // Encoder management
    id<MTLComputeCommandEncoder> getComputeCommandEncoder();
    void endEncoding(id<MTLComputeCommandEncoder> encoder);

    // Memory operations
    void fill(id<MTLBuffer> buffer, uint8_t value, size_t length, size_t offset, SyncType syncType = SyncType::NONE);
    void copy(id<MTLBuffer> srcBuffer, id<MTLBuffer> dstBuffer, size_t length,
             size_t srcOffset, size_t dstOffset, SyncType syncType = SyncType::NONE);

private:
    // Private members
    id<MTLDevice> device_;
    id<MTLCommandQueue> commandQueue_;
    id<MTLCommandBuffer> commandBuffer_;
    id<MTLCommandBuffer> prevCommandBuffer_;  // For commit-and-continue pattern
    id<MTLComputeCommandEncoder> commandEncoder_;
    dispatch_queue_t serialQueue_;  // For thread safety

    // Configuration
    bool enableCommitAndContinue_;

    // Legacy compatibility
    std::vector<id<MTLCommandBuffer>> activeCommandBuffers_;

    // Private synchronization methods
    void commit();
    void commitAndWait();
    void commitAndContinue();

    // Singleton instance
    static ETMetalStream* defaultStream_;
};

// =======================
// Global storage management functions
// =======================
void storeFunctionHandle(ETMetalKernelFunction* raw_function, std::shared_ptr<ETMetalKernelFunction> function_shared_ptr);
void storeLibraryHandle(ETMetalShaderLibrary* raw_library, std::unique_ptr<ETMetalShaderLibrary> library);
bool removeFunctionHandle(ETMetalKernelFunction* raw_function);
bool removeLibraryHandle(ETMetalShaderLibrary* raw_library);

// =======================
// Global stream access functions
// =======================
ETMetalStream* getCurrentMetalStream();
void setCurrentMetalStream(ETMetalStream* stream);

// =======================
// Metal helper functions (C interface)
// =======================
#ifdef __cplusplus
extern "C" {
#endif

// Metal initialization and management
void metal_init_if_needed();
void* metal_allocate_buffer(long bytes);
void metal_cleanup_resources();

// Memory management functions for Metal
bool metal_is_device_pointer(void* ptr);
int metal_copy_memory(void* dst, const void* src, size_t nbytes, bool src_is_device, bool dst_is_device);

#ifdef __OBJC__
// Helper functions to access Metal objects (Objective-C only)
id<MTLDevice> get_metal_device();
id<MTLCommandQueue> get_metal_command_queue();
#endif

#ifdef __cplusplus
}

// C++ only - expose the Metal buffer mapping
#ifdef __OBJC__
extern std::unordered_map<void*, id<MTLBuffer>> ptr_to_mtl_buffer;
#endif

#endif

} // namespace aoti
} // namespace backends
} // namespace executorch
