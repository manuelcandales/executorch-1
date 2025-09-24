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
#include "et_metal.h"
#include "memory.h"
#include <algorithm>

namespace executorch {
namespace backends {
namespace aoti {

// =======================
// Global Variables and Storage
// =======================

// Metal device and command queue globals
static id<MTLDevice> metalDevice = nil;
static id<MTLCommandQueue> metalCommandQueue = nil;

// Global Metal buffer mapping - accessible for MPS shim
std::unordered_map<void*, id<MTLBuffer>> ptr_to_mtl_buffer;

// Global storage to keep shared_ptr alive while raw pointers are used
static std::unordered_map<ETMetalKernelFunction*, std::shared_ptr<ETMetalKernelFunction>> function_storage;
static std::unordered_map<ETMetalShaderLibrary*, std::unique_ptr<ETMetalShaderLibrary>> library_storage;

// Static singleton instance for default stream
ETMetalStream* ETMetalStream::defaultStream_ = nullptr;

// Thread-local current stream
static thread_local ETMetalStream* currentStream_ = nullptr;

// =======================
// Metal Helper Functions (C Interface)
// =======================

extern "C" {

void metal_init_if_needed() {
    if (!metalDevice) {
        @autoreleasepool {
            metalDevice = MTLCreateSystemDefaultDevice();
            if (!metalDevice) {
                ET_LOG(Error, "Failed to create Metal device");
                return;
            }

            metalCommandQueue = [metalDevice newCommandQueue];
            if (!metalCommandQueue) {
                ET_LOG(Error, "Failed to create Metal command queue");
                return;
            }
            ET_LOG(Info, "Metal initialized successfully");
        }
    }
}

void* metal_allocate_buffer(long bytes) {
    metal_init_if_needed();
    if (!metalDevice) {
        ET_LOG(Error, "Failed to initialize Metal device");
        return nullptr;
    }

    @autoreleasepool {
        id<MTLBuffer> buffer = [metalDevice newBufferWithLength:bytes options:MTLResourceStorageModeShared];
        if (!buffer) {
            ET_LOG(Error, "Failed to allocate %ld bytes on Metal device", bytes);
            return nullptr;
        }

        void* ptr = [buffer contents];
        ptr_to_mtl_buffer[ptr] = buffer;

        ET_LOG(Debug, "Allocated %ld bytes on Metal device", bytes);
        return ptr;
    }
}

void metal_cleanup_resources() {
    if (!ptr_to_mtl_buffer.empty()) {
        @autoreleasepool {
            for (auto& pair : ptr_to_mtl_buffer) {
                pair.second = nil;
            }
            ptr_to_mtl_buffer.clear();
        }
    }

    metalCommandQueue = nil;
    metalDevice = nil;
}

bool metal_is_device_pointer(void* ptr) {
    return ptr_to_mtl_buffer.find(ptr) != ptr_to_mtl_buffer.end();
}

int metal_copy_memory(void* dst, const void* src, size_t nbytes, bool src_is_device, bool dst_is_device) {
    if (!src || !dst || nbytes == 0) {
        ET_LOG(Error, "Metal copy: Invalid parameters");
        return -1;
    }

    @autoreleasepool {
        std::memcpy(dst, src, nbytes);

        // For shared memory buffers, synchronization is typically not needed
        // since both CPU and GPU access the same memory space
        if ((src_is_device || dst_is_device) && metalCommandQueue) {
            // Only synchronize if we actually need to ensure GPU operations complete
            id<MTLCommandBuffer> commandBuffer = [metalCommandQueue commandBuffer];
            if (commandBuffer) {
                [commandBuffer commit];
                [commandBuffer waitUntilCompleted];
            }
        }
    }

    ET_LOG(Debug, "Metal memory copy completed: %zu bytes", nbytes);
    return 0;
}

id<MTLDevice> get_metal_device() {
    metal_init_if_needed();
    return metalDevice;
}

id<MTLCommandQueue> get_metal_command_queue() {
    metal_init_if_needed();
    return metalCommandQueue;
}

} // extern "C"

// =======================
// ETMetalShaderLibrary Implementation
// =======================

ETMetalShaderLibrary::ETMetalShaderLibrary(const std::string& source) : shaderSource_(source) {
    compileLibrary();
}

ETMetalShaderLibrary::~ETMetalShaderLibrary() {
    @autoreleasepool {
        if (library_) {
            [library_ release];
            library_ = nil;
        }

        for (auto& pair : pipelineStates_) {
            [pair.second.first release];
            [pair.second.second release];
        }
        pipelineStates_.clear();
    }
}

void ETMetalShaderLibrary::compileLibrary() {
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

std::pair<id<MTLComputePipelineState>, id<MTLFunction>> ETMetalShaderLibrary::getLibraryPipelineState(const std::string& functionName) {
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

        [pipelineState retain];
        [function retain];
        pipelineStates_[functionName] = {pipelineState, function};

        ET_LOG(Debug, "ETMetalShaderLibrary: Created pipeline state for function '%s'", functionName.c_str());
        return {pipelineState, function};
    }
}

std::shared_ptr<ETMetalKernelFunction> ETMetalShaderLibrary::getKernelFunction(const std::string& name) {
    auto pipelineStatePair = getLibraryPipelineState(name);
    if (!pipelineStatePair.first || !pipelineStatePair.second) {
        ET_LOG(Error, "ETMetalShaderLibrary::getKernelFunction: Failed to get pipeline state for '%s'", name.c_str());
        return nullptr;
    }

    return std::make_shared<ETMetalKernelFunction>(pipelineStatePair.first, pipelineStatePair.second);
}

// =======================
// ETMetalKernelFunction Implementation
// =======================

ETMetalKernelFunction::ETMetalKernelFunction(id<MTLComputePipelineState> cps, id<MTLFunction> func)
    : cps_(cps), func_(func), encoder_(nil) {
    if (cps_) [cps_ retain];
    if (func_) [func_ retain];
}

ETMetalKernelFunction::~ETMetalKernelFunction() {
    @autoreleasepool {
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
}

void ETMetalKernelFunction::startEncoding() {
    @autoreleasepool {
        if (encoder_) {
            [encoder_ release];
            encoder_ = nil;
        }

        ETMetalStream* stream = getCurrentMetalStream();
        encoder_ = stream->getComputeCommandEncoder();
        if (!encoder_) {
            ET_LOG(Error, "ETMetalKernelFunction: Failed to get encoder from stream");
            return;
        }

        [encoder_ retain];
        [encoder_ setComputePipelineState:cps_];

        ET_LOG(Debug, "ETMetalKernelFunction: Started encoding");
    }
}

void ETMetalKernelFunction::setArg(unsigned idx, const executorch::runtime::etensor::Tensor& tensor) {
    if (!encoder_) {
        ET_LOG(Error, "ETMetalKernelFunction::setArg: No active encoder");
        return;
    }

    void* data_ptr = tensor.mutable_data_ptr();
    size_t totalSize = tensor.numel() * tensor.element_size();

    auto it = ptr_to_mtl_buffer.find(data_ptr);
    if (it != ptr_to_mtl_buffer.end()) {
        // Use existing Metal buffer
        id<MTLBuffer> mtlBuffer = it->second;
        [encoder_ setBuffer:mtlBuffer offset:0 atIndex:idx];
        ET_LOG(Debug, "ETMetalKernelFunction::setArg: Set Metal buffer at index %u (size: %zu)", idx, totalSize);
    } else {
        // Handle CPU tensor data
        if (totalSize <= 4096) {
            // Use setBytes for small data (more efficient)
            [encoder_ setBytes:data_ptr length:totalSize atIndex:idx];
            ET_LOG(Debug, "ETMetalKernelFunction::setArg: Set CPU tensor via setBytes at index %u (size: %zu)", idx, totalSize);
        } else {
            // Create temporary buffer for large data (should be rare)
            @autoreleasepool {
                id<MTLDevice> device = get_metal_device();
                if (device) {
                    id<MTLBuffer> tempBuffer = [device newBufferWithBytes:data_ptr
                                                                   length:totalSize
                                                                  options:MTLResourceStorageModeShared];
                    if (tempBuffer) {
                        [encoder_ setBuffer:tempBuffer offset:0 atIndex:idx];
                        ET_LOG(Debug, "ETMetalKernelFunction::setArg: Set large CPU tensor via temporary buffer at index %u (size: %zu)", idx, totalSize);
                    } else {
                        ET_LOG(Error, "ETMetalKernelFunction::setArg: Failed to create temporary buffer for index %u", idx);
                    }
                } else {
                    ET_LOG(Error, "ETMetalKernelFunction::setArg: No Metal device available for index %u", idx);
                }
            }
        }
    }
}

void ETMetalKernelFunction::setArg(unsigned idx, int64_t val) {
    if (!encoder_) {
        ET_LOG(Error, "ETMetalKernelFunction::setArg: No active encoder");
        return;
    }

    [encoder_ setBytes:&val length:sizeof(int64_t) atIndex:idx];
    ET_LOG(Debug, "ETMetalKernelFunction::setArg: Set int64_t value %lld at index %u", val, idx);
}

void ETMetalKernelFunction::dispatchSingle(uint64_t length) {
    if (!encoder_) {
        ET_LOG(Error, "ETMetalKernelFunction::dispatchSingle: No active encoder");
        return;
    }

    const auto maxThreadsPerGroup = static_cast<uint64_t>([cps_ maxTotalThreadsPerThreadgroup]);
    uint64_t actualGroupSize = std::min(maxThreadsPerGroup, length);

    auto size = MTLSizeMake(length, 1, 1);
    auto threadGroupSize = MTLSizeMake(actualGroupSize, 1, 1);

    [encoder_ dispatchThreads:size threadsPerThreadgroup:threadGroupSize];
    ET_LOG(Debug, "ETMetalKernelFunction::dispatchSingle: Dispatched with length %llu, group size %llu", length, actualGroupSize);

    endEncoding();
}

void ETMetalKernelFunction::dispatchSingleWithGroupSize(uint64_t length, uint64_t group_size) {
    if (!encoder_) {
        ET_LOG(Error, "ETMetalKernelFunction::dispatchSingleWithGroupSize: No active encoder");
        return;
    }

    const auto maxThreadsPerGroup = static_cast<uint64_t>([cps_ maxTotalThreadsPerThreadgroup]);
    uint64_t actualGroupSize = group_size > 0 ? std::min(group_size, maxThreadsPerGroup) : std::min(maxThreadsPerGroup, length);

    auto size = MTLSizeMake(length, 1, 1);
    auto threadGroupSize = MTLSizeMake(actualGroupSize, 1, 1);

    [encoder_ dispatchThreads:size threadsPerThreadgroup:threadGroupSize];
    ET_LOG(Debug, "ETMetalKernelFunction::dispatchSingleWithGroupSize: Dispatched with length %llu, group size %llu", length, actualGroupSize);

    endEncoding();
}

void ETMetalKernelFunction::dispatchArray(const uint64_t* length, size_t length_size) {
    if (!encoder_) {
        ET_LOG(Error, "ETMetalKernelFunction::dispatchArray: No active encoder");
        return;
    }

    if (!length || length_size == 0) {
        ET_LOG(Error, "ETMetalKernelFunction::dispatchArray: Invalid length array");
        return;
    }

    const auto maxThreadsPerGroup = static_cast<uint64_t>([cps_ maxTotalThreadsPerThreadgroup]);

    MTLSize size, threadGroupSize;

    if (length_size == 1) {
        size = MTLSizeMake(length[0], 1, 1);
        uint64_t actualGroupSize = std::min(maxThreadsPerGroup, length[0]);
        threadGroupSize = MTLSizeMake(actualGroupSize, 1, 1);
    } else if (length_size == 2) {
        size = MTLSizeMake(length[0], length[1], 1);
        uint64_t groupX = std::min(static_cast<uint64_t>(32), length[0]);
        uint64_t groupY = maxThreadsPerGroup / groupX;
        threadGroupSize = MTLSizeMake(groupX, groupY, 1);
    } else {
        size = MTLSizeMake(length[0], length[1], length_size > 2 ? length[2] : 1);
        uint64_t groupX = std::min(static_cast<uint64_t>(8), length[0]);
        uint64_t groupY = std::min(static_cast<uint64_t>(8), length[1]);
        uint64_t groupZ = maxThreadsPerGroup / (groupX * groupY);
        threadGroupSize = MTLSizeMake(groupX, groupY, groupZ);
    }

    [encoder_ dispatchThreads:size threadsPerThreadgroup:threadGroupSize];
    ET_LOG(Debug, "ETMetalKernelFunction::dispatchArray: Dispatched %zuD with size [%lu, %lu, %lu], group [%lu, %lu, %lu]",
           length_size, size.width, size.height, size.depth,
           threadGroupSize.width, threadGroupSize.height, threadGroupSize.depth);

    endEncoding();
}

void ETMetalKernelFunction::dispatchArrayWithGroupSize(const uint64_t* length, size_t length_size,
                                                      const uint64_t* group_size, size_t group_size_size) {
    if (!encoder_) {
        ET_LOG(Error, "ETMetalKernelFunction::dispatchArrayWithGroupSize: No active encoder");
        return;
    }

    if (!length || length_size == 0) {
        ET_LOG(Error, "ETMetalKernelFunction::dispatchArrayWithGroupSize: Invalid length array");
        return;
    }

    const auto maxThreadsPerGroup = static_cast<uint64_t>([cps_ maxTotalThreadsPerThreadgroup]);

    MTLSize size, threadGroupSize;

    if (length_size == 1) {
        size = MTLSizeMake(length[0], 1, 1);
        uint64_t actualGroupSize = maxThreadsPerGroup;
        if (group_size && group_size_size > 0) {
            actualGroupSize = std::min(maxThreadsPerGroup, group_size[0]);
        }
        threadGroupSize = MTLSizeMake(actualGroupSize, 1, 1);
    } else if (length_size == 2) {
        size = MTLSizeMake(length[0], length[1], 1);
        uint64_t groupX = std::min(static_cast<uint64_t>(32), length[0]);
        uint64_t groupY = maxThreadsPerGroup / groupX;
        if (group_size && group_size_size >= 2) {
            groupX = std::min(static_cast<uint64_t>(group_size[0]), length[0]);
            groupY = std::min(static_cast<uint64_t>(group_size[1]), length[1]);
        }
        threadGroupSize = MTLSizeMake(groupX, groupY, 1);
    } else {
        size = MTLSizeMake(length[0], length[1], length_size > 2 ? length[2] : 1);
        uint64_t groupX = std::min(static_cast<uint64_t>(8), length[0]);
        uint64_t groupY = std::min(static_cast<uint64_t>(8), length[1]);
        uint64_t groupZ = maxThreadsPerGroup / (groupX * groupY);
        if (group_size && group_size_size >= 3) {
            groupX = std::min(static_cast<uint64_t>(group_size[0]), length[0]);
            groupY = std::min(static_cast<uint64_t>(group_size[1]), length[1]);
            groupZ = std::min(static_cast<uint64_t>(group_size[2]), length_size > 2 ? length[2] : 1);
        }
        threadGroupSize = MTLSizeMake(groupX, groupY, groupZ);
    }

    [encoder_ dispatchThreads:size threadsPerThreadgroup:threadGroupSize];
    ET_LOG(Debug, "ETMetalKernelFunction::dispatchArrayWithGroupSize: Dispatched %zuD with size [%lu, %lu, %lu], group [%lu, %lu, %lu]",
           length_size, size.width, size.height, size.depth,
           threadGroupSize.width, threadGroupSize.height, threadGroupSize.depth);

    endEncoding();
}

void ETMetalKernelFunction::endEncoding() {
    @autoreleasepool {
        if (!encoder_) {
            ET_LOG(Error, "ETMetalKernelFunction::endEncoding: No active encoder");
            return;
        }

        ETMetalStream* stream = getCurrentMetalStream();
        stream->endEncoding(encoder_);

        [encoder_ release];
        encoder_ = nil;

        ET_LOG(Debug, "ETMetalKernelFunction::endEncoding: Ended encoding");
    }
}

void ETMetalKernelFunction::runCommandBlock(std::function<void(void)> f) {
    @autoreleasepool {
        if (!encoder_) {
            ET_LOG(Error, "ETMetalKernelFunction::runCommandBlock: No active encoder");
            return;
        }

        f();
        endEncoding();

        ET_LOG(Debug, "ETMetalKernelFunction::runCommandBlock: Executed command block");
    }
}

// =======================
// ETMetalStream Implementation
// =======================

ETMetalStream::ETMetalStream() {
    @autoreleasepool {
        commandQueue_ = get_metal_command_queue();
        if (commandQueue_) {
            [commandQueue_ retain];
            ET_LOG(Debug, "ETMetalStream: Created stream with command queue %p", commandQueue_);
        } else {
            ET_LOG(Error, "ETMetalStream: Failed to get Metal command queue");
        }
    }
}

ETMetalStream::~ETMetalStream() {
    @autoreleasepool {
        synchronize();

        if (commandQueue_) {
            [commandQueue_ release];
            commandQueue_ = nil;
        }

        ET_LOG(Debug, "ETMetalStream: Destroyed stream");
    }
}

ETMetalStream* ETMetalStream::getDefaultStream() {
    if (!defaultStream_) {
        defaultStream_ = new ETMetalStream();
    }
    return defaultStream_;
}

id<MTLCommandBuffer> ETMetalStream::getCommandBuffer() {
    @autoreleasepool {
        if (!commandQueue_) {
            ET_LOG(Error, "ETMetalStream::getCommandBuffer: No command queue available");
            return nil;
        }

        id<MTLCommandBuffer> commandBuffer = [commandQueue_ commandBuffer];
        if (!commandBuffer) {
            ET_LOG(Error, "ETMetalStream::getCommandBuffer: Failed to create command buffer");
            return nil;
        }

        activeCommandBuffers_.push_back(commandBuffer);
        [commandBuffer retain];

        ET_LOG(Debug, "ETMetalStream::getCommandBuffer: Created command buffer %p", commandBuffer);
        return commandBuffer;
    }
}

void ETMetalStream::commitCommandBuffer(id<MTLCommandBuffer> commandBuffer) {
    @autoreleasepool {
        if (!commandBuffer) {
            ET_LOG(Error, "ETMetalStream::commitCommandBuffer: null command buffer");
            return;
        }

        [commandBuffer commit];
        ET_LOG(Debug, "ETMetalStream::commitCommandBuffer: Committed command buffer %p", commandBuffer);
    }
}

id<MTLComputeCommandEncoder> ETMetalStream::getComputeCommandEncoder() {
    @autoreleasepool {
        id<MTLCommandBuffer> commandBuffer = getCommandBuffer();
        if (!commandBuffer) {
            ET_LOG(Error, "ETMetalStream::getComputeCommandEncoder: Failed to get command buffer");
            return nil;
        }

        id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];
        if (!encoder) {
            ET_LOG(Error, "ETMetalStream::getComputeCommandEncoder: Failed to create compute command encoder");
            return nil;
        }

        ET_LOG(Debug, "ETMetalStream::getComputeCommandEncoder: Created encoder %p from command buffer %p", encoder, commandBuffer);
        return encoder;
    }
}

void ETMetalStream::endEncoding(id<MTLComputeCommandEncoder> encoder) {
    @autoreleasepool {
        if (!encoder) {
            ET_LOG(Error, "ETMetalStream::endEncoding: null encoder");
            return;
        }

        [encoder endEncoding];

        id<MTLCommandBuffer> commandBuffer = [encoder commandBuffer];
        if (commandBuffer) {
            commitCommandBuffer(commandBuffer);
        }

        ET_LOG(Debug, "ETMetalStream::endEncoding: Ended encoding for encoder %p", encoder);
    }
}

void ETMetalStream::synchronize() {
    @autoreleasepool {
        ET_LOG(Debug, "ETMetalStream::synchronize: Synchronizing %zu active command buffers", activeCommandBuffers_.size());

        for (auto& commandBuffer : activeCommandBuffers_) {
            if (commandBuffer) {
                [commandBuffer waitUntilCompleted];
                [commandBuffer release];
            }
        }

        activeCommandBuffers_.clear();

        ET_LOG(Debug, "ETMetalStream::synchronize: Synchronization complete");
    }
}

void ETMetalStream::flush() {
    @autoreleasepool {
        // Clean up completed command buffers to prevent memory growth
        auto it = std::remove_if(activeCommandBuffers_.begin(), activeCommandBuffers_.end(),
            [](id<MTLCommandBuffer> commandBuffer) {
                if (!commandBuffer) {
                    return true; // Remove null entries
                }

                MTLCommandBufferStatus status = [commandBuffer status];
                if (status == MTLCommandBufferStatusCompleted ||
                    status == MTLCommandBufferStatusError) {
                    [commandBuffer release];
                    return true;
                }
                return false;
            });

        size_t removedCount = activeCommandBuffers_.end() - it;
        activeCommandBuffers_.erase(it, activeCommandBuffers_.end());

        if (removedCount > 0) {
            ET_LOG(Debug, "ETMetalStream::flush: Cleaned up %zu completed command buffers", removedCount);
        }
    }
}

bool ETMetalStream::isEmpty() const {
    return activeCommandBuffers_.empty();
}

// =======================
// Global Storage Management Functions
// =======================

void storeFunctionHandle(ETMetalKernelFunction* raw_function, std::shared_ptr<ETMetalKernelFunction> function_shared_ptr) {
    function_storage[raw_function] = function_shared_ptr;
}

void storeLibraryHandle(ETMetalShaderLibrary* raw_library, std::unique_ptr<ETMetalShaderLibrary> library) {
    library_storage[raw_library] = std::move(library);
}

bool removeFunctionHandle(ETMetalKernelFunction* raw_function) {
    auto it = function_storage.find(raw_function);
    if (it != function_storage.end()) {
        function_storage.erase(it);
        return true;
    }
    return false;
}

bool removeLibraryHandle(ETMetalShaderLibrary* raw_library) {
    auto it = library_storage.find(raw_library);
    if (it != library_storage.end()) {
        library_storage.erase(it);
        return true;
    }
    return false;
}

// =======================
// Global Stream Access Functions
// =======================

ETMetalStream* getCurrentMetalStream() {
    if (!currentStream_) {
        currentStream_ = ETMetalStream::getDefaultStream();
    }
    return currentStream_;
}

void setCurrentMetalStream(ETMetalStream* stream) {
    currentStream_ = stream;
}

} // namespace aoti
} // namespace backends
} // namespace executorch
