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
#include "et_metal_shim.h"
#include "et_metal_stream.h"
#include "metal_helper.h"
#include "memory.h"

namespace executorch {
namespace backends {
namespace aoti {

// Global storage to keep shared_ptr alive while raw pointers are used (matching PyTorch pattern)
static std::unordered_map<ETMetalKernelFunction*, std::shared_ptr<ETMetalKernelFunction>> function_storage;
static std::unordered_map<ETMetalShaderLibrary*, std::unique_ptr<ETMetalShaderLibrary>> library_storage;

// ETMetalShaderLibrary Implementation
ETMetalShaderLibrary::ETMetalShaderLibrary(const std::string& source) : shaderSource_(source) {
    compileLibrary();
}

ETMetalShaderLibrary::~ETMetalShaderLibrary() {
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

        // Retain objects and store in cache
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

// ETMetalKernelFunction Implementation
ETMetalKernelFunction::ETMetalKernelFunction(id<MTLComputePipelineState> cps, id<MTLFunction> func)
    : cps_(cps), func_(func), encoder_(nil) {
    if (cps_) [cps_ retain];
    if (func_) [func_ retain];
}

ETMetalKernelFunction::~ETMetalKernelFunction() {
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

void ETMetalKernelFunction::startEncoding() {
    @autoreleasepool {
        if (encoder_) {
            [encoder_ release];
            encoder_ = nil;
        }

        // Get encoder from the current Metal stream
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

    // End encoding after dispatch
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

    // End encoding after dispatch
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

    // Handle 1D, 2D, and 3D dispatch
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
    } else { // 3D or higher - treat as 3D
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

    // End encoding after dispatch
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

    // Handle 1D, 2D, and 3D dispatch
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
    } else { // 3D or higher - treat as 3D
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

    // End encoding after dispatch
    endEncoding();
}

void ETMetalKernelFunction::endEncoding() {
    @autoreleasepool {
        if (!encoder_) {
            ET_LOG(Error, "ETMetalKernelFunction::endEncoding: No active encoder");
            return;
        }

        // Use the stream to properly end encoding and commit
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

        // Execute the command block
        f();

        // End encoding after command block execution
        endEncoding();

        ET_LOG(Debug, "ETMetalKernelFunction::runCommandBlock: Executed command block");
    }
}

// Global storage management functions
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

} // namespace aoti
} // namespace backends
} // namespace executorch
