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

// ExecuTorch MetalShaderLibrary equivalent - simplified version of PyTorch's implementation
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

// ExecuTorch MetalKernelFunction equivalent - simplified version of PyTorch's implementation
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

private:
    id<MTLComputePipelineState> cps_;
    id<MTLFunction> func_;
    id<MTLComputeCommandEncoder> encoder_;
};

// Global storage management functions
void storeFunctionHandle(ETMetalKernelFunction* raw_function, std::shared_ptr<ETMetalKernelFunction> function_shared_ptr);
void storeLibraryHandle(ETMetalShaderLibrary* raw_library, std::unique_ptr<ETMetalShaderLibrary> library);
bool removeFunctionHandle(ETMetalKernelFunction* raw_function);
bool removeLibraryHandle(ETMetalShaderLibrary* raw_library);

} // namespace aoti
} // namespace backends
} // namespace executorch
