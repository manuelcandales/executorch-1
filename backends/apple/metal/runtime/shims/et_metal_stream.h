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
#include <vector>
#include <memory>

namespace executorch {
namespace backends {
namespace aoti {

// ExecuTorch Metal Stream - manages command buffers and synchronization
class ETMetalStream {
public:
    ETMetalStream();
    ~ETMetalStream();

    // Get the default stream (singleton)
    static ETMetalStream* getDefaultStream();

    // Command buffer management
    id<MTLCommandBuffer> getCommandBuffer();
    void commitCommandBuffer(id<MTLCommandBuffer> commandBuffer);

    // Command encoder management
    id<MTLComputeCommandEncoder> getComputeCommandEncoder();
    void endEncoding(id<MTLComputeCommandEncoder> encoder);

    // Synchronization
    void synchronize();
    void flush();

    // Stream state
    bool isEmpty() const;

    // Get the underlying command queue
    id<MTLCommandQueue> getCommandQueue() const { return commandQueue_; }

private:
    id<MTLCommandQueue> commandQueue_;
    std::vector<id<MTLCommandBuffer>> activeCommandBuffers_;

    // Singleton instance
    static ETMetalStream* defaultStream_;
};

// Global functions for stream access
ETMetalStream* getCurrentMetalStream();
void setCurrentMetalStream(ETMetalStream* stream);

} // namespace aoti
} // namespace backends
} // namespace executorch
