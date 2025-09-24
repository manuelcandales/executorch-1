/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#import <Metal/Metal.h>
#import <Foundation/Foundation.h>
#include <executorch/runtime/platform/log.h>
#include "et_metal_stream.h"
#include "metal_helper.h"
#include <algorithm>

namespace executorch {
namespace backends {
namespace aoti {

// Static singleton instance
ETMetalStream* ETMetalStream::defaultStream_ = nullptr;

// Thread-local current stream
static thread_local ETMetalStream* currentStream_ = nullptr;

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
        // Synchronize to ensure all command buffers complete before destruction
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

        // Add to active command buffers for tracking
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

        // Get the command buffer from the encoder and commit it
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

        // Wait for all active command buffers to complete
        for (auto& commandBuffer : activeCommandBuffers_) {
            if (commandBuffer) {
                [commandBuffer waitUntilCompleted];
                [commandBuffer release];
            }
        }

        // Clear the list of active command buffers
        activeCommandBuffers_.clear();

        ET_LOG(Debug, "ETMetalStream::synchronize: Synchronization complete");
    }
}

void ETMetalStream::flush() {
    @autoreleasepool {
        // For Metal, flushing means committing any uncommitted command buffers
        // Since we commit immediately in commitCommandBuffer, this is mostly a no-op
        // But we can clean up any completed command buffers

        auto it = std::remove_if(activeCommandBuffers_.begin(), activeCommandBuffers_.end(),
            [](id<MTLCommandBuffer> commandBuffer) {
                if (commandBuffer && [commandBuffer status] == MTLCommandBufferStatusCompleted) {
                    [commandBuffer release];
                    return true;
                }
                return false;
            });

        size_t removedCount = activeCommandBuffers_.end() - it;
        activeCommandBuffers_.erase(it, activeCommandBuffers_.end());

        ET_LOG(Debug, "ETMetalStream::flush: Cleaned up %zu completed command buffers", removedCount);
    }
}

bool ETMetalStream::isEmpty() const {
    return activeCommandBuffers_.empty();
}

// Global functions for stream access
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
