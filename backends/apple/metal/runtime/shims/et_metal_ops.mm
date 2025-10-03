/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#import <Metal/Metal.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>
#import <MetalPerformanceShadersGraph/MetalPerformanceShadersGraph.h>
#import <Foundation/Foundation.h>
#include <executorch/runtime/platform/log.h>
#include <executorch/runtime/core/exec_aten/exec_aten.h>
#include "et_metal_ops.h"
#include "et_metal.h"
#include "utils.h"
#include "memory.h"
#include <functional>
#include <unordered_map>

namespace executorch {
namespace backends {
namespace aoti {

// Forward declaration of dispatch_sync_with_rethrow from et_metal.mm
void dispatch_sync_with_rethrow(dispatch_queue_t queue, void (^block)());

// Declare the global mapping from et_metal.mm
extern std::unordered_map<void*, id<MTLBuffer>> ptr_to_mtl_buffer;

extern "C" {

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

  // Use the same dispatch pattern as aoti_torch_mps_run_command_block for consistent synchronization
  ETMetalStream* stream = getCurrentMetalStream();
  if (!stream) {
    ET_LOG(Error, "aoti_torch_mps_mm_out: Failed to get current Metal stream");
    return Error::Internal;
  }

  try {
    // Use dispatch_sync_with_rethrow to match custom kernel synchronization behavior
    dispatch_sync_with_rethrow(stream->queue(), ^() {
      @autoreleasepool {
        // Convert AOTITensorHandle to ExecutorTorch tensors
        auto out_tensor = reinterpret_cast<executorch::runtime::etensor::Tensor*>(out);
        auto self_tensor = reinterpret_cast<executorch::runtime::etensor::Tensor*>(self);
        auto mat2_tensor = reinterpret_cast<executorch::runtime::etensor::Tensor*>(mat2);

        ET_LOG(Debug, "aoti_torch_mps_mm_out: Converted tensor handles to ET tensors");

        // Validate tensor dimensions
        if (self_tensor->dim() != 2 || mat2_tensor->dim() != 2) {
          std::string error_msg = "aoti_torch_mps_mm_out: tensors must be 2-D, got " +
                                 std::to_string(self_tensor->dim()) + " and " +
                                 std::to_string(mat2_tensor->dim());
          ET_LOG(Error, "%s", error_msg.c_str());
          throw std::runtime_error(error_msg);
        }

        int64_t M = self_tensor->sizes()[0];  // rows of self
        int64_t K = self_tensor->sizes()[1];  // cols of self / rows of mat2
        int64_t N = mat2_tensor->sizes()[1];  // cols of mat2

        // Check matrix multiplication compatibility
        if (self_tensor->sizes()[1] != mat2_tensor->sizes()[0]) {
          std::string error_msg = "aoti_torch_mps_mm_out: incompatible matrix sizes for mm (" +
                                 std::to_string(M) + "x" + std::to_string(K) + " and " +
                                 std::to_string(mat2_tensor->sizes()[0]) + "x" + std::to_string(N) + ")";
          ET_LOG(Error, "%s", error_msg.c_str());
          throw std::runtime_error(error_msg);
        }

        // Log tensor shapes for debugging
        ET_LOG(Debug, "aoti_torch_mps_mm_out: self shape: [%d, %d], mat2 shape: [%d, %d], out shape: [%d, %d]",
               (int)M, (int)K, (int)mat2_tensor->sizes()[0], (int)N,
               out_tensor->dim() > 0 ? (int)out_tensor->sizes()[0] : 0,
               out_tensor->dim() > 1 ? (int)out_tensor->sizes()[1] : 0);

        // Get Metal device
        id<MTLDevice> device = get_metal_device();
        if (!device) {
          ET_LOG(Error, "aoti_torch_mps_mm_out: Failed to get Metal device");
          throw std::runtime_error("Failed to get Metal device");
        }

        // Get Metal buffers from tensors using the global mapping
        void* self_data_ptr = self_tensor->mutable_data_ptr();
        void* mat2_data_ptr = mat2_tensor->mutable_data_ptr();
        void* out_data_ptr = out_tensor->mutable_data_ptr();

        // Look up Metal buffers from the global mapping
        auto self_it = ptr_to_mtl_buffer.find(self_data_ptr);
        auto mat2_it = ptr_to_mtl_buffer.find(mat2_data_ptr);
        auto out_it = ptr_to_mtl_buffer.find(out_data_ptr);

        if (self_it == ptr_to_mtl_buffer.end()) {
          ET_LOG(Error, "aoti_torch_mps_mm_out: self tensor not found in Metal buffer mapping");
          throw std::runtime_error("self tensor not found in Metal buffer mapping");
        }
        if (mat2_it == ptr_to_mtl_buffer.end()) {
          ET_LOG(Error, "aoti_torch_mps_mm_out: mat2 tensor not found in Metal buffer mapping");
          throw std::runtime_error("mat2 tensor not found in Metal buffer mapping");
        }
        if (out_it == ptr_to_mtl_buffer.end()) {
          ET_LOG(Error, "aoti_torch_mps_mm_out: out tensor not found in Metal buffer mapping");
          throw std::runtime_error("out tensor not found in Metal buffer mapping");
        }

        id<MTLBuffer> self_buffer = self_it->second;
        id<MTLBuffer> mat2_buffer = mat2_it->second;
        id<MTLBuffer> out_buffer = out_it->second;

        ET_LOG(Debug, "aoti_torch_mps_mm_out: Using existing Metal buffers - self=%p, mat2=%p, out=%p",
               self_buffer, mat2_buffer, out_buffer);

        // End any existing kernel coalescing to ensure a clean state for MPS
        stream->endKernelCoalescing();

        // Get command buffer from stream (stream manages lifecycle)
        id<MTLCommandBuffer> commandBuffer = stream->commandBuffer();
        if (!commandBuffer) {
          ET_LOG(Error, "aoti_torch_mps_mm_out: Failed to get command buffer from stream");
          throw std::runtime_error("Failed to get command buffer from stream");
        }

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

        // Encode the matrix multiplication
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
      }
    });

    return Error::Ok;

  } catch (const std::exception& e) {
    ET_LOG(Error, "aoti_torch_mps_mm_out exception: %s", e.what());
    return Error::Internal;
  } catch (...) {
    ET_LOG(Error, "aoti_torch_mps_mm_out: unknown exception");
    return Error::Internal;
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

// Helper function to ensure 4D tensors (port from PyTorch)
static std::tuple<AOTITensorHandle, bool> ensure_4d_et(AOTITensorHandle x) {
  auto* x_tensor = reinterpret_cast<executorch::runtime::etensor::Tensor*>(x);

  if (x_tensor->dim() == 3) {
    // Need to unsqueeze - for now, return as is and handle in the main function
    return {x, true};
  } else if (x_tensor->dim() > 4) {
    // Need to view as 4D - for now, return as is and handle in the main function
    return {x, true};
  } else {
    return {x, false};
  }
}

AOTITorchError aoti_torch_mps__scaled_dot_product_attention_math_for_mps(
    AOTITensorHandle query,
    AOTITensorHandle key,
    AOTITensorHandle value,
    AOTITensorHandle* attn_mask,
    double dropout_p,
    int32_t is_causal,
    AOTITensorHandle* dropout_mask,
    double* scale,
    AOTITensorHandle* ret0,
    AOTITensorHandle* ret1) {

  ET_LOG(Debug, "aoti_torch_mps__scaled_dot_product_attention_math_for_mps: Starting with MPSGraph implementation");

  if (!query || !key || !value || !ret0 || !ret1) {
    ET_LOG(Error, "aoti_torch_mps__scaled_dot_product_attention_math_for_mps: null required tensor handles");
    return Error::InvalidArgument;
  }

  // Use the same dispatch pattern as other MPS operations for consistent synchronization
  ETMetalStream* stream = getCurrentMetalStream();
  if (!stream) {
    ET_LOG(Error, "aoti_torch_mps__scaled_dot_product_attention_math_for_mps: Failed to get current Metal stream");
    return Error::Internal;
  }

  try {
    @autoreleasepool {
      // Convert AOTITensorHandle to ExecutorTorch tensors
      auto* query_tensor = reinterpret_cast<executorch::runtime::etensor::Tensor*>(query);
      auto* key_tensor = reinterpret_cast<executorch::runtime::etensor::Tensor*>(key);
      auto* value_tensor = reinterpret_cast<executorch::runtime::etensor::Tensor*>(value);

      ET_LOG(Debug, "aoti_torch_mps__scaled_dot_product_attention_math_for_mps: Converted tensor handles to ET tensors");

        // Validate tensor dimensions
        if (query_tensor->dim() < 3 || key_tensor->dim() < 3 || value_tensor->dim() < 3) {
          std::string error_msg = "aoti_torch_mps__scaled_dot_product_attention_math_for_mps: tensors must be at least 3-D, got " +
                                 std::to_string(query_tensor->dim()) + ", " +
                                 std::to_string(key_tensor->dim()) + ", " +
                                 std::to_string(value_tensor->dim());
          ET_LOG(Error, "%s", error_msg.c_str());
          throw std::runtime_error(error_msg);
        }

        // Get tensor dimensions (assuming [batch, num_heads, seq_len, head_dim] format)
        int64_t batchSize = query_tensor->sizes()[0];
        int64_t num_heads = query_tensor->sizes()[1];
        int64_t qSize = query_tensor->sizes()[2];
        int64_t headSize = query_tensor->sizes()[3];
        int64_t kvSeqLength = key_tensor->sizes()[2];

        ET_LOG(Debug, "aoti_torch_mps__scaled_dot_product_attention_math_for_mps: batchSize=%lld, num_heads=%lld, qSize=%lld, headSize=%lld, kvSeqLength=%lld",
               batchSize, num_heads, qSize, headSize, kvSeqLength);

        // Calculate scale factor
        double scale_factor = scale ? *scale : (1.0 / sqrt(static_cast<double>(headSize)));
        ET_LOG(Debug, "aoti_torch_mps__scaled_dot_product_attention_math_for_mps: scale_factor=%f", scale_factor);

        // Get Metal device
        id<MTLDevice> device = get_metal_device();
        if (!device) {
          ET_LOG(Error, "aoti_torch_mps__scaled_dot_product_attention_math_for_mps: Failed to get Metal device");
          throw std::runtime_error("Failed to get Metal device");
        }

        // Get Metal buffers for input tensors
        void* query_data_ptr = query_tensor->mutable_data_ptr();
        void* key_data_ptr = key_tensor->mutable_data_ptr();
        void* value_data_ptr = value_tensor->mutable_data_ptr();

        id<MTLBuffer> query_buffer = nullptr;
        id<MTLBuffer> key_buffer = nullptr;
        id<MTLBuffer> value_buffer = nullptr;

        // Look up Metal buffers from the global mapping
        auto query_it = ptr_to_mtl_buffer.find(query_data_ptr);
        auto key_it = ptr_to_mtl_buffer.find(key_data_ptr);
        auto value_it = ptr_to_mtl_buffer.find(value_data_ptr);

        if (query_it != ptr_to_mtl_buffer.end()) {
          query_buffer = query_it->second;
        }
        if (key_it != ptr_to_mtl_buffer.end()) {
          key_buffer = key_it->second;
        }
        if (value_it != ptr_to_mtl_buffer.end()) {
          value_buffer = value_it->second;
        }

        // Create temporary Metal buffers if not found in mapping
        if (!query_buffer) {
          size_t query_size = query_tensor->numel() * sizeof(float);
          query_buffer = [device newBufferWithBytes:query_data_ptr
                                             length:query_size
                                            options:MTLResourceStorageModeShared];
          if (!query_buffer) {
            ET_LOG(Error, "aoti_torch_mps__scaled_dot_product_attention_math_for_mps: Failed to create Metal buffer for query tensor");
            throw std::runtime_error("Failed to create Metal buffer for query tensor");
          }
        }

        if (!key_buffer) {
          size_t key_size = key_tensor->numel() * sizeof(float);
          key_buffer = [device newBufferWithBytes:key_data_ptr
                                           length:key_size
                                          options:MTLResourceStorageModeShared];
          if (!key_buffer) {
            ET_LOG(Error, "aoti_torch_mps__scaled_dot_product_attention_math_for_mps: Failed to create Metal buffer for key tensor");
            throw std::runtime_error("Failed to create Metal buffer for key tensor");
          }
        }

        if (!value_buffer) {
          size_t value_size = value_tensor->numel() * sizeof(float);
          value_buffer = [device newBufferWithBytes:value_data_ptr
                                             length:value_size
                                            options:MTLResourceStorageModeShared];
          if (!value_buffer) {
            ET_LOG(Error, "aoti_torch_mps__scaled_dot_product_attention_math_for_mps: Failed to create Metal buffer for value tensor");
            throw std::runtime_error("Failed to create Metal buffer for value tensor");
          }
        }

        // Calculate output tensor dimensions
        std::vector<int64_t> output_sizes = {batchSize, num_heads, qSize, headSize};
        std::vector<int64_t> attn_sizes = {batchSize, num_heads, qSize, kvSeqLength};

        // Calculate strides for contiguous tensors
        std::vector<int64_t> out_strides = {
            num_heads * qSize * headSize,
            qSize * headSize,
            headSize,
            1
        };

        std::vector<int64_t> attn_strides = {
            num_heads * qSize * kvSeqLength,
            qSize * kvSeqLength,
            kvSeqLength,
            1
        };

        // Allocate memory for output tensors
        size_t out_size_bytes = batchSize * num_heads * qSize * headSize * sizeof(float);
        size_t attn_size_bytes = batchSize * num_heads * qSize * kvSeqLength * sizeof(float);

        void* out_data = std::malloc(out_size_bytes);
        void* attn_data = std::malloc(attn_size_bytes);

        if (!out_data || !attn_data) {
          ET_LOG(Error, "aoti_torch_mps__scaled_dot_product_attention_math_for_mps: Failed to allocate memory");
          if (out_data) std::free(out_data);
          if (attn_data) std::free(attn_data);
          throw std::runtime_error("Failed to allocate memory for output tensors");
        }

        // Create Metal buffers for outputs
        id<MTLBuffer> out_buffer = [device newBufferWithLength:out_size_bytes
                                                       options:MTLResourceStorageModeShared];
        id<MTLBuffer> attn_weights_buffer = [device newBufferWithLength:attn_size_bytes
                                                                options:MTLResourceStorageModeShared];

        if (!out_buffer || !attn_weights_buffer) {
          ET_LOG(Error, "aoti_torch_mps__scaled_dot_product_attention_math_for_mps: Failed to create output Metal buffers");
          std::free(out_data);
          std::free(attn_data);
          throw std::runtime_error("Failed to create output Metal buffers");
        }

        // End any existing kernel coalescing to ensure a clean state for MPS
        stream->endKernelCoalescing();

        // Method 1: Using MPSGraph scaledDotProductAttention API - with detailed error handling
        ET_LOG(Debug, "aoti_torch_mps__scaled_dot_product_attention_math_for_mps: Implementing using MPSGraph scaledDotProductAttention");

        @try {
          // Check if scaledDotProductAttentionWithQueryTensor is available
          MPSGraph* testGraph = [MPSGraph new];
          if (![testGraph respondsToSelector:@selector(scaledDotProductAttentionWithQueryTensor:keyTensor:valueTensor:maskTensor:scale:name:)]) {
            ET_LOG(Error, "aoti_torch_mps__scaled_dot_product_attention_math_for_mps: scaledDotProductAttentionWithQueryTensor API not available on this system");
            throw std::runtime_error("scaledDotProductAttentionWithQueryTensor API not available on this system");
          }
          ET_LOG(Debug, "aoti_torch_mps__scaled_dot_product_attention_math_for_mps: scaledDotProductAttentionWithQueryTensor API is available");

          // Create MPSGraph for scaled dot product attention
          MPSGraph* mpsGraph = [MPSGraph new];
          ET_LOG(Debug, "aoti_torch_mps__scaled_dot_product_attention_math_for_mps: Created MPSGraph instance");

          // Define tensor shapes for placeholders
          NSArray<NSNumber*>* queryShape = @[@(batchSize), @(num_heads), @(qSize), @(headSize)];
          NSArray<NSNumber*>* keyShape = @[@(batchSize), @(num_heads), @(kvSeqLength), @(headSize)];
          NSArray<NSNumber*>* valueShape = @[@(batchSize), @(num_heads), @(kvSeqLength), @(headSize)];

          ET_LOG(Debug, "aoti_torch_mps__scaled_dot_product_attention_math_for_mps: Creating placeholders with shapes Q:[%d,%d,%d,%d] K:[%d,%d,%d,%d] V:[%d,%d,%d,%d]",
                 (int)batchSize, (int)num_heads, (int)qSize, (int)headSize,
                 (int)batchSize, (int)num_heads, (int)kvSeqLength, (int)headSize,
                 (int)batchSize, (int)num_heads, (int)kvSeqLength, (int)headSize);

          // Create placeholders for input tensors
          MPSGraphTensor* queryPlaceholder = [mpsGraph placeholderWithShape:queryShape
                                                                   dataType:MPSDataTypeFloat32
                                                                       name:@"query"];
          ET_LOG(Debug, "aoti_torch_mps__scaled_dot_product_attention_math_for_mps: Created query placeholder");

          MPSGraphTensor* keyPlaceholder = [mpsGraph placeholderWithShape:keyShape
                                                                 dataType:MPSDataTypeFloat32
                                                                     name:@"key"];
          ET_LOG(Debug, "aoti_torch_mps__scaled_dot_product_attention_math_for_mps: Created key placeholder");

          MPSGraphTensor* valuePlaceholder = [mpsGraph placeholderWithShape:valueShape
                                                                   dataType:MPSDataTypeFloat32
                                                                       name:@"value"];
          ET_LOG(Debug, "aoti_torch_mps__scaled_dot_product_attention_math_for_mps: Created value placeholder");

          MPSGraphTensor* maskTensor = nil;

          // Handle causal mask
          if (is_causal) {
            ET_LOG(Debug, "aoti_torch_mps__scaled_dot_product_attention_math_for_mps: Creating causal mask");

            // Create a causal mask: lower triangular matrix filled with 0s, upper triangle with -inf
            // Shape should be [qSize, kvSeqLength]
            NSArray<NSNumber*>* maskShape = @[@(qSize), @(kvSeqLength)];

            // Create ones tensor
            MPSGraphTensor* onesTensor = [mpsGraph constantWithScalar:1.0f
                                                                shape:maskShape
                                                             dataType:MPSDataTypeFloat32];
            ET_LOG(Debug, "aoti_torch_mps__scaled_dot_product_attention_math_for_mps: Created ones tensor for causal mask");

            // Create lower triangular mask (including diagonal)
            MPSGraphTensor* causalMask = [mpsGraph bandPartWithTensor:onesTensor
                                                            numLower:-1
                                                            numUpper:0
                                                                name:@"causal_mask"];
            ET_LOG(Debug, "aoti_torch_mps__scaled_dot_product_attention_math_for_mps: Created causal mask using bandPartWithTensor");

            // Convert mask to attention weights format: 0 for allowed positions, -inf for masked
            MPSGraphTensor* zerosTensor = [mpsGraph constantWithScalar:0.0f
                                                                 shape:maskShape
                                                              dataType:MPSDataTypeFloat32];

            MPSGraphTensor* negInfTensor = [mpsGraph constantWithScalar:-1e9f
                                                                  shape:maskShape
                                                               dataType:MPSDataTypeFloat32];

            // Select: where causal_mask == 1, use 0.0, else use -inf
            maskTensor = [mpsGraph selectWithPredicateTensor:causalMask
                                         truePredicateTensor:zerosTensor
                                        falsePredicateTensor:negInfTensor
                                                        name:@"causal_mask_final"];
            ET_LOG(Debug, "aoti_torch_mps__scaled_dot_product_attention_math_for_mps: Created final causal mask using selectWithPredicateTensor");
          }

          // Handle explicit attention mask if provided
          MPSGraphTensor* explicitMaskPlaceholder = nil;
          if (attn_mask && *attn_mask) {
            auto* mask_tensor = reinterpret_cast<executorch::runtime::etensor::Tensor*>(*attn_mask);

            ET_LOG(Debug, "aoti_torch_mps__scaled_dot_product_attention_math_for_mps: Adding explicit attention mask");

            // Create mask placeholder
            NSMutableArray<NSNumber*>* maskShapeArray = [NSMutableArray array];
            for (int i = 0; i < mask_tensor->dim(); i++) {
              [maskShapeArray addObject:@(mask_tensor->sizes()[i])];
            }

            explicitMaskPlaceholder = [mpsGraph placeholderWithShape:maskShapeArray
                                                            dataType:MPSDataTypeFloat32
                                                                name:@"attention_mask"];

            if (maskTensor) {
              // Combine causal and explicit masks
              maskTensor = [mpsGraph additionWithPrimaryTensor:maskTensor
                                               secondaryTensor:explicitMaskPlaceholder
                                                          name:@"combined_mask"];
            } else {
              maskTensor = explicitMaskPlaceholder;
            }
            ET_LOG(Debug, "aoti_torch_mps__scaled_dot_product_attention_math_for_mps: Created explicit mask placeholder");
          }

          // Perform scaled dot product attention using MPSGraph
          ET_LOG(Debug, "aoti_torch_mps__scaled_dot_product_attention_math_for_mps: Calling scaledDotProductAttentionWithQueryTensor with scale=%f", scale_factor);

          MPSGraphTensor* outputTensor = [mpsGraph scaledDotProductAttentionWithQueryTensor:queryPlaceholder
                                                                                 keyTensor:keyPlaceholder
                                                                               valueTensor:valuePlaceholder
                                                                                maskTensor:maskTensor
                                                                                     scale:scale_factor
                                                                                      name:@"scaled_dot_product_attention"];
          ET_LOG(Debug, "aoti_torch_mps__scaled_dot_product_attention_math_for_mps: Successfully created SDPA tensor");

          // Create feeds dictionary for graph execution
          NSMutableDictionary* feeds = [NSMutableDictionary dictionary];
          ET_LOG(Debug, "aoti_torch_mps__scaled_dot_product_attention_math_for_mps: Created feeds dictionary");

          // Create MPSGraphTensorData objects for input tensors
          MPSGraphTensorData* queryData = [[MPSGraphTensorData alloc] initWithMTLBuffer:query_buffer
                                                                                  shape:queryShape
                                                                               dataType:MPSDataTypeFloat32];
          MPSGraphTensorData* keyData = [[MPSGraphTensorData alloc] initWithMTLBuffer:key_buffer
                                                                                shape:keyShape
                                                                             dataType:MPSDataTypeFloat32];
          MPSGraphTensorData* valueData = [[MPSGraphTensorData alloc] initWithMTLBuffer:value_buffer
                                                                                  shape:valueShape
                                                                               dataType:MPSDataTypeFloat32];
          ET_LOG(Debug, "aoti_torch_mps__scaled_dot_product_attention_math_for_mps: Created MPSGraphTensorData objects for inputs");

          feeds[queryPlaceholder] = queryData;
          feeds[keyPlaceholder] = keyData;
          feeds[valuePlaceholder] = valueData;
          ET_LOG(Debug, "aoti_torch_mps__scaled_dot_product_attention_math_for_mps: Added input tensors to feeds");

          // Add explicit mask data to feeds if provided
          if (explicitMaskPlaceholder && attn_mask && *attn_mask) {
            auto* mask_tensor = reinterpret_cast<executorch::runtime::etensor::Tensor*>(*attn_mask);
            void* mask_data_ptr = mask_tensor->mutable_data_ptr();

            // Get or create Metal buffer for mask
            id<MTLBuffer> mask_buffer = nullptr;
            auto mask_it = ptr_to_mtl_buffer.find(mask_data_ptr);
            if (mask_it != ptr_to_mtl_buffer.end()) {
              mask_buffer = mask_it->second;
            } else {
              size_t mask_size = mask_tensor->numel() * sizeof(float);
              mask_buffer = [device newBufferWithBytes:mask_data_ptr
                                                length:mask_size
                                               options:MTLResourceStorageModeShared];
              if (!mask_buffer) {
                ET_LOG(Error, "aoti_torch_mps__scaled_dot_product_attention_math_for_mps: Failed to create Metal buffer for attention mask");
                throw std::runtime_error("Failed to create Metal buffer for attention mask");
              }
            }

            NSMutableArray<NSNumber*>* maskShapeArray = [NSMutableArray array];
            for (int i = 0; i < mask_tensor->dim(); i++) {
              [maskShapeArray addObject:@(mask_tensor->sizes()[i])];
            }

            MPSGraphTensorData* maskData = [[MPSGraphTensorData alloc] initWithMTLBuffer:mask_buffer
                                                                                   shape:maskShapeArray
                                                                                dataType:MPSDataTypeFloat32];
            feeds[explicitMaskPlaceholder] = maskData;
            ET_LOG(Debug, "aoti_torch_mps__scaled_dot_product_attention_math_for_mps: Added explicit mask tensor to feeds");
          }

          // Create results dictionary
          NSArray<NSNumber*>* outputShape = @[@(batchSize), @(num_heads), @(qSize), @(headSize)];
          MPSGraphTensorData* outputData = [[MPSGraphTensorData alloc] initWithMTLBuffer:out_buffer
                                                                                    shape:outputShape
                                                                                 dataType:MPSDataTypeFloat32];

          NSDictionary* results = @{outputTensor: outputData};
          ET_LOG(Debug, "aoti_torch_mps__scaled_dot_product_attention_math_for_mps: Created results dictionary");

          // Execute the MPSGraph using a corrected direct approach
          ET_LOG(Debug, "aoti_torch_mps__scaled_dot_product_attention_math_for_mps: Executing MPSGraph with corrected approach");

          // Use dispatch_sync_with_rethrow to match PyTorch's approach for MPSGraph
          dispatch_sync_with_rethrow(stream->queue(), ^() {
            @autoreleasepool {
              // Get a fresh command buffer for this specific operation
              id<MTLCommandBuffer> cmdBuf = [stream->commandQueue() commandBuffer];
              if (!cmdBuf) {
                ET_LOG(Error, "aoti_torch_mps__scaled_dot_product_attention_math_for_mps: Failed to create fresh command buffer");
                throw std::runtime_error("Failed to create fresh command buffer");
              }

              // Use the newer MPSGraph API with resultsDictionary
              [mpsGraph encodeToCommandBuffer:cmdBuf
                                        feeds:feeds
                             targetOperations:nil
                            resultsDictionary:results
                          executionDescriptor:nil];

              // Commit and wait for completion
              [cmdBuf commit];
              [cmdBuf waitUntilCompleted];

              // Check for errors
              if (cmdBuf.status == MTLCommandBufferStatusError) {
                NSString* errorDesc = cmdBuf.error ? cmdBuf.error.description : @"Unknown error";
                ET_LOG(Error, "aoti_torch_mps__scaled_dot_product_attention_math_for_mps: Command buffer execution failed: %s", [errorDesc UTF8String]);
                throw std::runtime_error("Command buffer execution failed");
              }
            }
          });

          ET_LOG(Debug, "aoti_torch_mps__scaled_dot_product_attention_math_for_mps: MPSGraph execution completed successfully");

          // Copy results back to CPU memory
          memcpy(out_data, [out_buffer contents], out_size_bytes);

          ET_LOG(Debug, "aoti_torch_mps__scaled_dot_product_attention_math_for_mps: MPSGraph execution completed successfully");

        } @catch (NSException *exception) {
          ET_LOG(Error, "aoti_torch_mps__scaled_dot_product_attention_math_for_mps: NSException caught: %s - %s",
                 [[exception name] UTF8String], [[exception reason] UTF8String]);
          throw std::runtime_error("MPSGraph operation failed with NSException");
        }

        // For attention weights, we'll create a dummy tensor filled with zeros for now
        // In a full implementation, you'd need to extract attention weights from the graph
        std::memset(attn_data, 0, attn_size_bytes);

        // Create output tensor handles
        AOTITensorHandle out_tensor_handle = nullptr;
        AOTITensorHandle attn_tensor_handle = nullptr;

        AOTITorchError create_out_result = aoti_torch_create_tensor_from_blob_v2(
            out_data,
            4,  // ndim
            output_sizes.data(),
            out_strides.data(),
            0,  // storage_offset
            static_cast<int32_t>(SupportedDTypes::FLOAT32),
            0,  // device_type (CPU)
            0,  // device_index
            &out_tensor_handle,
            0,  // layout (strided)
            nullptr,  // opaque_metadata
            0   // opaque_metadata_size
        );

        AOTITorchError create_attn_result = aoti_torch_create_tensor_from_blob_v2(
            attn_data,
            4,  // ndim
            attn_sizes.data(),
            attn_strides.data(),
            0,  // storage_offset
            static_cast<int32_t>(SupportedDTypes::FLOAT32),
            0,  // device_type (CPU)
            0,  // device_index
            &attn_tensor_handle,
            0,  // layout (strided)
            nullptr,  // opaque_metadata
            0   // opaque_metadata_size
        );

        if (create_out_result != Error::Ok || create_attn_result != Error::Ok ||
            !out_tensor_handle || !attn_tensor_handle) {
          ET_LOG(Error, "aoti_torch_mps__scaled_dot_product_attention_math_for_mps: Failed to create output tensors");
          std::free(out_data);
          std::free(attn_data);
          throw std::runtime_error("Failed to create output tensors");
        }

        // Mark that we own the memory for these tensors
        auto* out_et_tensor = reinterpret_cast<executorch::runtime::etensor::Tensor*>(out_tensor_handle);
        auto* attn_et_tensor = reinterpret_cast<executorch::runtime::etensor::Tensor*>(attn_tensor_handle);
        is_tensor_own_memory[out_et_tensor] = true;
        is_tensor_own_memory[attn_et_tensor] = true;

        // Set output tensor handles
        *ret0 = out_tensor_handle;
        *ret1 = attn_tensor_handle;

      ET_LOG(Debug, "aoti_torch_mps__scaled_dot_product_attention_math_for_mps: MPSGraph implementation completed successfully");
    }

    return Error::Ok;

  } catch (const std::exception& e) {
    ET_LOG(Error, "aoti_torch_mps__scaled_dot_product_attention_math_for_mps exception: %s", e.what());
    return Error::Internal;
  } catch (...) {
    ET_LOG(Error, "aoti_torch_mps__scaled_dot_product_attention_math_for_mps: unknown exception");
    return Error::Internal;
  }
}

} // extern "C"

} // namespace aoti
} // namespace backends
} // namespace executorch
