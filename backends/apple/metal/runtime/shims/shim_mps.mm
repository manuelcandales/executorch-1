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
#include "memory.h"  // For tensors and is_tensor_own_memory globals

namespace executorch {
namespace backends {
namespace aoti {

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

      // Create tensor using from_blob to ensure we have control over the memory
      auto output_tensor = executorch::extension::from_blob(
          tensor_data,
          output_sizes,
          executorch::runtime::etensor::ScalarType::Float
      );

      if (!output_tensor) {
        ET_LOG(Error, "aoti_torch_mps_convolution: Failed to create output tensor");
        std::free(tensor_data);  // Free the allocated memory on failure
        return Error::Internal;
      }

      // Verify the tensor was created with the correct size
      size_t actual_numel = output_tensor->numel();
      ET_LOG(Debug, "aoti_torch_mps_convolution: Created tensor with actual numel = %zu", actual_numel);

      if (actual_numel != expected_numel) {
        ET_LOG(Error, "aoti_torch_mps_convolution: Tensor size mismatch. Expected %zu, got %zu", expected_numel, actual_numel);
        std::free(tensor_data);  // Free the allocated memory on failure
        return Error::Internal;
      }

      // Store the tensor so it doesn't get destroyed - mark that we own the memory
      // since we manually allocated it with malloc
      tensors.insert(output_tensor);
      *ret0 = reinterpret_cast<AtenTensorHandle>(output_tensor.get());
      is_tensor_own_memory[output_tensor.get()] = true;  // We allocated the memory manually

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

} // namespace aoti
} // namespace backends
} // namespace executorch
