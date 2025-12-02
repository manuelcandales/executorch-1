/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <cassert>
#include <cmath>
#include <cstring>
#include <iostream>
#include <vector>

#include <executorch/backends/apple/metal/runtime/shims/et_metal_ops.h>
#include <executorch/backends/apple/metal/runtime/shims/memory.h>
#include <executorch/backends/apple/metal/runtime/shims/types.h>
#include <executorch/backends/apple/metal/runtime/shims/utils.h>
#include <executorch/runtime/core/exec_aten/util/scalar_type_util.h>

using namespace executorch::backends::metal;

// Helper function to create a tensor with specified data
AOTITensorHandle create_test_tensor(
    const std::vector<float>& data,
    const std::vector<int64_t>& sizes,
    int32_t dtype) {
  AOTITensorHandle tensor = nullptr;
  
  // Calculate strides for contiguous tensor
  std::vector<int64_t> strides(sizes.size());
  int64_t stride = 1;
  for (int64_t i = static_cast<int64_t>(sizes.size()) - 1; i >= 0; --i) {
    strides[i] = stride;
    stride *= sizes[i];
  }
  
  // Allocate memory for the tensor data
  size_t element_size = (dtype == static_cast<int32_t>(SupportedDTypes::FLOAT32))
                            ? sizeof(float)
                            : sizeof(uint16_t);
  size_t data_size = data.size() * element_size;
  void* data_ptr = malloc(data_size);
  
  if (dtype == static_cast<int32_t>(SupportedDTypes::FLOAT32)) {
    std::memcpy(data_ptr, data.data(), data_size);
  } else if (dtype == static_cast<int32_t>(SupportedDTypes::BFLOAT16)) {
    // Convert float to bfloat16 (simple truncation for testing)
    uint16_t* bf16_data = static_cast<uint16_t*>(data_ptr);
    for (size_t i = 0; i < data.size(); ++i) {
      uint32_t bits = *reinterpret_cast<const uint32_t*>(&data[i]);
      bf16_data[i] = static_cast<uint16_t>(bits >> 16);
    }
  }
  
  AOTITorchError err = aoti_torch_create_tensor_from_blob_v2(
      data_ptr, static_cast<int64_t>(sizes.size()), sizes.data(),
      strides.data(), 0, dtype, 13,  // device_type: MPS
      0,                              // device_index
      &tensor, 0,                     // layout: strided
      nullptr, 0);                    // opaque_metadata
  
  if (err != Error::Ok || !tensor) {
    std::cerr << "Failed to create tensor: error " << static_cast<int>(err)
              << std::endl;
    free(data_ptr);
    return nullptr;
  }
  
  return tensor;
}

// Helper function to print tensor contents
void print_tensor(const char* name, AOTITensorHandle tensor, int max_elements = 10) {
  if (!tensor) {
    std::cout << name << ": null" << std::endl;
    return;
  }
  
  auto* t = reinterpret_cast<Tensor*>(tensor);
  std::cout << name << ": shape=[";
  for (int i = 0; i < t->dim(); ++i) {
    if (i > 0) std::cout << ", ";
    std::cout << t->sizes()[i];
  }
  std::cout << "], dtype=" << static_cast<int>(t->scalar_type()) << std::endl;
  
  // Print first few elements
  if (t->numel() > 0) {
    size_t numel = static_cast<size_t>(t->numel());
    std::cout << "  First " << std::min(max_elements, static_cast<int>(numel))
              << " elements: ";
    size_t num_to_print = std::min(static_cast<size_t>(max_elements), numel);
    
    if (t->scalar_type() == executorch::aten::ScalarType::Float) {
      const float* data = t->const_data_ptr<float>();
      for (size_t i = 0; i < num_to_print; ++i) {
        if (i > 0) std::cout << ", ";
        std::cout << data[i];
      }
    } else if (t->scalar_type() == executorch::aten::ScalarType::BFloat16) {
      const uint16_t* data = t->const_data_ptr<uint16_t>();
      for (size_t i = 0; i < num_to_print; ++i) {
        if (i > 0) std::cout << ", ";
        // Convert bfloat16 to float for display
        uint32_t bits = static_cast<uint32_t>(data[i]) << 16;
        float val = *reinterpret_cast<const float*>(&bits);
        std::cout << val;
      }
    }
    std::cout << std::endl;
  }
}

// Helper function to check if tensor is all zeros
bool is_all_zeros(AOTITensorHandle tensor) {
  if (!tensor) return true;
  
  auto* t = reinterpret_cast<Tensor*>(tensor);
  size_t numel = t->numel();
  if (numel == 0) return true;
  
  if (t->scalar_type() == executorch::aten::ScalarType::Float) {
    const float* data = t->const_data_ptr<float>();
    for (size_t i = 0; i < numel; ++i) {
      if (std::abs(data[i]) > 1e-6f) return false;
    }
  } else if (t->scalar_type() == executorch::aten::ScalarType::BFloat16) {
    const uint16_t* data = t->const_data_ptr<uint16_t>();
    for (size_t i = 0; i < numel; ++i) {
      if (data[i] != 0) return false;
    }
  }
  
  return true;
}

// Test case: Simple SDPA with small tensors
bool test_sdpa_basic() {
  std::cout << "\n=== Test: Basic SDPA ===" << std::endl;
  
  // Create test data: batch=2, num_heads=4, seq_len=16, head_dim=64
  const int64_t batch = 2;
  const int64_t num_heads = 4;
  const int64_t seq_len = 16;
  const int64_t head_dim = 64;
  const int64_t kv_seq_len = 16;
  
  const size_t q_size = batch * num_heads * seq_len * head_dim;
  const size_t kv_size = batch * num_heads * kv_seq_len * head_dim;
  
  // Create query tensor with small random-like values
  std::vector<float> q_data(q_size);
  for (size_t i = 0; i < q_size; ++i) {
    q_data[i] = 0.1f * (static_cast<float>(i % 100) / 100.0f - 0.5f);
  }
  
  // Create key tensor
  std::vector<float> k_data(kv_size);
  for (size_t i = 0; i < kv_size; ++i) {
    k_data[i] = 0.1f * (static_cast<float>((i + 50) % 100) / 100.0f - 0.5f);
  }
  
  // Create value tensor
  std::vector<float> v_data(kv_size);
  for (size_t i = 0; i < kv_size; ++i) {
    v_data[i] = 0.1f * (static_cast<float>((i + 25) % 100) / 100.0f - 0.5f);
  }
  
  int32_t dtype = static_cast<int32_t>(SupportedDTypes::FLOAT32);
  
  AOTITensorHandle query = create_test_tensor(
      q_data, {batch, num_heads, seq_len, head_dim}, dtype);
  AOTITensorHandle key = create_test_tensor(
      k_data, {batch, num_heads, kv_seq_len, head_dim}, dtype);
  AOTITensorHandle value = create_test_tensor(
      v_data, {batch, num_heads, kv_seq_len, head_dim}, dtype);
  
  if (!query || !key || !value) {
    std::cerr << "Failed to create input tensors" << std::endl;
    return false;
  }
  
  print_tensor("Query", query, 5);
  print_tensor("Key", key, 5);
  print_tensor("Value", value, 5);
  
  // Call SDPA
  AOTITensorHandle output = nullptr;
  AOTITensorHandle attn_weights = nullptr;
  AOTITensorHandle* attn_mask = nullptr;
  double dropout_p = 0.0;
  int32_t is_causal = 0;
  AOTITensorHandle* dropout_mask = nullptr;
  double* scale = nullptr;  // Use default scale
  
  AOTITorchError err = aoti_torch_mps__scaled_dot_product_attention_math_for_mps(
      query, key, value, attn_mask, dropout_p, is_causal, dropout_mask, scale,
      &output, &attn_weights);
  
  if (err != Error::Ok) {
    std::cerr << "SDPA failed with error: " << static_cast<int>(err) << std::endl;
    return false;
  }
  
  if (!output || !attn_weights) {
    std::cerr << "SDPA returned null output tensors" << std::endl;
    return false;
  }
  
  print_tensor("Output", output, 10);
  print_tensor("Attention Weights", attn_weights, 5);
  
  // Check if output is all zeros (this would indicate a bug)
  if (is_all_zeros(output)) {
    std::cerr << "ERROR: Output tensor is all zeros!" << std::endl;
    return false;
  }
  
  std::cout << "✓ Test passed: Output is not all zeros" << std::endl;
  
  // Cleanup
  if (output) aoti_torch_delete_tensor_object(output);
  if (attn_weights) aoti_torch_delete_tensor_object(attn_weights);
  if (query) aoti_torch_delete_tensor_object(query);
  if (key) aoti_torch_delete_tensor_object(key);
  if (value) aoti_torch_delete_tensor_object(value);
  
  return true;
}

// Test case: SDPA with bfloat16
bool test_sdpa_bfloat16() {
  std::cout << "\n=== Test: SDPA with BFloat16 ===" << std::endl;
  
  const int64_t batch = 1;
  const int64_t num_heads = 4;
  const int64_t seq_len = 8;
  const int64_t head_dim = 64;
  const int64_t kv_seq_len = 8;
  
  const size_t q_size = batch * num_heads * seq_len * head_dim;
  const size_t kv_size = batch * num_heads * kv_seq_len * head_dim;
  
  std::vector<float> q_data(q_size, 0.1f);
  std::vector<float> k_data(kv_size, 0.1f);
  std::vector<float> v_data(kv_size, 0.1f);
  
  int32_t dtype = static_cast<int32_t>(SupportedDTypes::BFLOAT16);
  
  AOTITensorHandle query = create_test_tensor(
      q_data, {batch, num_heads, seq_len, head_dim}, dtype);
  AOTITensorHandle key = create_test_tensor(
      k_data, {batch, num_heads, kv_seq_len, head_dim}, dtype);
  AOTITensorHandle value = create_test_tensor(
      v_data, {batch, num_heads, kv_seq_len, head_dim}, dtype);
  
  if (!query || !key || !value) {
    std::cerr << "Failed to create input tensors" << std::endl;
    return false;
  }
  
  AOTITensorHandle output = nullptr;
  AOTITensorHandle attn_weights = nullptr;
  
  AOTITorchError err = aoti_torch_mps__scaled_dot_product_attention_math_for_mps(
      query, key, value, nullptr, 0.0, 0, nullptr, nullptr, &output,
      &attn_weights);
  
  if (err != Error::Ok) {
    std::cerr << "SDPA failed with error: " << static_cast<int>(err) << std::endl;
    return false;
  }
  
  if (is_all_zeros(output)) {
    std::cerr << "ERROR: Output tensor is all zeros!" << std::endl;
    return false;
  }
  
  std::cout << "✓ Test passed: BFloat16 output is not all zeros" << std::endl;
  
  // Cleanup
  if (output) aoti_torch_delete_tensor_object(output);
  if (attn_weights) aoti_torch_delete_tensor_object(attn_weights);
  if (query) aoti_torch_delete_tensor_object(query);
  if (key) aoti_torch_delete_tensor_object(key);
  if (value) aoti_torch_delete_tensor_object(value);
  
  return true;
}

// Test case: SDPA with custom scale
bool test_sdpa_custom_scale() {
  std::cout << "\n=== Test: SDPA with Custom Scale ===" << std::endl;
  
  const int64_t batch = 1;
  const int64_t num_heads = 2;
  const int64_t seq_len = 4;
  const int64_t head_dim = 64;
  const int64_t kv_seq_len = 4;
  
  const size_t q_size = batch * num_heads * seq_len * head_dim;
  const size_t kv_size = batch * num_heads * kv_seq_len * head_dim;
  
  std::vector<float> q_data(q_size, 0.5f);
  std::vector<float> k_data(kv_size, 0.5f);
  std::vector<float> v_data(kv_size, 1.0f);
  
  int32_t dtype = static_cast<int32_t>(SupportedDTypes::FLOAT32);
  
  AOTITensorHandle query = create_test_tensor(
      q_data, {batch, num_heads, seq_len, head_dim}, dtype);
  AOTITensorHandle key = create_test_tensor(
      k_data, {batch, num_heads, kv_seq_len, head_dim}, dtype);
  AOTITensorHandle value = create_test_tensor(
      v_data, {batch, num_heads, kv_seq_len, head_dim}, dtype);
  
  if (!query || !key || !value) {
    std::cerr << "Failed to create input tensors" << std::endl;
    return false;
  }
  
  AOTITensorHandle output = nullptr;
  AOTITensorHandle attn_weights = nullptr;
  double custom_scale = 0.125;  // Custom scale factor
  
  AOTITorchError err = aoti_torch_mps__scaled_dot_product_attention_math_for_mps(
      query, key, value, nullptr, 0.0, 0, nullptr, &custom_scale, &output,
      &attn_weights);
  
  if (err != Error::Ok) {
    std::cerr << "SDPA failed with error: " << static_cast<int>(err) << std::endl;
    return false;
  }
  
  if (is_all_zeros(output)) {
    std::cerr << "ERROR: Output tensor is all zeros!" << std::endl;
    return false;
  }
  
  print_tensor("Output (custom scale)", output, 5);
  std::cout << "✓ Test passed: Custom scale output is not all zeros" << std::endl;
  
  // Cleanup
  if (output) aoti_torch_delete_tensor_object(output);
  if (attn_weights) aoti_torch_delete_tensor_object(attn_weights);
  if (query) aoti_torch_delete_tensor_object(query);
  if (key) aoti_torch_delete_tensor_object(key);
  if (value) aoti_torch_delete_tensor_object(value);
  
  return true;
}

int main(int argc, char** argv) {  
  std::cout << "SDPA Unit Tests" << std::endl;
  std::cout << "===============" << std::endl;
  
  bool all_passed = true;
  
  all_passed &= test_sdpa_basic();
  all_passed &= test_sdpa_bfloat16();
  all_passed &= test_sdpa_custom_scale();
  
  std::cout << "\n===============" << std::endl;
  if (all_passed) {
    std::cout << "All tests PASSED" << std::endl;
    return 0;
  } else {
    std::cout << "Some tests FAILED" << std::endl;
    return 1;
  }
}

