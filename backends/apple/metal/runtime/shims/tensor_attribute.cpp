/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "tensor_attribute.h"
#include <iostream>
#include "utils.h"

namespace executorch {
namespace backends {
namespace metal {

extern "C" {

// Metal-specific device type constant
__attribute__((__visibility__("default"))) int32_t
aoti_torch_device_type_mps() {
  // Let's use 2 for MPS
  return 2;
}

// Override aoti_torch_get_device_type to return MPS device type
AOTITorchError aoti_torch_get_device_type(
    AOTITensorHandle tensor,
    int32_t* ret_device_type) {
  *ret_device_type = aoti_torch_device_type_mps();
  return Error::Ok;
}

// Metal-specific storage size function (not supported)
AOTITorchError aoti_torch_get_storage_size(
    AOTITensorHandle tensor,
    int64_t* ret_size) {
  throw std::runtime_error("Cannot get storage size on ETensor");
}

} // extern "C"

} // namespace metal
} // namespace backends
} // namespace executorch
