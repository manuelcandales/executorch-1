# MLX Metal Matmul Integration Summary

## Overview

This document summarizes the integration of MLX's optimized Metal matrix multiplication kernels into ExecuTorch's Metal backend, replacing the previous MPS-based implementation.

## Changes Made

### 1. New Files Created

#### C++ Implementation Files

**`backends/apple/metal/runtime/mlx/mlx_matmul.h`**
- Header file defining the MLX matmul interface for ExecuTorch
- Declares `metal_mm_out()` function - the main entry point
- Declares `init_mlx_matmul_kernels()` for initialization
- Defines GEMM parameter structures adapted from MLX

**`backends/apple/metal/runtime/mlx/mlx_matmul.mm`**
- Implementation of MLX matmul adapted for ExecuTorch
- Includes:
  - `metal_mm_out()` - Main matmul entry point
  - `steel_matmul_regular()` - Steel GEMM kernel dispatch
  - Helper functions for transpose detection, dtype mapping, etc.
- Adapts MLX's dispatch logic to use ExecuTorch abstractions:
  - MLX `array` → ExecuTorch `Tensor`
  - MLX `metal::Device` → ExecuTorch Metal device
  - MLX `Stream` → ExecuTorch `ETMetalStream`

#### Documentation

**`backends/apple/metal/runtime/mlx/README.md`**
- Comprehensive documentation of the integration
- Lists completed work and TODOs
- Explains what needs to be done to make it fully functional
- Includes build integration instructions

**`INTEGRATION_SUMMARY.md`** (this file)
- High-level summary of all changes

### 2. MLX Kernel Files Copied

#### Metal Shader Source Files

**Steel GEMM Kernels:**
- `runtime/mlx/kernels/steel/gemm/kernels/steel_gemm_fused.metal`
- `runtime/mlx/kernels/steel/gemm/kernels/steel_gemm_fused.h`
- `runtime/mlx/kernels/steel/gemm/kernels/steel_gemm_splitk.metal`
- `runtime/mlx/kernels/steel/gemm/kernels/steel_gemm_splitk.h`

**GEMV Kernels:**
- `runtime/mlx/kernels/gemv.metal`

#### Header Files

**Steel GEMM Headers:**
- `runtime/mlx/kernels/steel/gemm/gemm.h`
- `runtime/mlx/kernels/steel/gemm/loader.h`
- `runtime/mlx/kernels/steel/gemm/mma.h`
- `runtime/mlx/kernels/steel/gemm/params.h`
- `runtime/mlx/kernels/steel/gemm/transforms.h`

**Steel Utility Headers:**
- `runtime/mlx/kernels/steel/defines.h`
- `runtime/mlx/kernels/steel/utils/integral_constant.h`
- `runtime/mlx/kernels/steel/utils/type_traits.h`

**General Headers:**
- `runtime/mlx/kernels/defines.h`
- `runtime/mlx/kernels/utils.h`
- `runtime/mlx/kernels/bf16_math.h`
- `runtime/mlx/kernels/complex.h`
- `runtime/mlx/params.h`
- `runtime/mlx/defines.h`

### 3. Modified Files

**`backends/apple/metal/runtime/shims/et_metal_ops.mm`**

Changes made:
1. Added include for MLX matmul header:
   ```cpp
   #include <executorch/backends/apple/metal/runtime/mlx/mlx_matmul.h>
   ```

2. Replaced MPS implementation in `aoti_torch_mps_mm_out()`:
   ```cpp
   // NEW: Call MLX matmul implementation
   ETMetalStream* stream = getCurrentMetalStream();
   bool success = mlx::metal_mm_out(out_tensor, self_tensor, mat2_tensor, stream);
   ```

3. Commented out old MPS implementation:
   - Wrapped original MPSGraph-based code in `#if 0 ... #endif`
   - Preserved for reference and potential fallback
   - Added explanatory comments

## Architecture

### Call Flow

```
User Code
  ↓
aoti_torch_mps_mm_out()               [et_metal_ops.mm]
  ↓
mlx::metal_mm_out()                   [mlx_matmul.mm]
  ↓
steel_matmul_regular()                [mlx_matmul.mm]
  ↓
[MLX Steel GEMM kernels]              [steel_gemm_fused.metal]
  ↓
Metal GPU
```

### Data Flow

```
ExecuTorch Tensor (Tensor*)
  ↓
Extract dimensions, strides, dtype
  ↓
Detect transpose, calculate leading dimensions
  ↓
Build GEMMParams struct
  ↓
Get Metal buffers from ptr_to_mtl_buffer
  ↓
Encode kernel dispatch
  ↓
Execute on Metal GPU
```

## Current Status

### ✅ What Works

1. **File structure** - All MLX kernel files are copied to the correct locations
2. **C++ interface** - `mlx_matmul.h` and `mlx_matmul.mm` provide the integration layer
3. **Call site integration** - `aoti_torch_mps_mm_out()` calls the new implementation
4. **Old code preserved** - MPS implementation is commented out but available for reference

### ⚠️ What Doesn't Work Yet (Critical Issues)

1. **Kernel Compilation** - The `.metal` files are not yet compiled or loaded
   - `get_mlx_kernel_source()` returns a placeholder
   - Need to either:
     - Embed kernel source as strings and compile at runtime
     - Pre-compile to `.metallib` and load at runtime

2. **Function Constants** - MLX uses Metal function constants for specialization
   - ExecuTorch's `ETMetalShaderLibrary` may not support this
   - Need alternative approach or extension

3. **Dtype Mapping** - Hardcoded assumptions about dtype encoding
   - Need proper mapping from ExecuTorch scalar types to Metal types

4. **Batched Operations** - Only 2D matmul is supported
   - Need to implement batch dimension handling

## Next Steps (Priority Order)

### 1. Critical Path to Minimal Functionality

These are the minimum changes needed to get basic 2D matmul working:

1. **Implement kernel source loading** (CRITICAL)
   ```cpp
   // Option A: Embed as strings
   const char* STEEL_GEMM_SOURCE = R"(
   #include <metal_stdlib>
   // ... embedded MLX kernel source ...
   )";
   
   // Option B: Load from compiled metallib
   NSString* libPath = [[NSBundle mainBundle] pathForResource:@"mlx_kernels" ofType:@"metallib"];
   id<MTLLibrary> library = [device newLibraryWithFile:libPath error:&error];
   ```

2. **Fix dtype mapping** (HIGH)
   - Query actual ExecuTorch scalar type enum values
   - Map to correct Metal type names

3. **Implement proper Metal buffer handling** (HIGH)
   - Verify `ptr_to_mtl_buffer` lookup works correctly
   - Add error handling and fallbacks

4. **Test basic 2D matmul** (HIGH)
   - Create simple test case with known inputs
   - Verify outputs match CPU reference

### 2. Enhanced Functionality

Once basic functionality works:

5. **Add batched matmul support** (MEDIUM)
   - Implement `collapse_batches()` logic
   - Handle batch strides

6. **Add GEMV specialization** (MEDIUM)
   - Implement `gemv()` dispatch
   - Use optimized kernels for M=1 or N=1

7. **Add device architecture detection** (LOW)
   - Query Metal device capabilities
   - Select optimal tile sizes

8. **Add split-K GEMM** (LOW)
   - Implement `steel_gemm_splitk_axpby()`
   - Add dispatch heuristics

## Build System Integration (TODO)

The following needs to be added to ExecuTorch's CMake build:

```cmake
# In backends/apple/metal/CMakeLists.txt

# Add MLX implementation
target_sources(executorch_metal PRIVATE
  runtime/mlx/mlx_matmul.mm
)

# Include MLX headers
target_include_directories(executorch_metal PRIVATE
  runtime/mlx
  runtime/mlx/kernels
)

# Option 1: Compile Metal shaders to metallib
add_custom_command(
  OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/mlx_kernels.metallib
  COMMAND xcrun -sdk macosx metal -c ${CMAKE_CURRENT_SOURCE_DIR}/runtime/mlx/kernels/steel/gemm/kernels/steel_gemm_fused.metal -o steel_gemm_fused.air
  COMMAND xcrun -sdk macosx metallib steel_gemm_fused.air -o mlx_kernels.metallib
  DEPENDS runtime/mlx/kernels/steel/gemm/kernels/steel_gemm_fused.metal
)
add_custom_target(mlx_kernels ALL DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/mlx_kernels.metallib)

# Option 2: Embed source as strings
# (Use a code generator to create mlx_kernel_source.h)
```

## Testing Plan

### Unit Tests

Create tests for:
1. Basic 2D matmul (square matrices)
2. Non-square matrices
3. Different dtypes (float32, bfloat16)
4. Transposed inputs
5. Small and large matrices
6. Edge cases (1x1, very large)

### Integration Tests

Test with:
1. Real model workloads
2. PyTorch model export to ExecuTorch
3. Performance benchmarking vs MPS

### Performance Validation

Compare against:
1. MPS implementation (baseline)
2. MLX standalone (target)
3. Other Metal GEMM libraries

## Known Limitations

1. **No complex number support** - MLX supports complex64 but adapter doesn't
2. **No quantized matmul** - MLX has quantized kernels but not integrated
3. **No gather/segmented matmul** - MLX advanced features not ported
4. **Limited error handling** - Need more robust error recovery

## References

### Original MLX Implementation

- Source: `mlx/backend/metal/matmul.cpp`
- Kernels: `mlx/backend/metal/kernels/steel/gemm/`
- License: BSD (Copyright © 2023-2024 Apple Inc.)

### ExecuTorch Metal Backend

- Location: `executorch/backends/apple/metal/`
- Stream: `ETMetalStream` in `runtime/shims/et_metal.h`
- Ops: `aoti_torch_mps_*` in `runtime/shims/et_metal_ops.mm`

## Inline Comments

Throughout the code, you'll find several types of comments marking adaptation points:

- `// FIXME:` - Critical issues that must be addressed for functionality
- `// TODO:` - Enhancements and optimizations
- `// Adapted from MLX:` - Indicates code ported from MLX
- `// ExecuTorch:` - Indicates ExecuTorch-specific adaptations

## Contact

For questions about this integration:
- Check `backends/apple/metal/runtime/mlx/README.md` for detailed documentation
- Review MLX source code at https://github.com/ml-explore/mlx
- Review ExecuTorch documentation

---

**Last Updated:** October 21, 2024
**Status:** Framework complete, kernel compilation and testing remain

