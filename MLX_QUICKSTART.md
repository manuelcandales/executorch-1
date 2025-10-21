# MLX Metal Matmul Integration - Quick Start Guide

## What Was Done

The ExecuTorch Metal backend has been modified to use MLX's optimized Metal matrix multiplication kernels instead of MetalPerformanceShadersGraph (MPSGraph).

### Changes Summary

✅ **Completed:**
- ✅ Copied all MLX Metal shader files (~91KB of optimized kernels)
- ✅ Created C++ adapter layer (`mlx_matmul.h`, `mlx_matmul.mm`)
- ✅ Integrated into ExecuTorch entry point (`aoti_torch_mps_mm_out`)
- ✅ Commented out old MPS implementation (preserved for reference)
- ✅ Created comprehensive documentation

⚠️ **Incomplete (Critical):**
- ❌ Metal kernel compilation (kernels not loaded yet)
- ❌ Dtype mapping (hardcoded assumptions)
- ❌ Function constant specialization
- ❌ Testing

## File Locations

```
executorch/
├── backends/apple/metal/runtime/
│   ├── mlx/                           ← NEW: MLX integration code
│   │   ├── README.md                  ← Detailed documentation
│   │   ├── mlx_matmul.h               ← Interface
│   │   ├── mlx_matmul.mm              ← Implementation
│   │   └── kernels/                   ← MLX Metal shaders
│   │       └── steel/gemm/kernels/    ← Optimized GEMM kernels
│   └── shims/
│       └── et_metal_ops.mm            ← MODIFIED: Calls MLX now
├── INTEGRATION_SUMMARY.md             ← High-level summary
├── MLX_INTEGRATION_DIFF.md            ← Complete diff
└── MLX_QUICKSTART.md                  ← This file
```

## What Works Now

The **structure** is in place:
- ✅ Directory structure created
- ✅ Files copied
- ✅ Code paths wired up
- ✅ Entry point redirected

## What Doesn't Work Yet

The **implementation** needs completion:
- ❌ Kernels not compiled/loaded → will fail at runtime
- ❌ No actual kernel execution
- ❌ Untested

## Next Steps to Make It Work

### Step 1: Compile Metal Shaders (CRITICAL)

The MLX `.metal` files need to be compiled. Choose one approach:

#### Option A: Embed Source at Build Time (Recommended)

1. Create a script to embed Metal source as C++ strings:

```python
# generate_kernel_source.py
import os

def embed_file(filepath):
    with open(filepath, 'r') as f:
        content = f.read()
    # Escape special characters
    content = content.replace('\\', '\\\\').replace('"', '\\"').replace('\n', '\\n')
    return f'R"({content})"'

# Generate embedded source
sources = {
    'STEEL_GEMM_FUSED': 'mlx/kernels/steel/gemm/kernels/steel_gemm_fused.metal',
    'STEEL_GEMM_SPLITK': 'mlx/kernels/steel/gemm/kernels/steel_gemm_splitk.metal',
    'GEMV': 'mlx/kernels/gemv.metal',
}

with open('mlx_kernel_sources.h', 'w') as out:
    out.write('#pragma once\n\n')
    for name, path in sources.items():
        out.write(f'static const char* {name}_SOURCE = {embed_file(path)};\n\n')
```

2. Update `mlx_matmul.mm`:

```cpp
#include "mlx_kernel_sources.h"

std::string get_mlx_kernel_source() {
  std::ostringstream source;
  source << STEEL_GEMM_FUSED_SOURCE;
  source << STEEL_GEMM_SPLITK_SOURCE;
  source << GEMV_SOURCE;
  return source.str();
}
```

#### Option B: Pre-compile to .metallib

```bash
# Compile Metal shaders
cd executorch/backends/apple/metal/runtime/mlx
xcrun -sdk macosx metal -c kernels/steel/gemm/kernels/steel_gemm_fused.metal -o steel_gemm_fused.air
xcrun -sdk macosx metallib steel_gemm_fused.air -o mlx_kernels.metallib

# Bundle with app and load at runtime
NSString* libPath = [[NSBundle mainBundle] pathForResource:@"mlx_kernels" ofType:@"metallib"];
id<MTLLibrary> library = [device newLibraryWithFile:libPath error:&error];
```

### Step 2: Fix Dtype Mapping (HIGH)

Update `type_to_name()` in `mlx_matmul.mm`:

```cpp
static std::string type_to_name(const Tensor* t) {
  auto dtype = t->scalar_type();
  // TODO: Get actual ExecuTorch dtype enum values
  switch (dtype) {
    case ScalarType::Float: return "float32";
    case ScalarType::Half: return "float16";
    case ScalarType::BFloat16: return "bfloat16";
    default:
      ET_LOG(Error, "Unsupported dtype: %d", static_cast<int>(dtype));
      return "float32";
  }
}
```

### Step 3: Add Build Configuration (HIGH)

Add to `backends/apple/metal/CMakeLists.txt`:

```cmake
# Add MLX matmul implementation
target_sources(executorch_metal PRIVATE
  runtime/mlx/mlx_matmul.mm
)

# Add include directories
target_include_directories(executorch_metal PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/runtime/mlx
  ${CMAKE_CURRENT_SOURCE_DIR}/runtime/mlx/kernels
)

# Optional: Compile Metal shaders
# add_custom_command(...)
```

### Step 4: Test (HIGH)

Create a simple test:

```python
# test_mlx_matmul.py
import torch

# Small test case
a = torch.randn(32, 64, dtype=torch.float32).to("mps")
b = torch.randn(64, 128, dtype=torch.float32).to("mps")

try:
    c = torch.mm(a, b)
    c_cpu = torch.mm(a.cpu(), b.cpu())
    
    if torch.allclose(c.cpu(), c_cpu, rtol=1e-4):
        print("✅ MLX matmul works!")
    else:
        print("❌ Output mismatch")
except Exception as e:
    print(f"❌ Error: {e}")
```

## Debugging Tips

### Check Kernel Loading

Add debug logging in `init_mlx_matmul_kernels()`:

```cpp
void init_mlx_matmul_kernels() {
  ET_LOG(Info, "Loading MLX kernels...");
  std::string source = get_mlx_kernel_source();
  ET_LOG(Info, "Kernel source size: %zu bytes", source.size());
  
  g_mlx_shader_library = std::make_unique<ETMetalShaderLibrary>(source);
  ET_LOG(Info, "MLX kernels loaded successfully");
}
```

### Check Kernel Dispatch

Add logging in `steel_matmul_regular()`:

```cpp
ET_LOG(Debug, "Kernel: %s", hash_name.c_str());
ET_LOG(Debug, "Grid: [%d,%d,%d], Group: [%d,%d,%d]",
       tn, tm, batch_size_out, 32, wn, wm);
ET_LOG(Debug, "Params: M=%d, N=%d, K=%d", M, N, K);
```

### Enable Verbose Logging

Set environment variable:

```bash
export EXECUTORCH_LOG_LEVEL=Debug
```

## Common Issues & Solutions

### Issue: "Failed to get kernel function"

**Cause:** Kernel not compiled or wrong name

**Solution:**
- Check kernel source is loaded: `ET_LOG(Info, "Source: %s", source.c_str());`
- Verify kernel names match between C++ and Metal
- Ensure all includes are present in Metal source

### Issue: "Metal buffer lookup failed"

**Cause:** Tensors not on Metal device

**Solution:**
- Ensure tensors are created with `.to("mps")`
- Check `ptr_to_mtl_buffer` is populated
- Add fallback to create buffer if missing

### Issue: Compilation errors

**Cause:** Missing includes or incompatible Metal version

**Solution:**
- Add missing header includes
- Check Metal version compatibility (need Metal 3.0+)
- Review MLX's minimum requirements

## Performance Tuning (After Basic Functionality)

Once it works, optimize:

1. **Device Architecture Detection**
   - Query actual device capabilities
   - Select optimal tile sizes

2. **Batch Dimension Handling**
   - Implement batch collapsing
   - Calculate correct strides

3. **GEMV Specialization**
   - Add matrix-vector fast path
   - Use optimized GEMV kernels

4. **Split-K for Tall/Skinny Matrices**
   - Implement split-K dispatch
   - Tune partition count

## Expected Performance

After full implementation, expect vs MPS:
- **FP32:** ~1.2-1.5x faster
- **BF16:** ~1.5-2.0x faster
- **Small batch:** ~1.3-1.8x faster

## Documentation

For more details, see:
- **`backends/apple/metal/runtime/mlx/README.md`** - Complete technical documentation
- **`INTEGRATION_SUMMARY.md`** - Architecture and design
- **`MLX_INTEGRATION_DIFF.md`** - Detailed code changes

## Getting Help

If you encounter issues:

1. Check debug logs with verbose logging enabled
2. Review MLX source code for reference implementation
3. Compare with MPS implementation (in `#if 0` block)
4. Check that kernel source is loading correctly

## Status Checklist

Use this to track completion:

- [ ] Step 1: Kernel compilation working
- [ ] Step 2: Dtype mapping fixed
- [ ] Step 3: Build configuration added
- [ ] Step 4: Basic test passes
- [ ] Step 5: Batched matmul working
- [ ] Step 6: GEMV specialization added
- [ ] Step 7: Performance validated
- [ ] Step 8: Production ready

## Estimated Time to Completion

- **Kernel compilation:** 4-8 hours
- **Dtype mapping:** 1-2 hours
- **Build config:** 1-2 hours
- **Basic testing:** 2-4 hours
- **Bug fixes:** 4-8 hours
- **Enhanced features:** 8-16 hours
- **Total minimum:** ~12-24 hours
- **Total with enhancements:** ~20-40 hours

Good luck! 🚀

