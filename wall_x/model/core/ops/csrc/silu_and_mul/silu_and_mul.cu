// Re-enable CUDA half operators (PyTorch disables them)
#undef __CUDA_NO_HALF_OPERATORS__
#undef __CUDA_NO_HALF_CONVERSIONS__
#undef __CUDA_NO_HALF2_OPERATORS__
#undef __CUDA_NO_BFLOAT16_CONVERSIONS__

/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>

#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include "../common/cuda_utils.h"

namespace wallx_cuda_silu_and_mul {

static int get_sm_count() {
    static int sm_count = -1;
    if (sm_count < 0) {
        int device;
        cudaGetDevice(&device);
        cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, device);
    }
    return sm_count;
}

// Returns the stride (in elements) of dimension d.
// When strides is null (fully contiguous, row-major), computes from shape.
// When strides is non-null, reads the stored value directly.
static int64_t safe_stride(const at::Tensor& t, int d) {
    if (t.is_contiguous()) {
        int64_t s = 1;
        for (int i = t.dim() - 1; i > d; --i) s *= t.size(i);
        return s;
    }
    return t.stride(d);
}

// ============================================================
// Kernel 1: fully-contiguous path (used by non-inplace SiluAndMul)
// Persistent: launched with sm_count blocks, each loops with stride = gridDim.x * blockDim.x.
// output[i] = silu(input1[i]) * input2[i]
// ============================================================
template<typename T>
__global__ void siluAndMulPersistentKernel(
    T* __restrict__ output,
    const T* __restrict__ input1,
    const T* __restrict__ input2,
    int64_t vec_n  // total elements / VEC_SIZE
) {
    using Vec = typename VecTraits<T>::Type;
    constexpr int VEC_SIZE = VecTraits<T>::vec_size;

    const int64_t tid = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    const int64_t stride = (int64_t)gridDim.x * blockDim.x;

    for (int64_t i = tid; i < vec_n; i += stride) {
        Vec x = VecTraits<T>::load(input1 + i * VEC_SIZE);
        Vec g = VecTraits<T>::load(input2 + i * VEC_SIZE);
        Vec out;

        if constexpr (std::is_same_v<T, float>) {
            out.x = x.x / (1.0f + __expf(-x.x)) * g.x;
            out.y = x.y / (1.0f + __expf(-x.y)) * g.y;
            out.z = x.z / (1.0f + __expf(-x.z)) * g.z;
            out.w = x.w / (1.0f + __expf(-x.w)) * g.w;
        } else if constexpr (std::is_same_v<T, __half>) {
            const float x0 = __half2float(x.x), x1 = __half2float(x.y);
            const float g0 = __half2float(g.x), g1 = __half2float(g.y);
            out.x = __float2half(x0 / (1.0f + __expf(-x0)) * g0);
            out.y = __float2half(x1 / (1.0f + __expf(-x1)) * g1);
        } else {
            const float x0 = __bfloat162float(x.x), x1 = __bfloat162float(x.y);
            const float g0 = __bfloat162float(g.x), g1 = __bfloat162float(g.y);
            out.x = __float2bfloat16(x0 / (1.0f + __expf(-x0)) * g0);
            out.y = __float2bfloat16(x1 / (1.0f + __expf(-x1)) * g1);
        }

        VecTraits<T>::store(output + i * VEC_SIZE, out);
    }
}

// ============================================================
// Kernel 2: last-dim-contiguous path (used by inplace SiluAndMulInplace)
// Requirements: stride(ndim-1) == 1 for both tensors; outer dims may be non-contiguous.
// Each work item maps to a (row, col_vec) pair.
// Row offset for each tensor is computed from its outer strides.
// ============================================================
constexpr int MAX_OUTER_DIMS = 7;  // supports tensors up to 8-D

struct SiluMulInplaceArgs {
    int64_t strides1[MAX_OUTER_DIMS];  // outer-dim strides of inout1,  in elements
    int64_t strides2[MAX_OUTER_DIMS];  // outer-dim strides of input2,  in elements
    int64_t shapes[MAX_OUTER_DIMS];    // outer-dim sizes (same for both tensors)
    int outer_ndim;                    // ndim - 1  (0 for 1-D tensors)
};

struct SiluMulSplitArgs {
    int64_t output_strides[MAX_OUTER_DIMS];   // outer-dim strides of output
    int64_t gate_up_strides[MAX_OUTER_DIMS];  // outer-dim strides of gate_up
    int64_t shapes[MAX_OUTER_DIMS];           // outer-dim sizes
    int outer_ndim;                           // ndim - 1
};

template<typename T>
__global__ void siluAndMulInplacePersistentKernel(
    T* __restrict__ inout1,
    const T* __restrict__ input2,
    SiluMulInplaceArgs args,
    int64_t num_rows,   // product of outer dims
    int64_t vec_last    // last_dim / VEC_SIZE
) {
    using Vec = typename VecTraits<T>::Type;
    constexpr int VEC_SIZE = VecTraits<T>::vec_size;

    const int64_t tid = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    const int64_t grid_stride = (int64_t)gridDim.x * blockDim.x;
    const int64_t total_vecs = num_rows * vec_last;

    for (int64_t vi = tid; vi < total_vecs; vi += grid_stride) {
        const int64_t row = vi / vec_last;
        const int64_t col_vec = vi % vec_last;

        // Compute per-row pointer offsets by decomposing the flat row index
        // into per-dimension indices and accumulating strides.
        int64_t offset1 = 0, offset2 = 0;
        int64_t r = row;
        for (int d = args.outer_ndim - 1; d >= 0; --d) {
            const int64_t idx = r % args.shapes[d];
            r /= args.shapes[d];
            offset1 += idx * args.strides1[d];
            offset2 += idx * args.strides2[d];
        }

        T* ptr1 = inout1 + offset1 + col_vec * VEC_SIZE;
        const T* ptr2 = input2 + offset2 + col_vec * VEC_SIZE;

        Vec x = VecTraits<T>::load(ptr1);
        Vec g = VecTraits<T>::load(ptr2);
        Vec out;

        if constexpr (std::is_same_v<T, float>) {
            out.x = x.x / (1.0f + __expf(-x.x)) * g.x;
            out.y = x.y / (1.0f + __expf(-x.y)) * g.y;
            out.z = x.z / (1.0f + __expf(-x.z)) * g.z;
            out.w = x.w / (1.0f + __expf(-x.w)) * g.w;
        } else if constexpr (std::is_same_v<T, __half>) {
            const float x0 = __half2float(x.x), x1 = __half2float(x.y);
            const float g0 = __half2float(g.x), g1 = __half2float(g.y);
            out.x = __float2half(x0 / (1.0f + __expf(-x0)) * g0);
            out.y = __float2half(x1 / (1.0f + __expf(-x1)) * g1);
        } else {
            const float x0 = __bfloat162float(x.x), x1 = __bfloat162float(x.y);
            const float g0 = __bfloat162float(g.x), g1 = __bfloat162float(g.y);
            out.x = __float2bfloat16(x0 / (1.0f + __expf(-x0)) * g0);
            out.y = __float2bfloat16(x1 / (1.0f + __expf(-x1)) * g1);
        }

        VecTraits<T>::store(ptr1, out);
    }
}

// ============================================================
// Helpers
// ============================================================

static int64_t get_total_elements(const at::Tensor& t) {
    int64_t n = 1;
    for (int i = 0; i < t.dim(); ++i) n *= t.size(i);
    return n;
}

static void dispatch_contiguous(
    void* output_ptr,
    const void* input1_ptr,
    const void* input2_ptr,
    int64_t n,
    const at::Tensor& ref_tensor,
    cudaStream_t stream)
{
    const int blockSize = 256;
    const int num_blocks = get_sm_count();

    if (ref_tensor.scalar_type() == at::kFloat) {
        ASSERT_CHECK(n % VecTraits<float>::vec_size == 0);
        siluAndMulPersistentKernel<float><<<num_blocks, blockSize, 0, stream>>>(
            static_cast<float*>(output_ptr),
            static_cast<const float*>(input1_ptr),
            static_cast<const float*>(input2_ptr),
            n / VecTraits<float>::vec_size);
    } else if (ref_tensor.scalar_type() == at::kHalf) {
        ASSERT_CHECK(n % VecTraits<__half>::vec_size == 0);
        siluAndMulPersistentKernel<__half><<<num_blocks, blockSize, 0, stream>>>(
            static_cast<__half*>(output_ptr),
            static_cast<const __half*>(input1_ptr),
            static_cast<const __half*>(input2_ptr),
            n / VecTraits<__half>::vec_size);
    } else if (ref_tensor.scalar_type() == at::kBFloat16) {
        ASSERT_CHECK(n % VecTraits<__nv_bfloat16>::vec_size == 0);
        siluAndMulPersistentKernel<__nv_bfloat16><<<num_blocks, blockSize, 0, stream>>>(
            static_cast<__nv_bfloat16*>(output_ptr),
            static_cast<const __nv_bfloat16*>(input1_ptr),
            static_cast<const __nv_bfloat16*>(input2_ptr),
            n / VecTraits<__nv_bfloat16>::vec_size);
    } else {
        throw std::runtime_error("silu_and_mul: unsupported dtype");
    }
}

template<typename T>
static void dispatch_inplace_strided(
    T* inout1,
    const T* input2,
    const SiluMulInplaceArgs& args,
    int64_t num_rows,
    int64_t vec_last,
    cudaStream_t stream)
{
    const int blockSize = 256;
    const int num_blocks = get_sm_count();
    siluAndMulInplacePersistentKernel<T><<<num_blocks, blockSize, 0, stream>>>(
        inout1, input2, args, num_rows, vec_last);
}

template<typename T>
__global__ void siluAndMulSplitPersistentKernel(
    T* __restrict__ output,
    const T* __restrict__ gate_up,
    SiluMulSplitArgs args,
    int64_t num_rows,
    int64_t half_vec)
{
    using Vec = typename VecTraits<T>::Type;
    constexpr int VEC_SIZE = VecTraits<T>::vec_size;
    const int64_t intermediate_size = half_vec * VEC_SIZE;

    const int64_t col_vec = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (col_vec >= half_vec) return;

    #pragma unroll 4
    for (int64_t row = (int64_t)blockIdx.y; row < num_rows; row += (int64_t)gridDim.y) {
        int64_t output_offset = 0;
        int64_t gate_up_offset = 0;
        if (args.outer_ndim == 1) {
            output_offset = row * args.output_strides[0];
            gate_up_offset = row * args.gate_up_strides[0];
        } else if (args.outer_ndim > 1) {
            int64_t r = row;
            #pragma unroll MAX_OUTER_DIMS
            for (int d = MAX_OUTER_DIMS - 1; d >= 0; --d) {
                if (d < args.outer_ndim) {
                    const int64_t idx = r % args.shapes[d];
                    r /= args.shapes[d];
                    output_offset += idx * args.output_strides[d];
                    gate_up_offset += idx * args.gate_up_strides[d];
                }
            }
        }

        T* out_ptr = output + output_offset + col_vec * VEC_SIZE;
        const T* gate_ptr = gate_up + gate_up_offset + col_vec * VEC_SIZE;
        const T* up_ptr = gate_up + gate_up_offset + intermediate_size + col_vec * VEC_SIZE;

        Vec x = VecTraits<T>::load(gate_ptr);
        Vec g = VecTraits<T>::load(up_ptr);
        Vec out;

        if constexpr (std::is_same_v<T, float>) {
            out.x = x.x / (1.0f + __expf(-x.x)) * g.x;
            out.y = x.y / (1.0f + __expf(-x.y)) * g.y;
            out.z = x.z / (1.0f + __expf(-x.z)) * g.z;
            out.w = x.w / (1.0f + __expf(-x.w)) * g.w;
        } else if constexpr (std::is_same_v<T, __half>) {
            const float x0 = __half2float(x.x), x1 = __half2float(x.y);
            const float g0 = __half2float(g.x), g1 = __half2float(g.y);
            out.x = __float2half(x0 / (1.0f + __expf(-x0)) * g0);
            out.y = __float2half(x1 / (1.0f + __expf(-x1)) * g1);
        } else {
            const float x0 = __bfloat162float(x.x), x1 = __bfloat162float(x.y);
            const float g0 = __bfloat162float(g.x), g1 = __bfloat162float(g.y);
            out.x = __float2bfloat16(x0 / (1.0f + __expf(-x0)) * g0);
            out.y = __float2bfloat16(x1 / (1.0f + __expf(-x1)) * g1);
        }

        VecTraits<T>::store(out_ptr, out);
    }
}

template<typename T>
static void dispatch_split_strided(
    T* output,
    const T* gate_up,
    const SiluMulSplitArgs& args,
    int64_t num_rows,
    int64_t half_vec,
    cudaStream_t stream)
{
    constexpr int BLOCK_X = 256;
    const int grid_x = (int)((half_vec + BLOCK_X - 1) / BLOCK_X);
    const int grid_y = (int)std::min(num_rows, (int64_t)get_sm_count());
    siluAndMulSplitPersistentKernel<T><<<dim3(grid_x, grid_y), BLOCK_X, 0, stream>>>(
        output, gate_up, args, num_rows, half_vec);
}

// ============================================================
// Public functions
// ============================================================

// Non-inplace: output = silu(input1) * input2  (all tensors must be fully contiguous)
void SiluAndMul(
    const at::Tensor& output,
    const at::Tensor& input1,
    const at::Tensor& input2)
{
    ASSERT_CHECK(output.is_contiguous());
    ASSERT_CHECK(input1.is_contiguous());
    ASSERT_CHECK(input2.is_contiguous());
    ASSERT_CHECK(output.dim() == input1.dim());
    ASSERT_CHECK(input1.dim() == input2.dim());
    for (int i = 0; i < input1.dim(); ++i) {
        ASSERT_CHECK(output.size(i) == input1.size(i));
        ASSERT_CHECK(input1.size(i) == input2.size(i));
    }

    const int64_t n = get_total_elements(input1);
    cudaStream_t stream = static_cast<cudaStream_t>(
        at::cuda::getCurrentCUDAStream().stream());

    dispatch_contiguous(output.data_ptr(), input1.data_ptr(), input2.data_ptr(), n, input1, stream);
    sync_check_cuda_error();
}

// Inplace: input1 = silu(input1) * input2
// Only requires the last dimension to be contiguous (stride[-1] == 1).
// Outer dimensions may be non-contiguous; their offsets are computed via strides.
void SiluAndMulInplace(
    const at::Tensor& input1,
    const at::Tensor& input2)
{
    const int ndim = input1.dim();
    ASSERT_CHECK(ndim >= 1);
    ASSERT_CHECK(ndim == input2.dim());
    ASSERT_CHECK(ndim - 1 <= MAX_OUTER_DIMS);

    // Shape must match on every dimension.
    for (int i = 0; i < ndim; ++i) {
        ASSERT_CHECK(input1.size(i) == input2.size(i));
    }

    // Only the last dimension must be contiguous.
    ASSERT_CHECK(safe_stride(input1, ndim - 1) == 1);
    ASSERT_CHECK(safe_stride(input2, ndim - 1) == 1);

    const int64_t last_dim = input1.size(ndim - 1);
    const int outer_ndim = ndim - 1;

    // Build kernel args: collect outer-dim strides and shapes.
    SiluMulInplaceArgs args{};
    args.outer_ndim = outer_ndim;
    int64_t num_rows = 1;
    for (int d = 0; d < outer_ndim; ++d) {
        args.strides1[d] = safe_stride(input1, d);
        args.strides2[d] = safe_stride(input2, d);
        args.shapes[d]   = input1.size(d);
        num_rows *= input1.size(d);
    }

    cudaStream_t stream = static_cast<cudaStream_t>(
        at::cuda::getCurrentCUDAStream().stream());

    if (input1.scalar_type() == at::kFloat) {
        ASSERT_CHECK(last_dim % VecTraits<float>::vec_size == 0);
        dispatch_inplace_strided(
            static_cast<float*>(input1.data_ptr()),
            static_cast<const float*>(input2.data_ptr()),
            args, num_rows, last_dim / VecTraits<float>::vec_size, stream);
    } else if (input1.scalar_type() == at::kHalf) {
        ASSERT_CHECK(last_dim % VecTraits<__half>::vec_size == 0);
        dispatch_inplace_strided(
            static_cast<__half*>(input1.data_ptr()),
            static_cast<const __half*>(input2.data_ptr()),
            args, num_rows, last_dim / VecTraits<__half>::vec_size, stream);
    } else if (input1.scalar_type() == at::kBFloat16) {
        ASSERT_CHECK(last_dim % VecTraits<__nv_bfloat16>::vec_size == 0);
        dispatch_inplace_strided(
            static_cast<__nv_bfloat16*>(input1.data_ptr()),
            static_cast<const __nv_bfloat16*>(input2.data_ptr()),
            args, num_rows, last_dim / VecTraits<__nv_bfloat16>::vec_size, stream);
    } else {
        throw std::runtime_error("silu_and_mul_inplace: unsupported dtype");
    }

    sync_check_cuda_error();
}

// ============================================================
// Kernel 3: split+silu+mul inplace  (2-D grid, division-free hot path)
//
// Grid layout:
//   blockIdx.x * blockDim.x + threadIdx.x  →  col_vec ∈ [0, half_vec)
//   blockIdx.y  (grid-stride over rows)     →  row    ∈ [0, num_rows)
//
// This eliminates the per-iteration  vi/half_vec  and  vi%half_vec
// 64-bit integer divisions present in the original 1-D scheme.
//
// Input:  gate_up [..., 2*N], last dim contiguous (stride[-1] == 1).
// Output (inplace): gate_up[..., :N] = silu(gate_up[..., :N]) * gate_up[..., N:]
// ============================================================
struct SiluMulSplitInplaceArgs {
    int64_t outer_strides[MAX_OUTER_DIMS];  // outer-dim strides, in elements
    int64_t shapes[MAX_OUTER_DIMS];         // outer-dim sizes
    int outer_ndim;                         // ndim - 1
};

struct SiluMulSplitBackwardArgs {
    int64_t grad_gate_up_strides[MAX_OUTER_DIMS];  // outer-dim strides of grad_gate_up
    int64_t grad_output_strides[MAX_OUTER_DIMS];   // outer-dim strides of grad_output
    int64_t gate_up_strides[MAX_OUTER_DIMS];       // outer-dim strides of original gate_up
    int64_t shapes[MAX_OUTER_DIMS];                // outer-dim sizes
    int outer_ndim;                                // ndim - 1
};

// Load a vector through the read-only data cache (__ldg).
// up_ptr shares the same base allocation as inout but is never written,
// so routing it through the texture/RO path is both safe and faster.
template<typename T>
__device__ __forceinline__ typename VecTraits<T>::Type ldg_vec(const T* ptr) {
    using Vec = typename VecTraits<T>::Type;
    return __ldg(reinterpret_cast<const Vec*>(ptr));
}

template<typename T>
__global__ void siluAndMulSplitInplacePersistentKernel(
    T* __restrict__ inout,
    SiluMulSplitInplaceArgs args,
    int64_t num_rows,
    int64_t half_vec        // intermediate_size / VEC_SIZE
) {
    using Vec = typename VecTraits<T>::Type;
    constexpr int VEC_SIZE = VecTraits<T>::vec_size;
    // Derive intermediate_size inside the kernel — avoids one extra parameter.
    const int64_t intermediate_size = half_vec * VEC_SIZE;

    const int64_t col_vec = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (col_vec >= half_vec) return;    // guard last partial block

    // Unroll by 2: allows the compiler to interleave memory loads from two
    // consecutive rows, hiding global-memory latency at the cost of ~2x
    // register pressure on the row-local variables.
    #pragma unroll 4
    for (int64_t row = (int64_t)blockIdx.y; row < num_rows; row += (int64_t)gridDim.y) {
        // ------------------------------------------------------------------
        // Row offset computation
        //   outer_ndim == 0  →  always row_offset = 0 (1-D tensor)
        //   outer_ndim == 1  →  single multiply, no division  (most common)
        //   outer_ndim  > 1  →  fully-unrolled decomposition loop
        // ------------------------------------------------------------------
        int64_t row_offset = 0;
        if (args.outer_ndim == 1) {
            row_offset = row * args.outer_strides[0];
        } else if (args.outer_ndim > 1) {
            int64_t r = row;
            // Unroll over the compile-time maximum depth; the runtime guard
            // `d < args.outer_ndim` is loop-invariant and uniform across the
            // warp, so the compiler emits predicated instructions with no
            // divergence.
            #pragma unroll MAX_OUTER_DIMS
            for (int d = MAX_OUTER_DIMS - 1; d >= 0; --d) {
                if (d < args.outer_ndim) {
                    const int64_t idx = r % args.shapes[d];
                    r /= args.shapes[d];
                    row_offset += idx * args.outer_strides[d];
                }
            }
        }

        T*       gate_ptr = inout + row_offset + col_vec * VEC_SIZE;
        const T* up_ptr   = inout + row_offset + intermediate_size + col_vec * VEC_SIZE;

        Vec x = VecTraits<T>::load(gate_ptr);
        Vec g = ldg_vec<T>(up_ptr);     // read-only: steer through RO cache
        Vec out;

        if constexpr (std::is_same_v<T, float>) {
            // float4 — 4 lanes, explicitly unrolled
            out.x = x.x / (1.0f + __expf(-x.x)) * g.x;
            out.y = x.y / (1.0f + __expf(-x.y)) * g.y;
            out.z = x.z / (1.0f + __expf(-x.z)) * g.z;
            out.w = x.w / (1.0f + __expf(-x.w)) * g.w;
        } else if constexpr (std::is_same_v<T, __half>) {
            // half2 — 2 lanes, explicitly unrolled
            const float x0 = __half2float(x.x), x1 = __half2float(x.y);
            const float g0 = __half2float(g.x), g1 = __half2float(g.y);
            out.x = __float2half(x0 / (1.0f + __expf(-x0)) * g0);
            out.y = __float2half(x1 / (1.0f + __expf(-x1)) * g1);
        } else {
            // bfloat162 — 2 lanes, explicitly unrolled
            const float x0 = __bfloat162float(x.x), x1 = __bfloat162float(x.y);
            const float g0 = __bfloat162float(g.x), g1 = __bfloat162float(g.y);
            out.x = __float2bfloat16(x0 / (1.0f + __expf(-x0)) * g0);
            out.y = __float2bfloat16(x1 / (1.0f + __expf(-x1)) * g1);
        }

        VecTraits<T>::store(gate_ptr, out);
    }
}

template<typename T>
static void dispatch_split_inplace_strided(
    T* inout,
    const SiluMulSplitInplaceArgs& args,
    int64_t num_rows,
    int64_t half_vec,
    cudaStream_t stream)
{
    constexpr int BLOCK_X = 256;
    // grid_x covers the col_vec (intermediate) dimension exactly.
    // grid_y covers rows up to sm_count; the kernel loops with stride gridDim.y.
    const int grid_x = (int)((half_vec + BLOCK_X - 1) / BLOCK_X);
    const int grid_y = (int)std::min(num_rows, (int64_t)get_sm_count());
    siluAndMulSplitInplacePersistentKernel<T><<<dim3(grid_x, grid_y), BLOCK_X, 0, stream>>>(
        inout, args, num_rows, half_vec);
}

template<typename T>
__device__ __forceinline__ float to_float(T x);

template<>
__device__ __forceinline__ float to_float<float>(float x) {
    return x;
}

template<>
__device__ __forceinline__ float to_float<__half>(__half x) {
    return __half2float(x);
}

template<>
__device__ __forceinline__ float to_float<__nv_bfloat16>(__nv_bfloat16 x) {
    return __bfloat162float(x);
}

template<typename T>
__device__ __forceinline__ T from_float(float x);

template<>
__device__ __forceinline__ float from_float<float>(float x) {
    return x;
}

template<>
__device__ __forceinline__ __half from_float<__half>(float x) {
    return __float2half(x);
}

template<>
__device__ __forceinline__ __nv_bfloat16 from_float<__nv_bfloat16>(float x) {
    return __float2bfloat16(x);
}

template<typename T>
__device__ __forceinline__ void silu_mul_backward_lane(
    T gate,
    T up,
    T grad_output,
    T& grad_gate,
    T& grad_up)
{
    const float gate_f = to_float<T>(gate);
    const float up_f = to_float<T>(up);
    const float grad_f = to_float<T>(grad_output);
    const float sigmoid = 1.0f / (1.0f + __expf(-gate_f));
    const float silu = gate_f * sigmoid;
    const float silu_grad = sigmoid * (1.0f + gate_f * (1.0f - sigmoid));

    grad_gate = from_float<T>(grad_f * up_f * silu_grad);
    grad_up = from_float<T>(grad_f * silu);
}

template<typename T>
__device__ __forceinline__ void silu_mul_backward_vec(
    typename VecTraits<T>::Type gate,
    typename VecTraits<T>::Type up,
    typename VecTraits<T>::Type grad_output,
    typename VecTraits<T>::Type& grad_gate,
    typename VecTraits<T>::Type& grad_up)
{
    silu_mul_backward_lane<T>(gate.x, up.x, grad_output.x, grad_gate.x, grad_up.x);
    silu_mul_backward_lane<T>(gate.y, up.y, grad_output.y, grad_gate.y, grad_up.y);

    if constexpr (std::is_same_v<T, float>) {
        silu_mul_backward_lane<T>(gate.z, up.z, grad_output.z, grad_gate.z, grad_up.z);
        silu_mul_backward_lane<T>(gate.w, up.w, grad_output.w, grad_gate.w, grad_up.w);
    }
}

// Backward for split+silu+mul inplace.
// Given original gate_up [..., 2*N] and grad_output [..., N], writes
// grad_gate_up[..., :N] = grad_output * up * d_silu(gate)
// grad_gate_up[..., N:] = grad_output * silu(gate)
template<typename T>
__global__ void siluAndMulSplitInplaceBackwardPersistentKernel(
    T* __restrict__ grad_gate_up,
    const T* __restrict__ grad_output,
    const T* __restrict__ gate_up,
    SiluMulSplitBackwardArgs args,
    int64_t num_rows,
    int64_t half_vec)
{
    using Vec = typename VecTraits<T>::Type;
    constexpr int VEC_SIZE = VecTraits<T>::vec_size;
    const int64_t intermediate_size = half_vec * VEC_SIZE;

    const int64_t col_vec = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (col_vec >= half_vec) return;

    #pragma unroll 4
    for (int64_t row = (int64_t)blockIdx.y; row < num_rows; row += (int64_t)gridDim.y) {
        int64_t grad_gate_up_offset = 0;
        int64_t grad_output_offset = 0;
        int64_t gate_up_offset = 0;

        if (args.outer_ndim == 1) {
            grad_gate_up_offset = row * args.grad_gate_up_strides[0];
            grad_output_offset = row * args.grad_output_strides[0];
            gate_up_offset = row * args.gate_up_strides[0];
        } else if (args.outer_ndim > 1) {
            int64_t r = row;
            #pragma unroll MAX_OUTER_DIMS
            for (int d = MAX_OUTER_DIMS - 1; d >= 0; --d) {
                if (d < args.outer_ndim) {
                    const int64_t idx = r % args.shapes[d];
                    r /= args.shapes[d];
                    grad_gate_up_offset += idx * args.grad_gate_up_strides[d];
                    grad_output_offset += idx * args.grad_output_strides[d];
                    gate_up_offset += idx * args.gate_up_strides[d];
                }
            }
        }

        T* grad_gate_ptr = grad_gate_up + grad_gate_up_offset + col_vec * VEC_SIZE;
        T* grad_up_ptr = grad_gate_up + grad_gate_up_offset + intermediate_size + col_vec * VEC_SIZE;
        const T* grad_output_ptr = grad_output + grad_output_offset + col_vec * VEC_SIZE;
        const T* gate_ptr = gate_up + gate_up_offset + col_vec * VEC_SIZE;
        const T* up_ptr = gate_up + gate_up_offset + intermediate_size + col_vec * VEC_SIZE;

        Vec gate = VecTraits<T>::load(gate_ptr);
        Vec up = VecTraits<T>::load(up_ptr);
        Vec grad = VecTraits<T>::load(grad_output_ptr);
        Vec grad_gate;
        Vec grad_up;

        silu_mul_backward_vec<T>(gate, up, grad, grad_gate, grad_up);

        VecTraits<T>::store(grad_gate_ptr, grad_gate);
        VecTraits<T>::store(grad_up_ptr, grad_up);
    }
}

template<typename T>
__global__ void siluAndMulBackwardPersistentKernel(
    T* __restrict__ grad_input1,
    T* __restrict__ grad_input2,
    const T* __restrict__ grad_output,
    const T* __restrict__ input1,
    const T* __restrict__ input2,
    int64_t vec_n)
{
    using Vec = typename VecTraits<T>::Type;
    constexpr int VEC_SIZE = VecTraits<T>::vec_size;

    const int64_t tid = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    const int64_t stride = (int64_t)gridDim.x * blockDim.x;

    for (int64_t i = tid; i < vec_n; i += stride) {
        Vec gate = VecTraits<T>::load(input1 + i * VEC_SIZE);
        Vec up = VecTraits<T>::load(input2 + i * VEC_SIZE);
        Vec grad = VecTraits<T>::load(grad_output + i * VEC_SIZE);
        Vec grad_gate;
        Vec grad_up;

        silu_mul_backward_vec<T>(gate, up, grad, grad_gate, grad_up);

        VecTraits<T>::store(grad_input1 + i * VEC_SIZE, grad_gate);
        VecTraits<T>::store(grad_input2 + i * VEC_SIZE, grad_up);
    }
}

static void dispatch_contiguous_backward(
    void* grad_input1_ptr,
    void* grad_input2_ptr,
    const void* grad_output_ptr,
    const void* input1_ptr,
    const void* input2_ptr,
    int64_t n,
    const at::Tensor& ref_tensor,
    cudaStream_t stream)
{
    const int blockSize = 256;
    const int num_blocks = get_sm_count();

    if (ref_tensor.scalar_type() == at::kFloat) {
        ASSERT_CHECK(n % VecTraits<float>::vec_size == 0);
        siluAndMulBackwardPersistentKernel<float><<<num_blocks, blockSize, 0, stream>>>(
            static_cast<float*>(grad_input1_ptr),
            static_cast<float*>(grad_input2_ptr),
            static_cast<const float*>(grad_output_ptr),
            static_cast<const float*>(input1_ptr),
            static_cast<const float*>(input2_ptr),
            n / VecTraits<float>::vec_size);
    } else if (ref_tensor.scalar_type() == at::kHalf) {
        ASSERT_CHECK(n % VecTraits<__half>::vec_size == 0);
        siluAndMulBackwardPersistentKernel<__half><<<num_blocks, blockSize, 0, stream>>>(
            static_cast<__half*>(grad_input1_ptr),
            static_cast<__half*>(grad_input2_ptr),
            static_cast<const __half*>(grad_output_ptr),
            static_cast<const __half*>(input1_ptr),
            static_cast<const __half*>(input2_ptr),
            n / VecTraits<__half>::vec_size);
    } else if (ref_tensor.scalar_type() == at::kBFloat16) {
        ASSERT_CHECK(n % VecTraits<__nv_bfloat16>::vec_size == 0);
        siluAndMulBackwardPersistentKernel<__nv_bfloat16><<<num_blocks, blockSize, 0, stream>>>(
            static_cast<__nv_bfloat16*>(grad_input1_ptr),
            static_cast<__nv_bfloat16*>(grad_input2_ptr),
            static_cast<const __nv_bfloat16*>(grad_output_ptr),
            static_cast<const __nv_bfloat16*>(input1_ptr),
            static_cast<const __nv_bfloat16*>(input2_ptr),
            n / VecTraits<__nv_bfloat16>::vec_size);
    } else {
        throw std::runtime_error("silu_and_mul_backward: unsupported dtype");
    }
}

template<typename T>
static void dispatch_split_inplace_backward_strided(
    T* grad_gate_up,
    const T* grad_output,
    const T* gate_up,
    const SiluMulSplitBackwardArgs& args,
    int64_t num_rows,
    int64_t half_vec,
    cudaStream_t stream)
{
    constexpr int BLOCK_X = 256;
    const int grid_x = (int)((half_vec + BLOCK_X - 1) / BLOCK_X);
    const int grid_y = (int)std::min(num_rows, (int64_t)get_sm_count());
    siluAndMulSplitInplaceBackwardPersistentKernel<T><<<dim3(grid_x, grid_y), BLOCK_X, 0, stream>>>(
        grad_gate_up, grad_output, gate_up, args, num_rows, half_vec);
}

// SiluAndMulSplit: fused split + silu + mul, out-of-place.
// gate_up has shape [..., 2*N], output has shape [..., N].
// Only the last dimension must be contiguous; outer dimensions may be strided.
void SiluAndMulSplit(
    const at::Tensor& output,
    const at::Tensor& gate_up)
{
    const int ndim = gate_up.dim();
    ASSERT_CHECK(ndim >= 1);
    ASSERT_CHECK(ndim == output.dim());
    ASSERT_CHECK(ndim - 1 <= MAX_OUTER_DIMS);
    ASSERT_CHECK(safe_stride(gate_up, ndim - 1) == 1);
    ASSERT_CHECK(safe_stride(output, ndim - 1) == 1);

    const int64_t last_dim = gate_up.size(ndim - 1);
    ASSERT_CHECK(last_dim % 2 == 0);
    const int64_t intermediate_size = last_dim / 2;
    ASSERT_CHECK(output.size(ndim - 1) == intermediate_size);

    const int outer_ndim = ndim - 1;
    SiluMulSplitArgs args{};
    args.outer_ndim = outer_ndim;
    int64_t num_rows = 1;
    for (int d = 0; d < outer_ndim; ++d) {
        ASSERT_CHECK(output.size(d) == gate_up.size(d));
        args.output_strides[d] = safe_stride(output, d);
        args.gate_up_strides[d] = safe_stride(gate_up, d);
        args.shapes[d] = gate_up.size(d);
        num_rows *= gate_up.size(d);
    }

    ASSERT_CHECK(output.scalar_type() == gate_up.scalar_type());
    ASSERT_CHECK(output.element_size() == gate_up.element_size());

    cudaStream_t stream = static_cast<cudaStream_t>(
        at::cuda::getCurrentCUDAStream().stream());

    if (gate_up.scalar_type() == at::kFloat) {
        ASSERT_CHECK(intermediate_size % VecTraits<float>::vec_size == 0);
        dispatch_split_strided(
            static_cast<float*>(output.data_ptr()),
            static_cast<const float*>(gate_up.data_ptr()),
            args, num_rows, intermediate_size / VecTraits<float>::vec_size, stream);
    } else if (gate_up.scalar_type() == at::kHalf) {
        ASSERT_CHECK(intermediate_size % VecTraits<__half>::vec_size == 0);
        dispatch_split_strided(
            static_cast<__half*>(output.data_ptr()),
            static_cast<const __half*>(gate_up.data_ptr()),
            args, num_rows, intermediate_size / VecTraits<__half>::vec_size, stream);
    } else if (gate_up.scalar_type() == at::kBFloat16) {
        ASSERT_CHECK(intermediate_size % VecTraits<__nv_bfloat16>::vec_size == 0);
        dispatch_split_strided(
            static_cast<__nv_bfloat16*>(output.data_ptr()),
            static_cast<const __nv_bfloat16*>(gate_up.data_ptr()),
            args, num_rows, intermediate_size / VecTraits<__nv_bfloat16>::vec_size, stream);
    } else {
        throw std::runtime_error("silu_and_mul_split: unsupported dtype");
    }

    sync_check_cuda_error();
}

// SiluAndMulSplitInplace: fused split + silu + mul, inplace.
// gate_up has shape [..., 2*N], last dim must be contiguous and even.
// After this call, gate_up[..., :N] = silu(gate_up[..., :N]) * gate_up[..., N:].
void SiluAndMulSplitInplace(const at::Tensor& gate_up)
{
    const int ndim = gate_up.dim();
    ASSERT_CHECK(ndim >= 1);
    ASSERT_CHECK(ndim - 1 <= MAX_OUTER_DIMS);
    ASSERT_CHECK(safe_stride(gate_up, ndim - 1) == 1);

    const int64_t last_dim = gate_up.size(ndim - 1);
    ASSERT_CHECK(last_dim % 2 == 0);
    const int64_t intermediate_size = last_dim / 2;

    const int outer_ndim = ndim - 1;
    SiluMulSplitInplaceArgs args{};
    args.outer_ndim = outer_ndim;
    int64_t num_rows = 1;
    for (int d = 0; d < outer_ndim; ++d) {
        args.outer_strides[d] = safe_stride(gate_up, d);
        args.shapes[d]        = gate_up.size(d);
        num_rows *= gate_up.size(d);
    }

    cudaStream_t stream = static_cast<cudaStream_t>(
        at::cuda::getCurrentCUDAStream().stream());

    if (gate_up.scalar_type() == at::kFloat) {
        ASSERT_CHECK(intermediate_size % VecTraits<float>::vec_size == 0);
        dispatch_split_inplace_strided(
            static_cast<float*>(gate_up.data_ptr()),
            args, num_rows, intermediate_size / VecTraits<float>::vec_size, stream);
    } else if (gate_up.scalar_type() == at::kHalf) {
        ASSERT_CHECK(intermediate_size % VecTraits<__half>::vec_size == 0);
        dispatch_split_inplace_strided(
            static_cast<__half*>(gate_up.data_ptr()),
            args, num_rows, intermediate_size / VecTraits<__half>::vec_size, stream);
    } else if (gate_up.scalar_type() == at::kBFloat16) {
        ASSERT_CHECK(intermediate_size % VecTraits<__nv_bfloat16>::vec_size == 0);
        dispatch_split_inplace_strided(
            static_cast<__nv_bfloat16*>(gate_up.data_ptr()),
            args, num_rows, intermediate_size / VecTraits<__nv_bfloat16>::vec_size, stream);
    } else {
        throw std::runtime_error("silu_and_mul_split_inplace: unsupported dtype");
    }

    sync_check_cuda_error();
}

void SiluAndMulSplitInplaceBackward(
    const at::Tensor& grad_gate_up,
    const at::Tensor& grad_output,
    const at::Tensor& gate_up)
{
    const int ndim = gate_up.dim();
    ASSERT_CHECK(ndim >= 1);
    ASSERT_CHECK(ndim == grad_gate_up.dim());
    ASSERT_CHECK(ndim == grad_output.dim());
    ASSERT_CHECK(ndim - 1 <= MAX_OUTER_DIMS);

    ASSERT_CHECK(safe_stride(grad_gate_up, ndim - 1) == 1);
    ASSERT_CHECK(safe_stride(grad_output, ndim - 1) == 1);
    ASSERT_CHECK(safe_stride(gate_up, ndim - 1) == 1);

    const int64_t last_dim = gate_up.size(ndim - 1);
    ASSERT_CHECK(last_dim % 2 == 0);
    const int64_t intermediate_size = last_dim / 2;

    for (int d = 0; d < ndim - 1; ++d) {
        ASSERT_CHECK(grad_gate_up.size(d) == gate_up.size(d));
        ASSERT_CHECK(grad_output.size(d) == gate_up.size(d));
    }
    ASSERT_CHECK(grad_gate_up.size(ndim - 1) == last_dim);
    ASSERT_CHECK(grad_output.size(ndim - 1) == intermediate_size);

    ASSERT_CHECK(grad_gate_up.scalar_type() == gate_up.scalar_type());
    ASSERT_CHECK(grad_gate_up.element_size() == gate_up.element_size());
    ASSERT_CHECK(grad_output.scalar_type() == gate_up.scalar_type());
    ASSERT_CHECK(grad_output.element_size() == gate_up.element_size());

    const int outer_ndim = ndim - 1;
    SiluMulSplitBackwardArgs args{};
    args.outer_ndim = outer_ndim;
    int64_t num_rows = 1;
    for (int d = 0; d < outer_ndim; ++d) {
        args.grad_gate_up_strides[d] = safe_stride(grad_gate_up, d);
        args.grad_output_strides[d] = safe_stride(grad_output, d);
        args.gate_up_strides[d] = safe_stride(gate_up, d);
        args.shapes[d] = gate_up.size(d);
        num_rows *= gate_up.size(d);
    }

    cudaStream_t stream = static_cast<cudaStream_t>(
        at::cuda::getCurrentCUDAStream().stream());

    if (gate_up.scalar_type() == at::kFloat) {
        ASSERT_CHECK(intermediate_size % VecTraits<float>::vec_size == 0);
        dispatch_split_inplace_backward_strided(
            static_cast<float*>(grad_gate_up.data_ptr()),
            static_cast<const float*>(grad_output.data_ptr()),
            static_cast<const float*>(gate_up.data_ptr()),
            args, num_rows, intermediate_size / VecTraits<float>::vec_size, stream);
    } else if (gate_up.scalar_type() == at::kHalf) {
        ASSERT_CHECK(intermediate_size % VecTraits<__half>::vec_size == 0);
        dispatch_split_inplace_backward_strided(
            static_cast<__half*>(grad_gate_up.data_ptr()),
            static_cast<const __half*>(grad_output.data_ptr()),
            static_cast<const __half*>(gate_up.data_ptr()),
            args, num_rows, intermediate_size / VecTraits<__half>::vec_size, stream);
    } else if (gate_up.scalar_type() == at::kBFloat16) {
        ASSERT_CHECK(intermediate_size % VecTraits<__nv_bfloat16>::vec_size == 0);
        dispatch_split_inplace_backward_strided(
            static_cast<__nv_bfloat16*>(grad_gate_up.data_ptr()),
            static_cast<const __nv_bfloat16*>(grad_output.data_ptr()),
            static_cast<const __nv_bfloat16*>(gate_up.data_ptr()),
            args, num_rows, intermediate_size / VecTraits<__nv_bfloat16>::vec_size, stream);
    } else {
        throw std::runtime_error("silu_and_mul_split_inplace_backward: unsupported dtype");
    }

    sync_check_cuda_error();
}

void SiluAndMulBackward(
    const at::Tensor& grad_input1,
    const at::Tensor& grad_input2,
    const at::Tensor& grad_output,
    const at::Tensor& input1,
    const at::Tensor& input2)
{
    ASSERT_CHECK(grad_input1.is_contiguous());
    ASSERT_CHECK(grad_input2.is_contiguous());
    ASSERT_CHECK(grad_output.is_contiguous());
    ASSERT_CHECK(input1.is_contiguous());
    ASSERT_CHECK(input2.is_contiguous());

    ASSERT_CHECK(input1.dim() == input2.dim());
    ASSERT_CHECK(input1.dim() == grad_output.dim());
    ASSERT_CHECK(input1.dim() == grad_input1.dim());
    ASSERT_CHECK(input1.dim() == grad_input2.dim());
    for (int i = 0; i < input1.dim(); ++i) {
        ASSERT_CHECK(input1.size(i) == input2.size(i));
        ASSERT_CHECK(input1.size(i) == grad_output.size(i));
        ASSERT_CHECK(input1.size(i) == grad_input1.size(i));
        ASSERT_CHECK(input1.size(i) == grad_input2.size(i));
    }

    ASSERT_CHECK(grad_input1.scalar_type() == input1.scalar_type());
    ASSERT_CHECK(grad_input1.element_size() == input1.element_size());
    ASSERT_CHECK(grad_input2.scalar_type() == input1.scalar_type());
    ASSERT_CHECK(grad_input2.element_size() == input1.element_size());
    ASSERT_CHECK(grad_output.scalar_type() == input1.scalar_type());
    ASSERT_CHECK(grad_output.element_size() == input1.element_size());
    ASSERT_CHECK(input2.scalar_type() == input1.scalar_type());
    ASSERT_CHECK(input2.element_size() == input1.element_size());

    const int64_t n = get_total_elements(input1);
    cudaStream_t stream = static_cast<cudaStream_t>(
        at::cuda::getCurrentCUDAStream().stream());

    dispatch_contiguous_backward(
        grad_input1.data_ptr(),
        grad_input2.data_ptr(),
        grad_output.data_ptr(),
        input1.data_ptr(),
        input2.data_ptr(),
        n,
        input1,
        stream);
    sync_check_cuda_error();
}

// Removed TVM FFI export (see pybind11 registration)
// Removed TVM FFI export (see pybind11 registration)
// Removed TVM FFI export (see pybind11 registration)
// Removed TVM FFI export (see pybind11 registration)
// Removed TVM FFI export (see pybind11 registration)
// Removed TVM FFI export (see pybind11 registration)

}  // namespace wallx_cuda_silu_and_mul
