// Copyright 2024 NXAI GmbH, All Rights Reserved
// Author: Korbinian Poeppel
// Adapted from the haste library
//
// See:
// Copyright 2020 LMNT, Inc. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//    http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// ==============================================================================

// fused kernel using 16x16x16 MM

#include <cublas_v2.h>
#include <cuda.h>
#include <cuda_device_runtime_api.h>
#include <cuda_runtime_api.h>

#include "../util/blas.h"
#include "../util/cuda_error.h"
#include "../util/inline_ops.cuh"
#include "flashrnn.h"
#include <cooperative_groups.h>
#include <driver_types.h>
#include <mma.h>
#include <stdio.h>

#ifndef _FLASHRNN_POINTWISE_INCLUDED
#include "flashrnn_fused_pointwise_base.cuh"
#endif

#define CEIL_DIV(a, b) (((a) + (b) - 1) / (b))
#define MAX(a, b) (((a) > (b)) ? (a) : (b))
#define MIN(a, b) (((a) < (b)) ? (a) : (b))

// #define DEBUG

namespace
{

namespace cg = cooperative_groups;
using namespace nvcuda;

// gate order: i f z o
// FLASHRNN_NUM_GATES_R:     1
//             1 - - -
// FLASHRNN_NUM_GATES_I:     3
//             - 1 1 1

// dimensions
// G: # gates
// FLASHRNN_NUM_GATES_R: # recurrent gates per hidden dimensions (1 for lstmhin,
// 4 for slstm) FLASHRNN_NUM_GATES_I: # gates from input FLASHRNN_NUM_GATES_T: #
// total gates S: # states T: # time steps B: # batch dim H: # hidden dim I: #
// input dim

// General naming convention: dim = real size in memory, count = number along
// axis -> high level dim = count * dim
// -> tile dim = total dim / tile count

#ifndef FLASHRNN_FORWARD_RECURRENT_TILING_COUNT_GATE

// optimized for hidden size 1024
#define FLASHRNN_FORWARD_RECURRENT_TILING_COUNT_HIDDEN 1 // FRTCH 16?
#define FLASHRNN_FORWARD_RECURRENT_TILING_COUNT_GATE 64  // FRTCG 1024 best 64
#define FLASHRNN_FORWARD_BLOCK_TILING_COUNT_BATCH 1      // Btcb
// means extra warps for threads
#define FLASHRNN_FORWARD_WARP_TILING_COUNT_BATCH 1 // Wtcb
// means each warp loops over batches stored in additional shared memory
#define FLASHRNN_FORWARD_WARP_LOOPING_COUNT_BATCH 1      // Wlcp
#define FLASHRNN_FORWARD_WARP_LOOPING_COUNT_GATE 1       // FWLCG
#define FLASHRNN_FORWARD_WARP_TILING_COUNT_HIDDEN 4      // FWTCH 1024 best 8
#define FLASHRNN_FORWARD_WARP_RECURRENT_CACHED_HIDDEN 16 // FWRCH 1024 best 8

#define FLASHRNN_FORWARD_MULTIHEAD_TILING_COUNT 1
#define FLASHRNN_FORWARD_SHARED_MEMORY_PADDING 8

#define FLASHRNN_HIDDEN_SIZE 1024
#define FLASHRNN_NUM_HEADS 1

#define FLASHRNN_FORWARD_WARP_TILING_DIM_BATCH 8 // FWTDB
#define FLASHRNN_FORWARD_WARP_TILING_DIM_GATE 32 // FWTDG

#endif

#define FRTCH                                                                                                          \
    FLASHRNN_FORWARD_RECURRENT_TILING_COUNT_HIDDEN // Recurrent 权重矩阵 R 在隐藏维度方向head_dim上的 tile
                                                   // 分块数量。即每个 head 的隐藏向量会被分成 FRTCH 份

#define FRTCG FLASHRNN_FORWARD_RECURRENT_TILING_COUNT_GATE
#define FBTCB FLASHRNN_FORWARD_BLOCK_TILING_COUNT_BATCH
#define FWTCB FLASHRNN_FORWARD_WARP_TILING_COUNT_BATCH
#define FWLCB FLASHRNN_FORWARD_WARP_LOOPING_COUNT_BATCH
#define FWLCG FLASHRNN_FORWARD_WARP_LOOPING_COUNT_GATE
#define FWTCH FLASHRNN_FORWARD_WARP_TILING_COUNT_HIDDEN
#define FWRCH FLASHRNN_FORWARD_WARP_RECURRENT_CACHED_HIDDEN
#define FMTC FLASHRNN_FORWARD_MULTIHEAD_TILING_COUNT
#define FSMP FLASHRNN_FORWARD_SHARED_MEMORY_PADDING
#define FWTDB FLASHRNN_FORWARD_WARP_TILING_DIM_BATCH
#define FWTDG FLASHRNN_FORWARD_WARP_TILING_DIM_GATE
#define FWTDH FLASHRNN_FORWARD_WARP_TILING_DIM_HIDDEN

#ifdef FLASHRNN_USE_DTYPE_FLOAT32
#define MAT_DTYPE wmma::precision::tf32
#define DTYPE float
#define ACC_DTYPE float
#endif
#ifdef FLASHRNN_USE_DTYPE_FLOAT16
#define MAT_DTYPE __half
#define DTYPE __half
#define ACC_DTYPE __half
#endif
#ifdef FLASHRNN_USE_DTYPE_BFLOAT16
#define MAT_DTYPE __nv_bfloat16
#define DTYPE __nv_bfloat16
#define ACC_DTYPE float
#endif

#define HS FLASHRNN_HIDDEN_SIZE
#define NH FLASHRNN_NUM_HEADS

#define WARP_SIZE 32

// #endif
#define _FLOAT4FACTOR 8

#define _FUSED_KERNEL_MAX_THREADS                                                                                      \
    (WARP_SIZE * FLASHRNN_FORWARD_WARP_TILING_COUNT_HIDDEN * FLASHRNN_NUM_GATES_R * FLASHRNN_HIDDEN_SIZE /             \
     FLASHRNN_NUM_HEADS / FLASHRNN_FORWARD_RECURRENT_TILING_COUNT_GATE / FLASHRNN_FORWARD_WARP_LOOPING_COUNT_GATE *    \
     FLASHRNN_FORWARD_WARP_TILING_COUNT_BATCH / FLASHRNN_FORWARD_WARP_TILING_DIM_GATE)

#define _FUSED_KERNEL_MIN_BLOCKS                                                                                       \
    (FLASHRNN_FORWARD_RECURRENT_TILING_COUNT_GATE * FLASHRNN_FORWARD_BLOCK_TILING_COUNT_BATCH *                        \
     FLASHRNN_FORWARD_RECURRENT_TILING_COUNT_HIDDEN * FLASHRNN_FORWARD_MULTIHEAD_TILING_COUNT)

// __launch_bounds__(每个线程块（block）中最大的线程数,每个SM上至少要运行的线程块数量)，用于控制寄存器使用和优化线程块调度
template <bool Training>
__global__ void __launch_bounds__(_FUSED_KERNEL_MAX_THREADS, _FUSED_KERNEL_MIN_BLOCKS)
    FLASHRNNCellFusedForward(const uint steps, const uint batch_dim,
                             const FLASHRNN_DTYPE_W *Wx, // Precomputed (Wx) vector [T, B, igate * H]
                             const FLASHRNN_DTYPE_R *R,  // recurrect matrix head_dim x head_dim [H,
                                                         // FLASHRNN_NUM_GATES_R * H]
                             const FLASHRNN_DTYPE_B *b,  // Bias for gates [G, FLASHRNN_NUM_GATES_T * H]
                             FLASHRNN_DTYPE_S *states,   // states [S, T + 1, B, H]
                             FLASHRNN_DTYPE_G *g_r_out,  // Output activations (Wx + Ry + b) [], also
                                                         // contains gate values [T, G-1, B, H] other gates
                             FLASHRNN_DTYPE_G *g_i_out,  // [FLASHRNN_NUM_GATES_T, T, B, H]?  input gate
                             ACC_DTYPE *gate_buffer)
{
    // 每一个头的隐藏维度
    const uint head_dim = FLASHRNN_HIDDEN_SIZE / FLASHRNN_NUM_HEADS;
    // assuming at least 8 as a batch size, at least 32 as a hidden dim
    // this is necessary for mm
    // each thread takes a tile of (8 x 32) of the pre-activations of one gate,
    // i.e. a (8 x 32) tile of the outputs
    const uint hidden_grid_dim = FRTCH; // equals to FRTCH=1
#ifdef DEBUG
    // FRTCH是指一个 head 的 hidden特征 head_dim被分为几块
    // FMHTC是指一个 head 的 hidden特征 的某一划分部分 是由几个 block 来计算
    // 在z方向（hidden特征方向）上 block的数量 gridDim.z=FMHTC*FRTCH
    if ((threadIdx.x == 0) && (hidden_grid_dim != gridDim.z / FMHTC))
    {
        printf("ERROR for hidden_grid_dim: %d, %d, %d\n", gridDim.z, FMHTC, FRTCH);
    }
#endif

    // blockIdx.z 被拆分为两个维度：hidden_block_idx = 0 和 multihead_idx = 0
    const uint hidden_block_idx =
        blockIdx.z % hidden_grid_dim; // block在一个head中的相对坐标（head被划分为hidden_grid_dim块）
    const uint multihead_idx = (blockIdx.z / hidden_grid_dim) * head_dim; // block所在head的入口

    /// tile of R within head_dim / Rtdh, FLASHRNN_NUM_GATES_R * head_dim / Rtdg
    // Rtdg是什么
    //  R_shared 和 mmul_buffer 共用sbuf：
    extern __shared__ float4 sbuf[];
    FLASHRNN_DTYPE_R *R_shared = (FLASHRNN_DTYPE_R *)sbuf; // 储存 tile 内的 recurrent 权重R

    FLASHRNN_DTYPE_S
    states_local[CEIL_DIV(FLASHRNN_FORWARD_WARP_LOOPING_COUNT_BATCH * FLASHRNN_FORWARD_WARP_TILING_DIM_BATCH *
                              FLASHRNN_FORWARD_WARP_TILING_DIM_GATE * FLASHRNN_FORWARD_WARP_LOOPING_COUNT_GATE,
                          FLASHRNN_NUM_GATES_R * FRTCH * FWTCH * WARP_SIZE)][FLASHRNN_NUM_STATES];
    FLASHRNN_DTYPE_B biases_local[FLASHRNN_FORWARD_WARP_LOOPING_COUNT_BATCH][FLASHRNN_NUM_GATES_T];

    // matrix multiplication buffer of size (batch_dim x head_dim / Rtdg)
    int head_dim_per_block_shared =
        ((int)(head_dim / FRTCH) - (int)(FWTDH * FWTCH * FWRCH)); //  计算共享内存中可用于 mmul_buffer 的空间

    if (head_dim_per_block_shared < 0)
    {
        head_dim_per_block_shared = 0;
    }

    // 矩阵乘R*S_t-1 结果的预留空间
    ACC_DTYPE *mmul_buffer =
        (ACC_DTYPE *)(((FLASHRNN_DTYPE_R *)(sbuf)) +
                      (head_dim_per_block_shared + FSMP) * (FLASHRNN_NUM_GATES_R * head_dim / FRTCG));
    // 没有预留空间，整个 shared memory sbuf 直接被 mmul_buffer 使用，不为 R_shared 分区
    if (head_dim_per_block_shared == 0)
    {
        mmul_buffer = (ACC_DTYPE *)sbuf;
    }

    // FWLCB * FWTDB = 每个线程要处理的 batch 数量
    // gridDim.y * blockDim.y = y方向上总线程数
    const uint BatchIterations =
        batch_dim / FWTDB / FWLCB / gridDim.y /
        blockDim.y; // 每个线程块要遍历多少个 batch tile，因为batch_dim可能非常大，需要block循环处理batch

    // 根据目前的设置
    // batch_idx = 0
    // block_batch_idx = 8
    const uint batch_idx = FWLCB * FWTDB * (blockIdx.y * blockDim.y + threadIdx.y); // 当前线程负责的 全局 batch
                                                                                    // 起始索引
    const uint block_batch_idx = FWLCB * FWTDB * threadIdx.y; // 当前线程在一个 block 内部的 batch 起始偏移

} // anonymous namespace

namespace flashrnn_fused
{
int ForwardPass::RunGRU(const int steps,
                        const FLASHRNN_DTYPE_R *R, // Weight matrix for recurrent state (Ry) [y,H*4]
                        const FLASHRNN_DTYPE_B *b, // Bias for gates (Wx + Ry + b) [H*4]
                        const FLASHRNN_DTYPE_W *x, // Input vector [T,N,C]
                        FLASHRNN_DTYPE_S *s,       // Cell states [S+1,N,H]
                        FLASHRNN_DTYPE_G *g_r,     // Output vector (Wx + Ry + b) [S,N,H*3])
                        FLASHRNN_DTYPE_G *g_i, FLASHRNN_ACC_DTYPE *gate_buffer)
{ // Output vector (Wx + Ry + b) [S,N,H]
    const blas<void>::set_pointer_mode scoped1(data_->main_blas_handle);

    const uint batch_size = data_->batch_size;
    const uint head_dim = data_->hidden_size / FLASHRNN_NUM_HEADS;

    const cublasHandle_t blas_handle = data_->main_blas_handle;

    const cudaStream_t *stream_K = data_->stream_K;
    const cudaEvent_t *event_K = data_->event_K;
    cudaStream_t blas_save_stream;

    // R 在 门（gate）维度上的并行 tile 数量
    uint recurrent_tiling_count_gate =
        MIN(FLASHRNN_NUM_GATES_R * head_dim / FLASHRNN_FORWARD_WARP_TILING_DIM_GATE /
                FLASHRNN_FORWARD_WARP_LOOPING_COUNT_GATE,
            MAX(WARP_SIZE * FLASHRNN_NUM_GATES_R * head_dim / 1024 / FLASHRNN_FORWARD_WARP_TILING_DIM_GATE /
                    FLASHRNN_FORWARD_WARP_LOOPING_COUNT_GATE,
                FLASHRNN_FORWARD_RECURRENT_TILING_COUNT_GATE)); // because the
                                                                // maximal block
                                                                // size is 1024

    if (recurrent_tiling_count_gate != FLASHRNN_FORWARD_RECURRENT_TILING_COUNT_GATE)
    {
        fprintf(stderr, "The specified forward RECURRENT_TILING_COUNT_GATE should be: %d\n",
                recurrent_tiling_count_gate);
        fprintf(stderr, "Values: RTCG: %d, RTCH: %d, WTCG: %d, BCB: %d, WTCH: %d\n",
                FLASHRNN_FORWARD_RECURRENT_TILING_COUNT_GATE, FLASHRNN_FORWARD_RECURRENT_TILING_COUNT_HIDDEN,
                FLASHRNN_FORWARD_WARP_LOOPING_COUNT_GATE, FLASHRNN_FORWARD_BLOCK_TILING_COUNT_BATCH,
                FLASHRNN_FORWARD_WARP_TILING_COUNT_HIDDEN);
        return 1;
    }

    const dim3 blockDim(WARP_SIZE * FLASHRNN_FORWARD_WARP_TILING_COUNT_HIDDEN * FLASHRNN_NUM_GATES_R *
                            FLASHRNN_HIDDEN_SIZE / FLASHRNN_NUM_HEADS / FLASHRNN_FORWARD_RECURRENT_TILING_COUNT_GATE /
                            FLASHRNN_FORWARD_WARP_LOOPING_COUNT_GATE / FLASHRNN_FORWARD_WARP_TILING_DIM_GATE,
                        FLASHRNN_FORWARD_WARP_TILING_COUNT_BATCH, 1);

    const dim3 gridDim(recurrent_tiling_count_gate, FLASHRNN_FORWARD_BLOCK_TILING_COUNT_BATCH,
                       FLASHRNN_FORWARD_RECURRENT_TILING_COUNT_HIDDEN * FLASHRNN_FORWARD_MULTIHEAD_TILING_COUNT);

    // Recurrent matrix tile and matmul gate buffer, 2 for float32 of mmul
    // buffer shared memory size is in bytes!, this is why the sizeof is needed

    int head_dim_per_block = (head_dim / FLASHRNN_FORWARD_RECURRENT_TILING_COUNT_HIDDEN -
                              (FLASHRNN_FORWARD_WARP_TILING_COUNT_HIDDEN * FLASHRNN_FORWARD_WARP_TILING_DIM_HIDDEN *
                               FLASHRNN_FORWARD_WARP_RECURRENT_CACHED_HIDDEN));
    const uint sharedMemorySizeR = (head_dim_per_block > 0)
                                       ? sizeof(DTYPE) * FLASHRNN_NUM_GATES_R * head_dim / recurrent_tiling_count_gate *
                                             (head_dim_per_block + FLASHRNN_FORWARD_SHARED_MEMORY_PADDING)
                                       : 0;
    const uint sharedMemorySizeMatmul =
        sizeof(ACC_DTYPE) * FLASHRNN_FORWARD_WARP_TILING_DIM_BATCH * FLASHRNN_FORWARD_WARP_LOOPING_COUNT_BATCH *
        FLASHRNN_FORWARD_WARP_TILING_COUNT_BATCH * FLASHRNN_FORWARD_WARP_TILING_COUNT_HIDDEN *
        (FLASHRNN_NUM_GATES_R * head_dim / recurrent_tiling_count_gate + FLASHRNN_FORWARD_SHARED_MEMORY_PADDING);
    const uint sharedMemorySize = sharedMemorySizeR + sharedMemorySizeMatmul;

#ifdef DEBUG
    printf("Shared Memory Size: %d (= %d (= %d * %d) + "
           "%d))\n",
           sharedMemorySize, sharedMemorySizeR, FLASHRNN_NUM_GATES_R * head_dim / recurrent_tiling_count_gate,
           head_dim_per_block, sharedMemorySizeMatmul);
#endif
    int maxActiveBlocks;

    // define kernel and increase shared memory from
    // default
    auto kernel = FLASHRNNCellFusedForward<true>;
    cudaError_t err = cudaSuccess;
    err = cudaFuncSetAttribute(kernel, cudaFuncAttributePreferredSharedMemoryCarveout, cudaSharedmemCarveoutMaxShared);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Error setting shared mem attribute carveout");
    }
    err = cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, sharedMemorySize);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Error setting shared mem attribute size");
    }

    bool use_blas_input_stream = false;
    if (cublasGetStream(blas_handle, &blas_save_stream) == CUBLAS_STATUS_SUCCESS)
    {
        use_blas_input_stream = true;
    }
    else
    {
        use_blas_input_stream = false;
    }
    cudaEventRecord(event_K[0], data_->stream);
    if (use_blas_input_stream)
    {
        cudaEventRecord(event_K[0], blas_save_stream);
    }

    // FLASHRNN_FORWARD_MULTIHEAD_TILING_COUNT： 每次循环让grid处理的头数
    for (uint i = 0; i < FLASHRNN_NUM_HEADS / FLASHRNN_FORWARD_MULTIHEAD_TILING_COUNT; i++)
    {
        uint head_idx = i * head_dim * FLASHRNN_FORWARD_MULTIHEAD_TILING_COUNT; // 当前头的在H中的起始维度坐标
        cudaStreamWaitEvent(stream_K[i], event_K[0]);

#if FLASHRNN_FORWARD_RECURRENT_TILING_COUNT_GATE * FLASHRNN_FORWARD_RECURRENT_TILING_COUNT_HIDDEN > 1
        void *x_h = (void *)(x + FLASHRNN_NUM_GATES_W * head_idx);
        void *g_r_h = (void *)(g_r + FLASHRNN_NUM_GATES_R * head_idx);
        void *g_i_h = (void *)(g_i + FLASHRNN_NUM_GATES_I * head_idx);
        void *R_h = (void *)(R + FLASHRNN_NUM_GATES_R * head_dim *
                                     head_idx); // 为什么要多乘head_dim？，每次处理4*head_dim*head_dim？
        void *b_h = (void *)(b + FLASHRNN_NUM_GATES_T * head_idx);
        void *s_h = (void *)(s + head_idx);
        void *gate_buffer_h = (void *)(gate_buffer + FLASHRNN_NUM_GATES_R * head_idx);

        // kernel的参数
        void *kernelArgs[] = {(void *)&steps, (void *)&batch_size, (void *)&x_h,   (void *)&R_h,          (void *)&b_h,
                              (void *)&s_h,   (void *)&g_r_h,      (void *)&g_i_h, (void *)&gate_buffer_h};

        // 调用前向的kernel
        err = cudaLaunchCooperativeKernel((void *)kernel, gridDim, blockDim, kernelArgs, sharedMemorySize, stream_K[i]);
#else

        kernel<<<gridDim, blockDim, sharedMemorySize, stream_K[i]>>>(
            steps, batch_size, x + FLASHRNN_NUM_GATES_W * head_idx, R + FLASHRNN_NUM_GATES_R * head_dim * head_idx,
            b + FLASHRNN_NUM_GATES_T * head_idx, s + head_idx, g_r + FLASHRNN_NUM_GATES_R * head_idx,
            g_i + FLASHRNN_NUM_GATES_I * head_idx, (ACC_DTYPE *)gate_buffer + FLASHRNN_NUM_GATES_R * head_idx);

#endif
        cudaEventRecord(event_K[i], stream_K[i]);
        cudaStreamWaitEvent(data_->stream, event_K[i]);
        if (use_blas_input_stream)
        {
            cudaStreamWaitEvent(blas_save_stream, event_K[i]);
        }
    }
    if (err == cudaSuccess)
    {
#ifdef DEBUG
        printf("NO ERROR until after execution\n");
#endif
        err = cudaPeekAtLastError();
    }
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Error after forward kernel launch: %s\n", cudaGetErrorString(err));
        fprintf(stderr,
                "Values: RTCG: %d, RTCH: %d, WTCG: %d, BCB: "
                "%d, WTCH: %d\n",
                FLASHRNN_FORWARD_RECURRENT_TILING_COUNT_GATE, FLASHRNN_FORWARD_RECURRENT_TILING_COUNT_HIDDEN,
                FLASHRNN_FORWARD_WARP_LOOPING_COUNT_GATE, FLASHRNN_FORWARD_BLOCK_TILING_COUNT_BATCH,
                FLASHRNN_FORWARD_WARP_TILING_COUNT_HIDDEN);
        cudaOccupancyMaxActiveBlocksPerMultiprocessor2(blockDim, sharedMemorySize, (void *)kernel);
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(&maxActiveBlocks, (void *)kernel, blockDim.x, sharedMemorySize);
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, 0);
        fprintf(stderr,
                "Multiprocessors: %d, Max active Blocks: "
                "%d, Shared Mem per Block "
                "%lu, per MP: %lu\n",
                prop.multiProcessorCount, maxActiveBlocks, prop.sharedMemPerBlock, prop.sharedMemPerMultiprocessor);
        fprintf(stderr, "gridDim: %d, %d, %d, blockDim: %d, %d, %d\n", gridDim.x, gridDim.y, gridDim.z, blockDim.x,
                blockDim.y, blockDim.z);
        fprintf(stderr, "R_block_tile size: %d, %d -> %d\n",
                FLASHRNN_NUM_GATES_R * head_dim / FLASHRNN_FORWARD_RECURRENT_TILING_COUNT_GATE,
                head_dim / FLASHRNN_FORWARD_RECURRENT_TILING_COUNT_HIDDEN, sharedMemorySizeR);
        fprintf(stderr, "MMUL BUF SIZE : %d\n", sharedMemorySizeMatmul);
        fprintf(stderr, "Pre-Kernel launch with shared mem: %d\n", sharedMemorySize);

        return 1;
    }
    if (use_blas_input_stream)
    {
        cublasSetStream(blas_handle, blas_save_stream);
    }
    return 0;
}

} // namespace flashrnn_fused
