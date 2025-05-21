
#pragma once

#include "../../util/util.h"
#include "../../util/blas.h"
#include "../../util/cuda_error.h"

#ifndef FLASHRNN_NUM_GATES_R
// all needed definitions from external
#define FLASHRNN_NUM_HEADS 1
#define FLASHRNN_HIDDEN_SIZE 512
#define FLASHRNN_BATCH_SIZE 8
#define FLASHRNN_NUM_GATES_R 4
#define FLASHRNN_NUM_GATES_W 4
#define FLASHRNN_NUM_GATES_I 4
#define FLASHRNN_NUM_GATES_T 4
#define FLASHRNN_NUM_STATES 4
#define FLASHRNN_DTYPE __nv_bfloat16
#define FLASHRNN_USE_DTYPE_BFLOAT16
#define FLASHRNN_DTYPE_R __nv_bfloat16
#define FLASHRNN_DTYPE_B __nv_bfloat16
#define FLASHRNN_DTYPE_W __nv_bfloat16
#define FLASHRNN_DTYPE_G __nv_bfloat16
#define FLASHRNN_DTYPE_S __nv_bfloat16
#define FLASHRNN_DTYPE_A __nv_bfloat16

// fused forward
// optimized for hidden size 1024

#define FLASHRNN_FORWARD_RECURRENT_TILING_COUNT_HIDDEN 1 // Rtch 16?
#define FLASHRNN_FORWARD_RECURRENT_TILING_COUNT_GATE 64  // Rtcg 1024 best 64
#define FLASHRNN_FORWARD_BLOCK_TILING_COUNT_BATCH 1      // Btcb
// means extra warps for threads
#define FLASHRNN_FORWARD_WARP_TILING_COUNT_BATCH 1 // Wtcb
// means each warp loops over batches stored in additional shared memory
#define FLASHRNN_FORWARD_WARP_LOOPING_COUNT_BATCH 1      // Wlcp
#define FLASHRNN_FORWARD_WARP_LOOPING_COUNT_GATE 1       // Wlcg
#define FLASHRNN_FORWARD_WARP_TILING_COUNT_HIDDEN 4      // Wtch 1024 best 8
#define FLASHRNN_FORWARD_WARP_RECURRENT_CACHED_HIDDEN 16 // Wrch 1024 best 8

#define FLASHRNN_FORWARD_MULTIHEAD_TILING_COUNT 1
#define FLASHRNN_FORWARD_SHARED_MEMORY_PADDING 8

#define FLASHRNN_FORWARD_WARP_TILING_DIM_BATCH 8   // Wtdb
#define FLASHRNN_FORWARD_WARP_TILING_DIM_HIDDEN 16 // Wtdg
#define FLASHRNN_FORWARD_WARP_TILING_DIM_GATE 32   // Wtdg

// fused backward
#define FLASHRNN_BACKWARD_RECURRENT_TILING_COUNT_HIDDEN 32 // Rtch 16?
#define FLASHRNN_BACKWARD_RECURRENT_TILING_COUNT_GATE 1    // Rtcg
#define FLASHRNN_BACKWARD_BLOCK_TILING_COUNT_BATCH 1       // Btcb
#define FLASHRNN_BACKWARD_WARP_TILING_COUNT_BATCH 1        // Wtcb
#define FLASHRNN_BACKWARD_WARP_LOOPING_COUNT_BATCH 1       // Wtlb
#define FLASHRNN_BACKWARD_WARP_LOOPING_COUNT_HIDDEN 1      // Wlch
#define FLASHRNN_BACKWARD_WARP_TILING_COUNT_GATE 8         // Wtcg
#define FLASHRNN_BACKWARD_WARP_RECURRENT_CACHED_GATE 32 // Wrcg optimal for 1024

#define FLASHRNN_BACKWARD_MULTIHEAD_TILING_COUNT 1
#define FLASHRNN_BACKWARD_SHARED_MEMORY_PADDING 8

#define FLASHRNN_BACKWARD_WARP_TILING_DIM_BATCH 8   // Wtdb
#define FLASHRNN_BACKWARD_WARP_TILING_DIM_GATE 16   // Wtdh
#define FLASHRNN_BACKWARD_WARP_TILING_DIM_HIDDEN 32 // Wtdh

// defines whether g = Wx + Ry + b for every gate, enables half the cache for
// backward
#define FLASHRNN_SIMPLE_AGG true
#endif


#define WARP_SIZE 32
#define CEIL_DIV(a, b) (((a) + (b) - 1) / (b))
#define MAX(a, b) (((a) > (b)) ? (a) : (b))
#define MIN(a, b) (((a) < (b)) ? (a) : (b))