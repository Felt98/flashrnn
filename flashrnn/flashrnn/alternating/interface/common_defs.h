
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
#define FLASHRNN_DTYPE_R float
#define FLASHRNN_DTYPE_B float
#define FLASHRNN_DTYPE_W float
#define FLASHRNN_DTYPE_G float
#define FLASHRNN_DTYPE_S float
#define FLASHRNN_DTYPE_A float
#define FLASHRNN_SIMPLE_AGG true
#endif

#define WARP_SIZE 32
#define CEIL_DIV(a, b) (((a) + (b) - 1) / (b))
#define MAX(a, b) (((a) > (b)) ? (a) : (b))
#define MIN(a, b) (((a) < (b)) ? (a) : (b))