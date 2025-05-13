import copy
import os
from pathlib import Path
from typing import Optional
from .config import FlashRNNConfig, permute_to, DTYPE_DICT_REV, DTYPE_DICT, _get_config
import torch

# from .autotune.constrint import ValueHeuristic, ValueRefinement
from autotune.constrint import ValueHeuristic, ValueRefinement

# from .cuda_init import load
from .cuda_init_parametric import load_parametric_and_test_and_bisect
from .gpu_info.gpu_info import get_gpu_info

# from .vanilla import (
#     flashrnn_forward,
#     flashrnn_forward_step,
#     flashrnn_pointwise_function_registry,
# )


curdir = Path(os.path.split(os.path.os.path.abspath(__file__))[0])


def round_to_multiple(n, m=8):
    return ((n + m - 1) // m) * m


def conditional_decorator(condition, decorator):
    """A higher-order decorator that applies 'decorator' only if 'condition' is True."""

    def dummy_decorator(func):
        """A dummy decorator that does nothing."""
        return func

    if condition:
        # If condition is True, return the actual decorator
        return decorator
    else:
        # If condition is False, return the dummy decorator
        return dummy_decorator


def round_to_divisible(x, y):
    """
    Round a number such that round(x) divides y
    """
    xnew = x
    while y % xnew != 0:
        znew = y // xnew
        xnew = (y + znew - 1) // znew
    return xnew


"""
JIT编译并缓存一个 FlashRNN 的 CUDA fused 版模块到 python中使用的类
被 FlashRNNFuncGeneratorFused 使用
"""


class _FlashRNNCUDAFused:
    mod: dict = {}

    # 根据FlashRNNConfig 生成一个模型实例保存在哈希表mod（不同config生成不同实例）
    @classmethod
    def instance(cls, config: FlashRNNConfig):
        device_id = torch.cuda.current_device()
        cfgdevstr = repr(config) + f"_device{device_id}"
        if cfgdevstr not in cls.mod:
            gpu_info = get_gpu_info(device_id=torch.cuda.current_device())
            VR = ValueRefinement
            LF = ValueHeuristic.LARGEST_FIRST
            SF = ValueHeuristic.SMALLEST_FIRST

            # 增加value_refinements和constraint_str配置

            # value_refinements：定义调优参数空间
            value_refinements = (
                VR(
                    "FLASHRNN_HIDDEN_DIM", LF
                ),  # 尝试设置 FLASHRNN_HIDDEN_DIM 参数的不同值，并用 Largest First（LF） 的策略优先尝试较大的值
                VR("FLASHRNN_NUM_HEADS", LF),
                VR("FLASHRNN_BACKWARD_SHARED_MEMORY_PADDING", SF),
                VR("FLASHRNN_FORWARD_SHARED_MEMORY_PADDING", SF),
                VR("FLASHRNN_BACKWARD_WARP_TILING_DIM_GATE", LF),
                VR("FLASHRNN_FORWARD_WARP_TILING_DIM_HIDDEN", LF),
                VR("FLASHRNN_BACKWARD_WARP_TILING_DIM_BATCH", LF),
                VR("FLASHRNN_FORWARD_WARP_TILING_DIM_BATCH", LF),
                VR("FLASHRNN_BACKWARD_WARP_TILING_DIM_HIDDEN", LF),
                VR("FLASHRNN_FORWARD_WARP_TILING_DIM_GATE", LF),
                VR("FLASHRNN_BACKWARD_MULTIHEAD_TILING_COUNT", LF),
                VR("FLASHRNN_FORWARD_MULTIHEAD_TILING_COUNT", LF),
                VR("FLASHRNN_BACKWARD_RECURRENT_TILING_COUNT_HIDDEN", LF),
                VR("FLASHRNN_FORWARD_RECURRENT_TILING_COUNT_GATE", LF),
                VR("FLASHRNN_BACKWARD_RECURRENT_TILING_COUNT_GATE", SF),
                VR("FLASHRNN_FORWARD_RECURRENT_TILING_COUNT_HIDDEN", SF),
                VR("FLASHRNN_BACKWARD_BLOCK_TILING_COUNT_BATCH", LF),
                VR("FLASHRNN_FORWARD_BLOCK_TILING_COUNT_BATCH", LF),
                VR("FLASHRNN_BACKWARD_WARP_LOOPING_COUNT_BATCH", LF),
                VR("FLASHRNN_FORWARD_WARP_LOOPING_COUNT_BATCH", LF),
                VR("FLASHRNN_BACKWARD_WARP_LOOPING_COUNT_HIDDEN", SF),
                VR("FLASHRNN_FORWARD_WARP_LOOPING_COUNT_GATE", SF),
                VR("FLASHRNN_BACKWARD_WARP_TILING_COUNT_GATE", LF),
                VR("FLASHRNN_FORWARD_WARP_TILING_COUNT_HIDDEN", LF),
                VR("FLASHRNN_BACKWARD_WARP_TILING_COUNT_BATCH", LF),
                VR("FLASHRNN_FORWARD_WARP_TILING_COUNT_BATCH", LF),
                VR("FLASHRNN_BACKWARD_WARP_RECURRENT_CACHED_GATE", LF),
                VR("FLASHRNN_FORWARD_WARP_RECURRENT_CACHED_HIDDEN", LF),
            )

            # constraint_str ：搜索空间的限制条件 和 硬件约束模型
            constraint_str = (
                """
                WARP_SIZE == 32;
                """
                + (
                    """
                FLASHRNN_FORWARD_WARP_TILING_DIM_BATCH == [ 8, 16, 32 ];    # 在调优时只能取值 8、16、32
                FLASHRNN_BACKWARD_WARP_TILING_DIM_BATCH == [ 8, 16, 32 ];
                FLASHRNN_FORWARD_WARP_TILING_DIM_GATE == [ 8, 16, 32 ];
                FLASHRNN_BACKWARD_WARP_TILING_DIM_HIDDEN == [ 8, 16, 32 ];
                FLASHRNN_FORWARD_WARP_TILING_DIM_HIDDEN == 16;
                FLASHRNN_BACKWARD_WARP_TILING_DIM_GATE == 16;
                """
                    if config.dtype == "float16" or config.dtype == "bfloat16"
                    else """
                FLASHRNN_FORWARD_WARP_TILING_DIM_BATCH == [ 16 ];
                FLASHRNN_BACKWARD_WARP_TILING_DIM_BATCH == [ 16 ];
                FLASHRNN_FORWARD_WARP_TILING_DIM_GATE == [ 16 ];
                FLASHRNN_BACKWARD_WARP_TILING_DIM_HIDDEN == [ 16 ];
                FLASHRNN_FORWARD_WARP_TILING_DIM_HIDDEN == 8;
                FLASHRNN_BACKWARD_WARP_TILING_DIM_GATE == 8;
                """
                )
                + f"""
                # Optimize not fully occupied
                MAX_THREADS_PER_BLOCK_FORWARD == {gpu_info['maxThreadsPerBlock'] // 4};
                MAX_THREADS_PER_BLOCK_BACKWARD == {gpu_info['maxThreadsPerBlock'] // 4};
                # Factor 2 as these are float32 registers
                REGISTERS_PER_BLOCK_FORWARD == {gpu_info['regsPerMultiprocessor'] * (2 if config.dtype != 'float32' else 1) } ;
                REGISTERS_PER_BLOCK_BACKWARD == {gpu_info['regsPerMultiprocessor'] * (2 if config.dtype != 'float32' else 1)} ;
                STREAMING_MULTIPROCESSORS == {gpu_info['multiProcessorCount']};
                SHARED_MEMORY_PER_BLOCK == {min(gpu_info['sharedMemPerBlockOptin'] - 1024, 227000)};
                FLASHRNN_NUM_GATES_R == {config.num_gates_r};
                FLASHRNN_NUM_GATES_W == {config.num_gates_w};
                SHARED_MEMORY_PADDING == 8;
                FLASHRNN_FORWARD_WARP_TILING_DIM_BATCH * FLASHRNN_FORWARD_WARP_TILING_DIM_GATE == 16 * 16 ;
                FLASHRNN_BACKWARD_WARP_TILING_DIM_BATCH * FLASHRNN_BACKWARD_WARP_TILING_DIM_HIDDEN == 16 * 16 ;

                # Sizes given by user
                HEAD_DIM == {config.hidden_dim // config.num_heads};
                NUM_HEADS == {config.num_heads};
                BATCH_DIM == {round_to_multiple(config.batch_size, 8 if config.dtype != 'float32' else 16)};
                FLASHRNN_HIDDEN_DIM == {config.hidden_dim};
                FLASHRNN_NUM_HEADS == {config.num_heads};

                # manual

                # this may be replaced by an inequality, to get a solution, but is slower
                # INTERNAL_HEAD_DIM is used internally
                HEAD_DIM == INTERNAL_HEAD_DIM ;
                FLASHRNN_FORWARD_SHARED_MEMORY_PADDING == SHARED_MEMORY_PADDING ;
                FLASHRNN_BACKWARD_SHARED_MEMORY_PADDING == SHARED_MEMORY_PADDING ;
                HEAD_DIM_SQ == HEAD_DIM ^ 2 ;
                # need this to match registers
                FLASHRNN_FORWARD_WARP_LOOPING_COUNT_BATCH <= 4 ;
                FLASHRNN_BACKWARD_WARP_LOOPING_COUNT_BATCH <= 4 ;
                FORWARD_FULL_COUNT_BATCH ==
                FLASHRNN_FORWARD_BLOCK_TILING_COUNT_BATCH * FLASHRNN_FORWARD_WARP_TILING_COUNT_BATCH * FLASHRNN_FORWARD_WARP_LOOPING_COUNT_BATCH;
                BACKWARD_FULL_COUNT_BATCH ==
                FLASHRNN_BACKWARD_BLOCK_TILING_COUNT_BATCH * FLASHRNN_BACKWARD_WARP_TILING_COUNT_BATCH * FLASHRNN_BACKWARD_WARP_LOOPING_COUNT_BATCH;

                # FORWARD PART
                FLASHRNN_FORWARD_WARP_TILING_COUNT_HIDDEN * FLASHRNN_FORWARD_WARP_TILING_COUNT_BATCH * HEAD_DIM * FLASHRNN_NUM_GATES_R
                == FORWARD_NUM_WARPS * FLASHRNN_FORWARD_RECURRENT_TILING_COUNT_GATE
                * FLASHRNN_FORWARD_WARP_LOOPING_COUNT_GATE * FLASHRNN_FORWARD_WARP_TILING_DIM_GATE ;
                HEAD_DIM % FLASHRNN_FORWARD_RECURRENT_TILING_COUNT_HIDDEN * FLASHRNN_FORWARD_WARP_TILING_COUNT_HIDDEN
                * FLASHRNN_FORWARD_WARP_TILING_DIM_HIDDEN == 0 ;
                FLASHRNN_NUM_GATES_R * HEAD_DIM % (FLASHRNN_FORWARD_WARP_LOOPING_COUNT_GATE
                * FLASHRNN_FORWARD_RECURRENT_TILING_COUNT_GATE * FLASHRNN_FORWARD_WARP_TILING_DIM_GATE) == 0 ;
                FORWARD_NUM_WARPS * WARP_SIZE <= MAX_THREADS_PER_BLOCK_FORWARD;

                BATCH_DIM % FORWARD_FULL_COUNT_BATCH * FLASHRNN_FORWARD_WARP_TILING_DIM_BATCH == 0;

                FLASHRNN_FORWARD_NUM_BLOCKS <= STREAMING_MULTIPROCESSORS ;

                FLASHRNN_FORWARD_NUM_BLOCKS == FLASHRNN_FORWARD_RECURRENT_TILING_COUNT_GATE
                * FLASHRNN_FORWARD_RECURRENT_TILING_COUNT_HIDDEN * FLASHRNN_FORWARD_BLOCK_TILING_COUNT_BATCH
                * FLASHRNN_FORWARD_MULTIHEAD_TILING_COUNT ;

                NUM_HEADS % FLASHRNN_FORWARD_MULTIHEAD_TILING_COUNT == 0 ;

                # recurrent register memory - measured in counts of fp (2 bytes or 4 bytes)
                FLASHRNN_FORWARD_WARP_TILING_DIM_HIDDEN * FLASHRNN_FORWARD_WARP_TILING_DIM_GATE
                * FLASHRNN_FORWARD_WARP_RECURRENT_CACHED_HIDDEN * FORWARD_NUM_WARPS
                == FORWARD_RECURRENT_REGISTER_MEMORY * FLASHRNN_FORWARD_WARP_TILING_COUNT_BATCH;
                FLASHRNN_FORWARD_RECURRENT_TILING_COUNT_HIDDEN * FLASHRNN_FORWARD_WARP_RECURRENT_CACHED_HIDDEN
                * FLASHRNN_FORWARD_WARP_TILING_COUNT_HIDDEN * FLASHRNN_FORWARD_WARP_TILING_DIM_HIDDEN
                <= INTERNAL_HEAD_DIM;
                FORWARD_RECURRENT_REGISTER_MEMORY
                + FORWARD_NUM_WARPS * FLASHRNN_FORWARD_WARP_TILING_DIM_BATCH * FLASHRNN_FORWARD_WARP_TILING_DIM_GATE
                * FLASHRNN_FORWARD_WARP_LOOPING_COUNT_BATCH * {config._internal_acc_dtype_size // config._internal_dtype_size}
                + FORWARD_NUM_WARPS * FLASHRNN_FORWARD_WARP_TILING_DIM_BATCH * FLASHRNN_FORWARD_WARP_TILING_DIM_HIDDEN
                <= REGISTERS_PER_BLOCK_FORWARD ;

                # Shared memory size constraint in bytes, add 64 to keep FORWARD_RECURRENT_SHARED_MEMORY non-zero
                {config._internal_dtype_size} * FLASHRNN_NUM_GATES_R * FLASHRNN_FORWARD_BLOCK_TILING_COUNT_BATCH * HEAD_DIM_SQ
                + {config._internal_dtype_size} * HEAD_DIM * FLASHRNN_NUM_GATES_R * SHARED_MEMORY_PADDING * FLASHRNN_FORWARD_BLOCK_TILING_COUNT_BATCH + 64
                == FORWARD_RECURRENT_SHARED_MEMORY
                + {config._internal_dtype_size} * FORWARD_RECURRENT_REGISTER_MEMORY * FLASHRNN_FORWARD_BLOCK_TILING_COUNT_BATCH
                * FLASHRNN_FORWARD_RECURRENT_TILING_COUNT_GATE * FLASHRNN_FORWARD_RECURRENT_TILING_COUNT_HIDDEN;

                # Memory for matrix multiplication results and aggregation
                {config._internal_acc_dtype_size} * FLASHRNN_FORWARD_WARP_TILING_COUNT_HIDDEN * FLASHRNN_FORWARD_RECURRENT_TILING_COUNT_HIDDEN
                * FLASHRNN_NUM_GATES_R * FORWARD_FULL_COUNT_BATCH * FLASHRNN_FORWARD_WARP_TILING_DIM_BATCH * INTERNAL_HEAD_DIM
                + {config._internal_acc_dtype_size} * FLASHRNN_FORWARD_WARP_TILING_COUNT_HIDDEN * FLASHRNN_FORWARD_RECURRENT_TILING_COUNT_GATE * FLASHRNN_FORWARD_RECURRENT_TILING_COUNT_HIDDEN
                * FORWARD_FULL_COUNT_BATCH * FLASHRNN_FORWARD_WARP_TILING_DIM_BATCH * SHARED_MEMORY_PADDING == FORWARD_STATE_SHARED_MEMORY ;

                # total shared memory
                FORWARD_RECURRENT_SHARED_MEMORY + FORWARD_STATE_SHARED_MEMORY == FLASHRNN_FORWARD_SHARED_MEMORY_PER_HEAD_USED;
                FLASHRNN_FORWARD_MULTIHEAD_TILING_COUNT * FLASHRNN_FORWARD_SHARED_MEMORY_PER_HEAD_USED
                <= SHARED_MEMORY_PER_BLOCK * FLASHRNN_FORWARD_NUM_BLOCKS;

                # BACKWARD PART
                FLASHRNN_BACKWARD_WARP_TILING_COUNT_GATE * FLASHRNN_BACKWARD_WARP_TILING_COUNT_BATCH * INTERNAL_HEAD_DIM
                == BACKWARD_NUM_WARPS * FLASHRNN_BACKWARD_RECURRENT_TILING_COUNT_HIDDEN
                * FLASHRNN_BACKWARD_WARP_LOOPING_COUNT_HIDDEN * FLASHRNN_BACKWARD_WARP_TILING_DIM_HIDDEN ;
                HEAD_DIM % ( FLASHRNN_BACKWARD_RECURRENT_TILING_COUNT_HIDDEN * FLASHRNN_BACKWARD_WARP_TILING_DIM_HIDDEN
                * FLASHRNN_BACKWARD_WARP_LOOPING_COUNT_HIDDEN ) == 0 ;
                FLASHRNN_NUM_GATES_R * HEAD_DIM % (FLASHRNN_BACKWARD_RECURRENT_TILING_COUNT_GATE
                * FLASHRNN_BACKWARD_WARP_TILING_COUNT_GATE * FLASHRNN_BACKWARD_WARP_TILING_DIM_GATE) == 0;
                BACKWARD_NUM_WARPS * WARP_SIZE <= MAX_THREADS_PER_BLOCK_BACKWARD;

                BATCH_DIM % BACKWARD_FULL_COUNT_BATCH * FLASHRNN_BACKWARD_WARP_TILING_DIM_BATCH == 0;

                FLASHRNN_BACKWARD_NUM_BLOCKS <= STREAMING_MULTIPROCESSORS ;

                FLASHRNN_BACKWARD_NUM_BLOCKS == FLASHRNN_BACKWARD_RECURRENT_TILING_COUNT_GATE
                * FLASHRNN_BACKWARD_RECURRENT_TILING_COUNT_HIDDEN * FLASHRNN_BACKWARD_BLOCK_TILING_COUNT_BATCH
                * FLASHRNN_BACKWARD_MULTIHEAD_TILING_COUNT;

                NUM_HEADS % FLASHRNN_BACKWARD_MULTIHEAD_TILING_COUNT == 0 ;

                # recurrent register memory in dtypes (2 or 4 bytes)
                FLASHRNN_BACKWARD_WARP_TILING_DIM_HIDDEN * FLASHRNN_BACKWARD_WARP_TILING_DIM_GATE
                * FLASHRNN_BACKWARD_WARP_RECURRENT_CACHED_GATE * BACKWARD_NUM_WARPS
                == BACKWARD_RECURRENT_REGISTER_MEMORY * FLASHRNN_BACKWARD_WARP_TILING_COUNT_BATCH;
                FLASHRNN_BACKWARD_RECURRENT_TILING_COUNT_GATE * FLASHRNN_BACKWARD_WARP_RECURRENT_CACHED_GATE
                * FLASHRNN_BACKWARD_WARP_TILING_COUNT_GATE * FLASHRNN_BACKWARD_WARP_TILING_DIM_GATE
                <= FLASHRNN_NUM_GATES_R * INTERNAL_HEAD_DIM;
                BACKWARD_RECURRENT_REGISTER_MEMORY
                + BACKWARD_NUM_WARPS * FLASHRNN_BACKWARD_WARP_TILING_DIM_BATCH * FLASHRNN_BACKWARD_WARP_TILING_DIM_HIDDEN
                * FLASHRNN_BACKWARD_WARP_LOOPING_COUNT_BATCH * {config._internal_acc_dtype_size // config._internal_dtype_size}
                + BACKWARD_NUM_WARPS * FLASHRNN_BACKWARD_WARP_TILING_DIM_BATCH * FLASHRNN_BACKWARD_WARP_TILING_DIM_GATE
                <= REGISTERS_PER_BLOCK_BACKWARD ;

                # Shared memory size constraint in bytes, add 64 to keep BACKWARD_RECURRENT_SHARED_MEMORY non-zero
                {config._internal_dtype_size} * FLASHRNN_NUM_GATES_R * FLASHRNN_BACKWARD_BLOCK_TILING_COUNT_BATCH * HEAD_DIM_SQ
                + {config._internal_dtype_size} * HEAD_DIM * SHARED_MEMORY_PADDING * FLASHRNN_BACKWARD_BLOCK_TILING_COUNT_BATCH + 64
                == BACKWARD_RECURRENT_SHARED_MEMORY + {config._internal_dtype_size} * BACKWARD_RECURRENT_REGISTER_MEMORY * FLASHRNN_BACKWARD_BLOCK_TILING_COUNT_BATCH
                * FLASHRNN_BACKWARD_RECURRENT_TILING_COUNT_HIDDEN * FLASHRNN_BACKWARD_RECURRENT_TILING_COUNT_GATE ;

                {config._internal_acc_dtype_size} * FLASHRNN_BACKWARD_WARP_TILING_COUNT_GATE * BACKWARD_FULL_COUNT_BATCH
                * FLASHRNN_BACKWARD_WARP_TILING_DIM_BATCH * HEAD_DIM  * FLASHRNN_BACKWARD_RECURRENT_TILING_COUNT_GATE
                + {config._internal_acc_dtype_size} * FLASHRNN_BACKWARD_WARP_TILING_COUNT_GATE * FLASHRNN_BACKWARD_RECURRENT_TILING_COUNT_HIDDEN * FLASHRNN_BACKWARD_RECURRENT_TILING_COUNT_GATE
                * BACKWARD_FULL_COUNT_BATCH * FLASHRNN_BACKWARD_WARP_TILING_DIM_BATCH * SHARED_MEMORY_PADDING == BACKWARD_GATES_SHARED_MEMORY ;
                BACKWARD_RECURRENT_SHARED_MEMORY + BACKWARD_GATES_SHARED_MEMORY == FLASHRNN_BACKWARD_SHARED_MEMORY_PER_HEAD_USED;

                FLASHRNN_BACKWARD_MULTIHEAD_TILING_COUNT * FLASHRNN_BACKWARD_SHARED_MEMORY_PER_HEAD_USED <= SHARED_MEMORY_PER_BLOCK * FLASHRNN_BACKWARD_NUM_BLOCKS ;

            """
            )

            for constant, val in config.constants.items():
                if isinstance(val, int):
                    constraint_str += f"\n{constant} == {val};\n"
            name = config.function + "_f"
            sources = [
                str(curdir / "fused" / "flashrnn.cc"),
                str(curdir / "fused" / "flashrnn_fused_forward.cu"),
                str(curdir / "fused" / "flashrnn_fused_backward.cu"),
                str(curdir / "fused" / "flashrnn_fused_backward_cut.cu"),
                str(curdir / "util" / "blas.cu"),
                str(curdir / "util" / "cuda_error.cu"),
            ]
            seq_len = 2

            # 使用load_parametric_and_test_and_bisect编译cuda fused模块到python
            module = load_parametric_and_test_and_bisect(
                name=name,
                sources=sources,
                constraint_str=constraint_str,
                value_refinements=value_refinements,
                model_class="FlashRNNFuncFused",
                model_args=(
                    True,
                    config.batch_size,
                    config.hidden_dim,
                    config.num_heads,
                ),
                test_input=(
                    True,
                    torch.zeros(
                        [
                            seq_len,
                            config.batch_size,
                            config.num_heads,
                            config.head_dim,
                            config.num_gates_w,
                        ],
                        dtype=config.torch_dtype_w,
                        device="cuda",
                    ),
                    torch.zeros(
                        [
                            config.num_states,
                            config.batch_size,
                            config.num_heads,
                            config.head_dim,
                        ],
                        dtype=config.torch_dtype_s,
                        device="cuda",
                    ),
                    torch.zeros(
                        [
                            config.num_heads,
                            config.head_dim,
                            config.num_gates_r,
                            config.head_dim,
                        ],
                        dtype=config.torch_dtype_r,
                        device="cuda",
                    ),
                    torch.zeros(
                        [config.num_heads, config.head_dim, config.num_gates_t],
                        dtype=config.torch_dtype_b,
                        device="cuda",
                    ),
                ),
                value_to_independently_bisect_upwards_forward="FLASHRNN_FORWARD_WARP_RECURRENT_CACHED_HIDDEN",
                value_to_independently_bisect_upwards_backward="FLASHRNN_BACKWARD_WARP_RECURRENT_CACHED_GATE",
                map_output_to_backward_input=lambda test_input, output: (
                    test_input[3].permute(0, 3, 1, 2).contiguous(),
                    test_input[4].contiguous(),
                    output[0],
                    output[1],
                    output[2],
                    torch.ones_like(output[0]),
                ),
                extra_cflags=config.defines,
                extra_cuda_cflags=[
                    "-include",
                    str(curdir / "fused" / f"{config.function}_fused_pointwise.cuh"),
                ],
            )

            cls.mod[cfgdevstr] = module.FlashRNNFuncFused(
                True,
                config.batch_size,
                config.hidden_dim,
                config.num_heads,
            )
        return cls.mod[cfgdevstr]


"""
CUDA fused版 kernel的生成函数
"""


def FlashRNNFuncGeneratorFused(training, config: FlashRNNConfig):
    # 根据config JIT编译生成一个flashrnn_cuda实例
    flashrnn_cuda = _FlashRNNCUDAFused.instance(config)
    # pad batch size to multiple of 8
    round_batch_size = (
        round_to_multiple(config.batch_size, 8 if config.dtype != "float32" else 16)
        if config.dtype != "float32"
        else 16
    )

    # 将cuda kenel 封装成前向和反向函数类
    class FlashRNNFunctionFused(torch.autograd.Function):
        @staticmethod
        @conditional_decorator(
            config.enable_automatic_mixed_precision,
            torch.amp.custom_fwd(
                device_type="cuda", cast_inputs=DTYPE_DICT[config.dtype]
            ),
        )
        def forward(ctx, training, *inputs):
            # pad input and state for batch size multiple of 8
            seq, bs, nheads, head_dim, gates_w = inputs[0].shape
            if bs % round_batch_size != 0:
                with torch.no_grad():
                    nstat, bs, nheads, head_dim = inputs[1].shape
                    bs_pad = round_batch_size * (
                        (bs + round_batch_size - 1) // round_batch_size
                    )
                    inp = torch.ones(
                        (seq, bs_pad, nheads, head_dim, gates_w),
                        dtype=inputs[0].dtype,
                        device=inputs[0].device,
                    )
                    stat = torch.ones(
                        (nstat, bs_pad, nheads, head_dim),
                        dtype=inputs[0].dtype,
                        device=inputs[0].device,
                    )
                    inp[:, :bs] = inputs[0][:, :bs]
                    stat[:, :bs] = inputs[1][:, :bs]
                    inputs = (inp, stat, inputs[2], inputs[3])
            inputs = (
                inputs[0].to(dtype=config.torch_dtype_w).contiguous(),
                inputs[1].to(dtype=config.torch_dtype_s).contiguous(),
                inputs[2].to(dtype=config.torch_dtype_r).contiguous(),
                inputs[3].to(dtype=config.torch_dtype_b).contiguous(),
            )
            states, cache_g_r, cache_g_i = flashrnn_cuda.forward(training, *inputs)
            if bs % round_batch_size != 0:
                states = states[:, :, :bs]
                cache_g_r = cache_g_r[:, :bs]
                if len(cache_g_i.shape) > 0:
                    cache_g_i = cache_g_i[:, :bs]
            ctx.save_for_backward(*inputs[2:], states, cache_g_r, cache_g_i)
            ctx.training = training
            return states

        @staticmethod
        @conditional_decorator(
            config.enable_automatic_mixed_precision,
            torch.amp.custom_bwd(device_type="cuda"),
        )
        def backward(ctx, states_grads):
            if not ctx.training:
                raise RuntimeError(
                    "FLASHRNN backward can only be called in training mode"
                )

            saved = [*ctx.saved_tensors]
            saved[0] = saved[0].permute(0, 3, 1, 2).contiguous()  # recurrent_kernel
            nstates, seq, bs, nheads, head_dim = states_grads.shape
            if bs % round_batch_size != 0:
                with torch.no_grad():
                    bs_pad = round_batch_size * (
                        (bs + round_batch_size - 1) // round_batch_size
                    )
                    states_grads_pad = torch.zeros(
                        (nstates, seq, bs_pad, nheads, head_dim),
                        dtype=states_grads.dtype,
                        device=states_grads.device,
                    )
                    states_grads_pad[:, :, :bs] = states_grads[:, :, :bs]
                    states_grads = states_grads_pad
                    states_pad = torch.ones(
                        (config.num_states, seq, bs_pad, nheads, head_dim),
                        dtype=saved[2].dtype,
                        device=saved[2].device,
                    )
                    states_pad[:, :, :bs] = saved[2][:, :, :bs]
                    cache_g_r_pad = torch.zeros(
                        (seq - 1, bs_pad, nheads, head_dim, saved[0].shape[3]),
                        dtype=saved[0].dtype,
                        device=saved[0].device,
                    )
                    cache_g_r_pad[:, :bs] = saved[3][:, :bs]
                    saved[2] = states_pad
                    saved[3] = cache_g_r_pad

                    if len(saved[4].shape) > 0:
                        cache_g_i_pad = torch.zeros(
                            (seq - 1, bs_pad, nheads, head_dim, saved[1]),
                            dtype=saved[4].dtype,
                            device=saved[4].device,
                        )
                        cache_g_i_pad[:, :bs] = saved[4][:, :bs]
                        saved[4] = cache_g_i_pad
            if config.gradient_recurrent_cut:
                grads = flashrnn_cuda.backward_cut(*saved, states_grads.contiguous())
            else:
                grads = flashrnn_cuda.backward(*saved, states_grads.contiguous())
            # print("grads[0] before:", grads[0].shape)

            with torch.no_grad():
                S, B, nheads, head_dim, _ = grads[0].shape
                if config.num_gates_w != config.num_gates_t:
                    # gru
                    grads[0] = (
                        grads[0]
                        .view(S, B, nheads, head_dim, config.num_gates_i)[
                            :, :, :, :, (config.num_gates_i - config.num_gates_w) :
                        ]
                        .permute(
                            0, 1, 2, 4, 3
                        )  # 交换最后两个维度：从 (head_dim, num_gates_w) → (num_gates_w, head_dim)
                        # .reshape(S, B, -1)
                    )
                    # print("grads[0] after:", grads[0].shape)

                if bs % round_batch_size != 0:
                    grads[0] = grads[0][:, :bs]
                    grads[1] = grads[1][:, :bs]

            return (None, *grads)

    return FlashRNNFunctionFused


# 根据 FlashRNNConfig 中的 _internal_input_permutation 来决定是否调整输入张量 x 的维度顺序。
def _permute_input(config: FlashRNNConfig, x: torch.Tensor) -> torch.Tensor:
    if config._internal_input_permutation is None:
        return x
    else:
        return x.permute(config._internal_input_permutation)


def _permute_recurrent_weight(config: FlashRNNConfig, R: torch.Tensor) -> torch.Tensor:
    if config._internal_recurrent_permutation is None:
        return R
    else:
        return R.permute(config._internal_recurrent_permutation)


def _permute_bias(config: FlashRNNConfig, b: torch.Tensor) -> torch.Tensor:
    if config._internal_bias_permutation is None:
        return b
    else:
        return b.permute(config._internal_bias_permutation)


def _permute_output(config: FlashRNNConfig, x: torch.Tensor) -> torch.Tensor:
    if config._internal_output_permutation is None:
        return x
    else:
        return x.permute(config._internal_output_permutation)


def _permute_output_backward(config: FlashRNNConfig, x: torch.Tensor) -> torch.Tensor:
    if config._internal_output_backward_permutation is None:
        return x
    else:
        return x.permute(config._internal_output_backward_permutation)


def _zero_state(config: FlashRNNConfig, inp: torch.Tensor) -> torch.Tensor:
    """Returns a nested structure of zero Tensors with the same structure
    and shape as []. The returned Tensors will have the same
    dtype and be on the same device as `inp`.

    Arguments:
        inp: Tensor, to specify the device and dtype of the returned tensors.
        shape_state: nested structure of integers.

    Returns:
        zero_state: a nested structure of zero Tensors.

    Raises:
        ValueError: if `state_shape` has non-integer values.
    """
    batch_dim = inp.shape[config.input_shape.index("B")]
    state = torch.zeros(
        (config.num_states, batch_dim, config.num_heads, config.head_dim),
        dtype=inp.dtype,
        device=inp.device,
    )
    with torch.no_grad():
        if isinstance(config.initial_val, float):
            state += config.initial_val
        else:
            for i in range(config.num_states):
                state[i] += config.initial_val[i]
    return state[None, :].permute(permute_to("TSBHD", config.output_shape))


# 封装成使用kernel的RNN nn.Module
class _FlashRNNCudaFusedLayer(torch.nn.Module):
    def __init__(self, Wx, R, b, config):
        super(_FlashRNNCudaFusedLayer, self).__init__()
        self.Wx = _permute_input(config, Wx)
        self.R = _permute_recurrent_weight(config, R)
        self.b = _permute_bias(config, b)
        self.config = config

    def forward(self, states):
        kernel = FlashRNNFuncGeneratorFused(torch.is_grad_enabled(), config=self.config)
        states = kernel.apply(
            torch.is_grad_enabled(),
            self.Wx.contiguous(),
            states[:, 0].contiguous(),
            self.R.contiguous(),
            self.b.contiguous(),
        )
        # h = states[:, 1:]
        # last_h = states[:, -1:]
        # return _permute_output(self.config, h), _permute_output(self.config, last_h)
        return states


class FlashRNNCudaFused(torch.nn.Module):
    def __init__(
        self,
        Wx_list,
        R_list,
        b_list,
        config=None,
        num_layers=1,
        function="lstm",
    ):
        super().__init__()
        assert len(Wx_list) == len(R_list) == len(b_list) == num_layers
        self.num_layers = num_layers
        self.layers = torch.nn.ModuleList()
        if config == None:
            config = _get_config(
                Wx_list[0],
                R_list[0],
                b_list[0],
                function=function,
                backend="cuda_fused",
            )
        for i in range(num_layers):

            self.layers.append(
                # _FlashRNNCudaFusedLayer(Wx_list[i], R_list[i], b_list[i], config)   # 这是浅拷贝，每个layer都是用同一个config
                _FlashRNNCudaFusedLayer(Wx_list[i], R_list[i], b_list[i], config)
            )
        self.config = config  # use first layer's config

    def forward(self, states=None):
        if states is None:
            states = _zero_state(self.config, self.layers[0].Wx)

        out = states
        for layer in self.layers:
            out = layer(out)

        h = out[:, 1:]
        last_h = out[:, -1:]
        return h, last_h, out


def build_flashrnn_stack(
    batch_size,
    seq_size,
    num_gate,
    num_heads=1,
    hidden_dim=768,
    num_layers=1,
    config=None,
    dtype_str="bfloat16",
    function="lstm",
):
    dtype = getattr(torch, dtype_str)
    Wx_list = []
    R_list = []
    b_list = []
    # config_list = []

    # 生成各层网络的输入（Wx，R，b）、配置config
    for _ in range(num_layers):
        # Wx shape: [B, T, NG, NH, D]
        Wx = torch.randn(
            batch_size,
            seq_size,
            num_gate,
            num_heads,
            hidden_dim,
            device="cuda",
            dtype=dtype,
            requires_grad=True,
        )

        # R shape: [NG, NH, D, D]
        R = torch.randn(
            num_gate,
            num_heads,
            hidden_dim,
            hidden_dim,
            device="cuda",
            dtype=dtype,
            requires_grad=True,
        )

        # b shape: [NG, NH, D]
        b = torch.randn(
            num_gate,
            num_heads,
            hidden_dim,
            device="cuda",
            dtype=dtype,
            requires_grad=True,
        )
        if config == None:
            config = _get_config(
                Wx, R, b, function=function, backend="cuda_fused", dtype=dtype_str
            )

        Wx_list.append(Wx)
        R_list.append(R)
        b_list.append(b)
    # print("function: ", function, config.function)
    model = FlashRNNCudaFused(
        Wx_list,
        R_list,
        b_list,
        config=config,
        num_layers=num_layers,
        function=function,
    )
    return model


""" FlashRNN 的入口函数
    Wx: torch.Tensor,                       # [T, B, G_in, N, I] 输入门数据
    R: torch.Tensor,                        # [H, P, G_r, D] Recurrent 权重
    b: torch.Tensor,                        # [H, G_b, D] Bias
    states: Optional[torch.Tensor] = None,  # 初始状态 [1, B, S, N, D]
    function: str = "lstm",                 # 使用的单元类型（lstm, gru, slstm等）
    config: Optional[FlashRNNConfig] = None,# 可选的完整配置对象
    backend: str = "cuda_fused",            # 后端选择
    dtype: str = "bfloat16",                # 数据精度
"""


def flashrnn_fused(
    Wx: torch.Tensor,
    R: torch.Tensor,
    b: torch.Tensor,
    states: Optional[torch.Tensor] = None,
    function: str = "lstm",
    config: Optional[FlashRNNConfig] = None,
    backend: str = "cuda_fused",
    dtype: str = "bfloat16",
):
    if config is None:
        config = _get_config(Wx, R, b, function, backend, dtype=dtype)

    # kernel = _get_kernel(config)
    kernel = FlashRNNFuncGeneratorFused(torch.is_grad_enabled(), config=config)
    if states is None:
        states = _zero_state(config, Wx)

    # permute 维度调整，确保数据对齐内核
    states = _permute_output_backward(config, states)
    Wx = _permute_input(config, Wx)
    R = _permute_recurrent_weight(config, R)
    b = _permute_bias(config, b)

    # torch.autograd.Function要使用apply调用

    states = kernel.apply(
        torch.is_grad_enabled(),
        Wx.contiguous(),
        states[:, 0].contiguous(),
        R.contiguous(),
        b.contiguous(),
    )
    h = states[:, 1:]
    last_h = states[:, -1:]
    return _permute_output(config, h), _permute_output(config, last_h)
