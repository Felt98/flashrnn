import os
from pathlib import Path
from typing import Optional
import torch
from torch.autograd.function import once_differentiable
from ...config import FlashRNNConfig, DTYPE_DICT
from ...cuda_init import load

curdir = Path(os.path.split(os.path.os.path.abspath(__file__))[0])
flashrnn_path = curdir.parent.parent


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


def permute_to(input_shape, output_shape) -> Optional[list[int]]:
    """
    >>> permute_to("ABC", "BAC")
    (1, 0, 2)
    """
    if input_shape == output_shape:
        return None
    p = []
    for x in output_shape:
        p.append(input_shape.index(x))
    return tuple(p)


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
编译并缓存一个 LSTM 的 CUDA alternate 版模块到 python中使用的类
被LSTMFuncGenerator使用
"""


class _LSTMCUDA:
    mod = {}  # 记录特定配置cuda模块的map，所有_LSTMCUDA类共享

    @classmethod
    def instance(cls, config: FlashRNNConfig):
        cfgdevstr = (
            repr(config) + f"_{torch.cuda.current_device()}"
        )  # 构造一个唯一标识当前module模块配置的 key= config + GPU编号
        if cfgdevstr not in cls.mod:  # 如果当前配置不在cls.mod中：
            # JIT方式加载cuda模块到python module
            # load在cuda_init.py中
            module = load(
                name=config.function,
                sources=[
                    # str(curdir / "alternating" / "flashrnn.cc"),
                    # str(curdir / "alternating" / "flashrnn_forward.cu"),
                    # str(curdir / "alternating" / "flashrnn_backward.cu"),
                    # str(curdir / "alternating" / "flashrnn_backward_cut.cu"),
                    # str(curdir / "alternating" / f"{config.function}_pointwise.cu"),
                    str(flashrnn_path / "alternating" / "lstm" / "lstm.cc"),
                    str(flashrnn_path / "alternating" / "lstm" / "lstm_forward.cu"),
                    str(flashrnn_path / "alternating" / "lstm" / "lstm_backward.cu"),
                    str(
                        flashrnn_path / "alternating" / "lstm" / "lstm_backward_cut.cu"
                    ),
                    str(flashrnn_path / "alternating" / "lstm" / "lstm_pointwise.cu"),
                    str(flashrnn_path / "util" / "blas.cu"),
                    str(flashrnn_path / "util" / "cuda_error.cu"),
                ],
                extra_cflags=[
                    f"-D{const}={constval}"
                    for const, constval in config.constants.items()
                ]
                + config.defines,
            )

            # 从module模块中实例化 LSTMFunc 对象并缓存到cls.mod[cfgdevstr]
            cls.mod[cfgdevstr] = module.LSTMFunc(
                True, config.batch_size, config.hidden_dim, config.num_heads
            )
        # 返回已经加载并初始化好的 LSTMFunc 对象
        return cls.mod[cfgdevstr]


"""
CUDA alternate版 kernel的生成函数，生成一个自定义的 PyTorch 自动求导函数类,用于封装 LSTM 的前向和反向 CUDA 计算逻辑。
forward调用流程：
    Python 调用LSTMFuncGenerator生成flashrnn_cuda
    ↓
    flashrnn_cuda.forward(...) ⬅️ 在 Python 中被调用
    ↓
    module.LSTMFunc.forward(...) ⬅️ 通过 JIT 编译返回的模块
    ↓
    flashrnn.cc 中注册了名为 "forward" 的函数 对应 LSTMFunc::forward（alternate版）
"""


def LSTMFuncGenerator(training, config: FlashRNNConfig):
    flashrnn_cuda = _LSTMCUDA.instance(config=config)  # CUDA 后端模块

    # 内部调用cuda kernel类LSTMFunction，需要继承自torch.autograd.Function
    class LSTMFunction(torch.autograd.Function):

        ### 前向传播
        @staticmethod
        # 如果启用 自动混合精度AMP，就加上 torch.amp.custom_fwd 装饰器
        @conditional_decorator(
            config.enable_automatic_mixed_precision,
            torch.amp.custom_fwd(
                device_type="cuda", cast_inputs=DTYPE_DICT[config.dtype]
            ),
        )

        # 调用编译后的 CUDA 模块的 .forward()
        def forward(ctx, training, *inputs):
            # 自动混合精度（AMP）
            if config.enable_automatic_mixed_precision:
                inputs = (
                    inputs[0].to(dtype=config.torch_dtype_w),
                    inputs[1].to(dtype=config.torch_dtype_s),
                    inputs[2].to(dtype=config.torch_dtype_r),
                    inputs[3].to(dtype=config.torch_dtype_b),
                )

            # 调用flashrnn_cuda的fuse forward
            states, cache_g_r, cache_g_i = flashrnn_cuda.forward(training, *inputs)

            # 保存输入和中间结果，用于反向传播时使用
            ctx.save_for_backward(*inputs[2:], states, cache_g_r, cache_g_i)
            ctx.training = training

            # 返回隐藏状态state
            return states

        ### 反向传播
        @staticmethod
        @once_differentiable
        @conditional_decorator(
            config.enable_automatic_mixed_precision,
            torch.amp.custom_bwd(device_type="cuda"),
        )
        def backward(ctx, grad_s):
            if not ctx.training:
                raise RuntimeError("LSTM backward can only be called in training mode")
            saved = [*ctx.saved_tensors]
            saved[0] = saved[0].permute(0, 2, 3, 1).contiguous()  # transpose R
            if config.gradient_recurrent_cut:
                grads = flashrnn_cuda.backward_cut(*saved, grad_s.contiguous())
            else:
                grads = flashrnn_cuda.backward(*saved, grad_s.contiguous())
            with torch.no_grad():
                S, B, num_heads, wgates, head_dim = grads[0].shape
                if config.num_gates_w != config.num_gates_t:
                    wgrad = grads[0].view(S, B, num_heads, config.num_gates_i, head_dim)
                    wgrad = wgrad[:, :, :, (config.num_gates_i - config.num_gates_w) :]
                    grads[0] = wgrad.reshape(
                        S, B, num_heads, config.num_gates_w, head_dim
                    )
            # print("grads:", [g.shape if g is not None else None for g in grads])
            return (None, *grads)

    return LSTMFunction
