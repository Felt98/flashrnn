# Copyright 2024 NXAI GmbH
# Korbinian Poeppel

import copy
import os
from pathlib import Path
from typing import Optional
from .config import FlashRNNConfig, permute_to, DTYPE_DICT_REV, DTYPE_DICT, _get_config
import torch

from torch.autograd.function import once_differentiable

from autotune.constrint import ValueHeuristic, ValueRefinement
from .cuda_init import load
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
编译并缓存一个 FlashRNN 的 CUDA alternate 版模块到 python中使用的类
被FlashRNNFuncGenerator使用
"""


class _FlashRNNCUDA:
    mod = {}  # 记录特定配置cuda模块的map，所有_FlashRNNCUDA类共享

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
                    str(curdir / "alternating" / "flashrnn.cc"),
                    str(curdir / "alternating" / "flashrnn_forward.cu"),
                    str(curdir / "alternating" / "flashrnn_backward.cu"),
                    str(curdir / "alternating" / "flashrnn_backward_cut.cu"),
                    str(curdir / "alternating" / f"{config.function}_pointwise.cu"),
                    str(curdir / "util" / "blas.cu"),
                    str(curdir / "util" / "cuda_error.cu"),
                ],
                extra_cflags=[
                    f"-D{const}={constval}"
                    for const, constval in config.constants.items()
                ]
                + config.defines,
            )

            # 从module模块中实例化 FlashRNNFunc 对象并缓存到cls.mod[cfgdevstr]
            cls.mod[cfgdevstr] = module.FlashRNNFunc(
                True, config.batch_size, config.hidden_dim, config.num_heads
            )
        # 返回已经加载并初始化好的 FlashRNNFunc 对象
        return cls.mod[cfgdevstr]


"""
CUDA alternate版 kernel的生成函数，生成一个自定义的 PyTorch 自动求导函数类,用于封装 FlashRNN 的前向和反向 CUDA 计算逻辑。
forward调用流程：
    Python 调用FlashRNNFuncGenerator生成flashrnn_cuda
    ↓
    flashrnn_cuda.forward(...) ⬅️ 在 Python 中被调用
    ↓
    module.FlashRNNFunc.forward(...) ⬅️ 通过 JIT 编译返回的模块
    ↓
    flashrnn.cc 中注册了名为 "forward" 的函数 对应 FlashRNNFunc::forward（alternate版）
"""


def FlashRNNFuncGenerator(training, config: FlashRNNConfig):
    flashrnn_cuda = _FlashRNNCUDA.instance(config=config)  # CUDA 后端模块

    # 内部调用cuda kernel类FlashRNNFunction，需要继承自torch.autograd.Function
    class FlashRNNFunction(torch.autograd.Function):

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
                raise RuntimeError(
                    "FlashRNN backward can only be called in training mode"
                )
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
            return (None, *grads)

    return FlashRNNFunction


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
        # print(
        #     "config._internal_output_backward_permutation: ",
        #     config._internal_output_backward_permutation,
        # )
        # print("output_backward_permutation: ", x.shape)
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
class _FlashRNNCudaLayer(torch.nn.Module):
    def __init__(self, Wx, R, b, config=None):
        # if config == None:
        #     config = _get_config(Wx, R, b, function, backend="cuda_fused", dtype=dtype)

        super(_FlashRNNCudaLayer, self).__init__()
        self.Wx = _permute_input(config, Wx)
        self.R = _permute_recurrent_weight(config, R)
        self.b = _permute_bias(config, b)
        self.config = config

    def forward(self, states):
        kernel = FlashRNNFuncGenerator(torch.is_grad_enabled(), config=self.config)
        # states = _permute_output_backward(self.config, states)

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


class FlashRNNCuda(torch.nn.Module):
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
                backend="cuda",
            )
        for i in range(num_layers):

            self.layers.append(
                _FlashRNNCudaLayer(Wx_list[i], R_list[i], b_list[i], config)
            )

        # self.config = self.layers[0].config  # use first layer's config
        self.config = config  # use first layer's config

    def forward(self, states=None):
        if states is None:
            states = _zero_state(self.config, self.layers[0].Wx)
        # print("_permute_output_backward after:", states.shape)

        out = states
        # out = _permute_output_backward(self.config, out)

        for layer in self.layers:
            out = layer(out)
            # out = _permute_output_backward(self.config, out)
        # print("out ---_permute_output_backward after:", out.shape)

        h = out[:, 1:]
        last_h = out[:, -1:]
        return h, last_h, out


# def build_flashrnn_stack(
#     batch_size,
#     seq_size,
#     num_gate,
#     num_heads=1,
#     hidden_dim=768,
#     num_layers=1,
#     config=None,
#     dtype_str="bfloat16",
# ):
#     dtype = getattr(torch, dtype_str)
#     Wx_list = []
#     R_list = []
#     b_list = []
#     # config_list = []

#     # 生成各层网络的输入（Wx，R，b）、配置config
#     for _ in range(num_layers):
#         # Wx shape: [B, T, NG, NH, D]
#         Wx = torch.randn(
#             batch_size,
#             seq_size,
#             num_gate,
#             num_heads,
#             hidden_dim,
#             device="cuda",
#             dtype=dtype,
#             requires_grad=True,
#         )

#         # R shape: [NG, NH, D, D]
#         R = torch.randn(
#             num_gate,
#             num_heads,
#             hidden_dim,
#             hidden_dim,
#             device="cuda",
#             dtype=dtype,
#             requires_grad=True,
#         )

#         # b shape: [NG, NH, D]
#         b = torch.randn(
#             num_gate,
#             num_heads,
#             hidden_dim,
#             device="cuda",
#             dtype=dtype,
#             requires_grad=True,
#         )
#         if config == None:
#             config = _get_config(
#                 Wx, R, b, function="lstm", backend="cuda", dtype=dtype_str
#             )

#         Wx_list.append(Wx)
#         R_list.append(R)
#         b_list.append(b)
#     model = FlashRNNCuda(
#         Wx_list,
#         R_list,
#         b_list,
#         config=config,
#         num_layers=num_layers,
#     )
#     return model


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


def flashrnn_alternating(
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
    kernel = FlashRNNFuncGenerator(torch.is_grad_enabled(), config=config)
    if states is None:
        states = _zero_state(config, Wx)

    # permute 维度调整，确保对齐backend所需的维度
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
