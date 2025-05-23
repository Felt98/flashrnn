# Copyright 2024 NXAI GmbH
# Korbinian Poeppel
import os
from pathlib import Path
from typing import Optional

from .flashrnn_triton import FlashRNNTritonFused
from .config import FlashRNNConfig, permute_to, DTYPE_DICT_REV, DTYPE_DICT
from .flashrnn_alternating import FlashRNNCuda, FlashRNNFuncGenerator
from .flashrnn_fused import FlashRNNCudaFused
import torch

from autotune.constrint import ValueHeuristic, ValueRefinement
from .vanilla import (
    flashrnn_forward,
    flashrnn_forward_step,
    flashrnn_pointwise_function_registry,
)


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


#  RNN forward函数：由后端类型（vanilla、cuda、cuda_fused）返回不同的前向计算函数 fn。
def _get_kernel_step(config: FlashRNNConfig):

    # 纯 Python手动实现的网络
    if config.backend == "vanilla":

        def fn(Wx, states, R, b, **kwargs):
            return flashrnn_forward_step(
                Wx,
                states,
                R,
                b,
                pointwise_forward=flashrnn_pointwise_function_registry[config.function],
                constants=config.constants,
                **kwargs,
            )

    elif config.backend == "cuda" or config.backend == "cuda_fused":

        def fn(Wx, states, R, b, **kwargs):
            # 生成cuda kernel FlashRNNFuncGenerator实例
            states = FlashRNNFuncGenerator(
                torch.is_grad_enabled(), config=config
            ).apply(  # 注意必须通过 .apply(...) 来运行前向传播并建立计算图，供后续 backward 使用
                torch.is_grad_enabled(),
                Wx.contiguous(),
                states[:, 0].contiguous(),
                R.contiguous(),
                b.contiguous(),
            )
            return states[:, 1:], states[:, -1:]

    return fn


# 对_get_kernel_step对扩充
def _get_kernel(config: FlashRNNConfig):
    if config.backend == "vanilla":

        def fn(Wx, states, R, b, **kwargs):
            return flashrnn_forward(
                Wx,
                states,
                R,
                b,
                pointwise_forward=flashrnn_pointwise_function_registry[config.function],
                constants=config.constants,
                **kwargs,
            )

    # 与vanilla不同的是 支持前向+反向一次完成 的优化实现
    elif config.backend == "vanilla_fwbw":
        if config.function == "lstm":
            from .vanilla_fwbw.fwbw import lstm_pt_fwbw

            def fn(Wx, states, R, b, **kwargs):
                return lstm_pt_fwbw(
                    states_initial=states,
                    Wx=Wx,
                    R=R,
                    b=b,
                    backward_recurrent_clip_val=config.gradient_recurrent_clipval,
                    autocast_kernel_dtype=config.dtype,
                )

        elif config.function == "slstm":
            from .vanilla_fwbw.fwbw import slstm_pt_fwbw

            def fn(Wx, states, R, b, **kwargs):
                return slstm_pt_fwbw(
                    states_initial=states,
                    Wx=Wx,
                    R=R,
                    b=b,
                    backward_recurrent_clip_val=config.gradient_recurrent_clipval,
                    autocast_kernel_dtype=config.dtype,
                )

        else:
            raise NotImplementedError(
                f"Function {config.function} not implemented for vanilla_fwbw backend."
            )

    # alternate版 cuda
    elif config.backend == "cuda":

        def fn(Wx_list, states, R_list, b_list, num_layers=1, **kwargs):
            model = FlashRNNCuda(
                Wx_list,
                R_list,
                b_list,
                config=config,
                num_layers=num_layers,
            )
            return model(states)

    elif config.backend == "cuda_fused":

        def fn(Wx_list, states, R_list, b_list, num_layers=1, **kwargs):
            model = FlashRNNCudaFused(
                Wx_list,
                R_list,
                b_list,
                config=config,
                num_layers=num_layers,
            )
            return model(states)

    elif config.backend == "triton_fused":
        if config.function == "lstm":
            from .triton_fused.fwbw import lstm_tr_fwbw

            def fn(Wx_list, states, R_list, b_list, num_layers=1, **kwargs):
                model = FlashRNNTritonFused(
                    Wx_list,
                    R_list,
                    b_list,
                    config=config,
                    num_layers=num_layers,
                )
                return model(states)

        elif config.function == "slstm":
            from .triton_fused.fwbw import slstm_tr_fwbw

            def fn(Wx, states, R, b, **kwargs):
                return slstm_tr_fwbw(
                    states_initial=states,
                    Wx=Wx,
                    R=R,
                    b=b,
                    backward_recurrent_clip_val=config.gradient_recurrent_clipval,
                    autocast_kernel_dtype=config.dtype,
                )

    else:
        raise ValueError(f"Unknown backend {config.backend}")

    return fn


def _get_config(
    Wx: torch.Tensor,
    R: torch.Tensor,
    b: torch.Tensor,
    function: str,
    backend: str,
    dtype: Optional[str],
) -> FlashRNNConfig:
    return FlashRNNConfig(
        head_dim=Wx.shape[4],
        num_heads=Wx.shape[3],
        batch_size=Wx.shape[0],
        function=function,
        backend=backend,
        dtype=dtype if dtype is not None else "bfloat16",
        dtype_w=DTYPE_DICT_REV[Wx.dtype],
        dtype_r=DTYPE_DICT_REV[R.dtype],
        dtype_b=DTYPE_DICT_REV[b.dtype],
    )


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


def flashrnn(
    Wx: torch.Tensor,
    R: torch.Tensor,
    b: torch.Tensor,
    states: Optional[torch.Tensor] = None,
    function: str = "lstm",
    config: Optional[FlashRNNConfig] = None,
    backend: str = "cuda_fused",
    dtype: str = "bfloat16",
):
    if backend in ("vanilla", "vanilla_fwbw"):
        backend = "cuda_fused"
    if config is None:
        config = _get_config(Wx, R, b, function, backend, dtype=dtype)

    kernel = _get_kernel(config)
    if states is None:
        states = _zero_state(config, Wx)

    # permute 维度调整，确保数据对齐内核
    states = _permute_output_backward(config, states)

    print("function: ", function)
    print("backend: ", backend)
    print("Wx: ", _permute_input(config, Wx).shape)
    print("R: ", _permute_recurrent_weight(config, R).shape)
    print("b: ", _permute_bias(config, b).shape)
    print("input states: ", states.shape)

    Wx_list = [Wx]
    R_list = [R]
    b_list = [b]
    h, last_h, out = kernel(
        Wx_list,
        states,
        R_list,
        b_list,
    )
    print("h: ", h.shape)
    print("last_h: ", last_h.shape)
    print("out: ", out.shape)
    return _permute_output(config, h), _permute_output(config, last_h)
