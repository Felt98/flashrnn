import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
from .config import FlashRNNConfig, permute_to, DTYPE_DICT_REV

from .triton_fused.fwbw import lstm_tr_fwbw

import torch


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


# 根据 FlashRNNConfig 中的 _internal_input_permutation 来决定重排输入张量 x 的维度顺序。
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


# 对_get_kernel_step对扩充
def _get_kernel(config: FlashRNNConfig):
    if config.backend == "triton_fused":
        if config.function == "lstm":
            from .triton_fused.fwbw import lstm_tr_fwbw

            def fn(Wx, states, R, b, **kwargs):
                return lstm_tr_fwbw(
                    states_initial=states,
                    Wx=Wx,
                    R=R,
                    b=b,
                    backward_recurrent_clip_val=config.gradient_recurrent_clipval,
                    autocast_kernel_dtype=config.dtype,
                )

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


class _FlashRNNTritonFusedLayer(torch.nn.Module):
    def __init__(self, Wx, R, b, config):
        super(_FlashRNNTritonFusedLayer, self).__init__()
        self.Wx = _permute_input(config, Wx)
        self.R = _permute_recurrent_weight(config, R)
        self.b = _permute_bias(config, b)
        self.config = config
        self.backward_recurrent_clip_val = config.gradient_recurrent_clipval

    def forward(self, states):
        states = lstm_tr_fwbw(
            states_initial=states,
            Wx=self.Wx,
            R=self.R,
            b=self.b,
            backward_recurrent_clip_val=self.config.gradient_recurrent_clipval,
            autocast_kernel_dtype=self.config.dtype,
        )
        # return states    lstm_tr_fwbw返回的是一个tuple
        return states


class FlashRNNTritonFused(torch.nn.Module):
    def __init__(
        self,
        Wx_list,
        R_list,
        b_list,
        config=None,
        num_layers=1,
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
                function="lstm",
                backend="triton_fused",
            )
        for i in range(num_layers):
            self.layers.append(
                _FlashRNNTritonFusedLayer(Wx_list[i], R_list[i], b_list[i], config)
            )

        # self.config = self.layers[0].config  # use first layer's config
        self.config = config  # use first layer's config

    def forward(self, states=None):
        if states is None:
            states = _zero_state(self.config, self.layers[0].Wx)

        out = states
        h = None
        last_h = None
        for layer in self.layers:
            # 可能存在问题 是传 h 还是 last_h
            h, last_h = layer(out)
            out = last_h
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
                Wx, R, b, function="lstm", backend="triton_fused", dtype=dtype_str
            )

        Wx_list.append(Wx)
        R_list.append(R)
        b_list.append(b)
    model = FlashRNNTritonFused(
        Wx_list,
        R_list,
        b_list,
        config=config,
        dtype_str=dtype_str,
        num_layers=num_layers,
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


def flashrnn_triton(
    Wx: torch.Tensor,
    R: torch.Tensor,
    b: torch.Tensor,
    states: Optional[torch.Tensor] = None,
    function: str = "lstm",
    config: Optional[FlashRNNConfig] = None,
    backend: str = "cuda_fused",
    dtype: str = "bfloat16",
):
    """_summary_

    Args:
        Wx (torch.Tensor): _description_
        R (torch.Tensor): _description_
        b (torch.Tensor): _description_
        states (Optional[torch.Tensor], optional): _description_. Defaults to None.
        function (str, optional): _description_. Defaults to "lstm".
        config (Optional[FlashRNNConfig], optional): _description_. Defaults to None.
        backend (str, optional): _description_. Defaults to "cuda_fused".
        dtype (str, optional): _description_. Defaults to "bfloat16".

    Returns:
        _type_: _description_
    """
    if config is None:
        config = _get_config(Wx, R, b, function, backend, dtype=dtype)

    kernel = _get_kernel(config)
    if states is None:
        states = _zero_state(config, Wx)

    # permute 维度重拍
    states = _permute_output_backward(config, states)
    print("states shape : ", states.shape)
    Wx = _permute_input(config, Wx)
    R = _permute_recurrent_weight(config, R)
    b = _permute_bias(config, b)
    h, last_h = kernel(Wx, states, R, b)
    print(" h shape : ", h.shape)
    print(" last_h shape : ", last_h.shape)
    print("_permute_output(config, h) shape: ", _permute_output(config, h).shape)
    print(
        "_permute_output(config, last_h) shape: ", _permute_output(config, last_h).shape
    )
    return _permute_output(config, h), _permute_output(config, last_h)
