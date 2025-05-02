import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional, Sequence, Union
from .triton_fused.fwbw import lstm_tr_fwbw

import torch


# from .cuda_init import load
# from cuda_init_parametric import load_parametric_and_test_and_bisect
# from gpu_info.gpu_info import get_gpu_info

# from .vanilla import (
#     flashrnn_forward,
#     flashrnn_forward_step,
#     flashrnn_pointwise_function_registry,
# )

LOGGER = logging.getLogger(__name__)


DTYPE_DICT = {
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
    "float32": torch.float32,
    "float64": torch.float64,
}
DTYPE_DICT_REV = {
    torch.bfloat16: "bfloat16",
    torch.float: "float32",
    torch.float16: "float16",
    torch.float64: "float64",
}

DTYPES = Literal["bfloat16", "float16", "float32"]

curdir = Path(os.path.split(os.path.os.path.abspath(__file__))[0])

# maps the rnn function to the following values
#   gates_states = (num_gates_r, num_gates_w, num_gates_i, num_gates_t, num_states)
#   constants = {}, # whatever the function needs
#   simple_agg = True/False # whether the function simply adds pre-activations Wx + Ry
#   initial_val = 0.  # initial value of the states
rnn_function_registry = {
    # standard variants, all connect
    "slstm": {
        "gates_states": (4, 4, 4, 4, 4),
        "constants": {},
        "simple_agg": True,
    },
    "lstm": {
        "gates_states": (4, 4, 4, 4, 2),
        "constants": {},  # no constants here
        "simple_agg": True,
    },
    "gru": {
        "gates_states": (3, 3, 4, 4, 1),
        "constants": {},  # no constants here
        "simple_agg": False,
    },
    "elman": {
        "gates_states": (1, 1, 1, 1, 1),
        "constants": {},  # no constants here
        "simple_agg": True,
    },
}

_python_dtype_to_cuda_dtype = {
    "float32": "float",
    "float": "float",
    "float16": "__half",
    "bfloat16": "__nv_bfloat16",
}


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
    (0,1,2) -> (1, 0, 2)
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


# flashRNN的各种启用配置类
@dataclass
class FlashRNNConfig:
    backend: Literal[
        "vanilla", "vanilla_fwbw", "cuda", "cuda_fused", "triton_fused"
    ] = "cuda_fused"
    # the type of function a cell computes
    function: str = "lstm"
    # this option cuts of the gradient for recurrent connection, i.e. no exploding gradient if False
    gradient_recurrent_cut: bool = False
    # this option clips the gradient values for recurrent connections at dy
    gradient_recurrent_clipval: Optional[float] = None
    # this option clips the y value
    forward_clipval: Optional[float] = None
    # additional scalar constants that might be modified
    constants: Optional[dict[str, float]] = None
    # whether all gate aggregations are of the type "R @ y + W @ x + b" or if there is a function g_r(R @ y) involved
    # this roughly doubles the memory （backward需要的内存翻倍） needed to be stored for backward
    simple_agg: bool = True

    hidden_dim: int = -1
    num_heads: int = (
        -1
    )  # this must divide the hidden size （num_heads要被hidden size整除）, is not yet supported by all versions in this directory
    head_dim: int = (
        -1
    )  # alternative to num_heads, equals to head_dim = hidden_dim // num_heads
    num_states: int = 4  # this is for the sLSTM, a standard LSTM  has 2

    num_gates_r: int = (
        4  # how many gates take recurrent input， LSTM 的4个门 input、forget、output、cell gate 都接收上一个时间步的隐藏状态
    )
    num_gates_w: int = (
        4  # how many gates take external input，所有门也都接收当前时间步输入向量 x 的投影Wx
    )
    num_gates_i: int = (
        4  # how many gates interact between cells (i.e. r and w together)，LSTM的每个门都参与 recurrent 和 input 的融合
    )
    num_gates_t: int = (
        4  # how many gates are there in total (including biases only)，4个门 input、forget、output、cell gate
    )
    # the gate order is as follows in case some are reduced (i.e. gates_r)
    # [gates_r ... ...]
    # [... gates_w ...]
    # [  gates_i   ...]
    # [    gates_t    ]

    # this can be ignored internally, but may be used to optimize kernels
    batch_size: int = 8

    # B = batch, T time/sequence dim, N num heads, S state dimension, P previous D dimension (R matrix)
    # D head dim or hidden dim, G gates
    input_shape: Literal["BTGHD", "TBGHD"] = "BTGHD"
    output_shape: Literal[
        "SBHTD",
        "STBHD",
        "STBHD",
    ] = "SBTHD"

    recurrent_shape: Literal["GHDP", "HGDP", "HPGD", "HPDG"] = "GHDP"
    bias_shape: Literal["GHD", "HGD", "HDG"] = "GHD"

    # internal shapes are overwritten by backend
    # if you use the shape when calling from outside you minimize transposes
    # Literal中是维度标签的简写，例如"TBGHD" => [Time, Batch, Gate, Head, Dim] => x.shape = [T, B, G, H, D]
    _internal_input_shape: Literal["TBGHD", "TBHGD", "TBHDG"] = "TBGHD"
    _internal_recurrent_shape: Literal["GHDP", "HGDP", "HPGD", "HPDG"] = "GHDP"
    _internal_bias_shape: Literal["GHD, HGD, HDG"] = "HDG"
    _internal_output_shape: Literal["TSHBD", "BTSHD"] = "STBHD"

    _internal_input_permutation: Optional[tuple[int, int, int, int, int]] = None
    _internal_recurrent_permutation: Optional[tuple[int, int, int, int]] = None
    _internal_bias_permutation: Optional[tuple[int, int, int]] = None
    _internal_output_permutation: Optional[tuple[int, int, int, int, int]] = None
    _internal_output_backward_permutation: Optional[tuple[int, int, int, int]] = None

    # this is moved to slstm
    # backend: str = "vanilla"
    dtype: DTYPES = "bfloat16"
    dtype_acc: Optional[DTYPES] = "float32"
    dtype_b: Optional[DTYPES] = None  # biases
    dtype_r: Optional[DTYPES] = None  # recurrent matrix
    dtype_w: Optional[DTYPES] = None  # inputs / w matrix
    dtype_g: Optional[DTYPES] = None  # gates
    dtype_s: Optional[DTYPES] = None  # states
    dtype_a: Optional[DTYPES] = None  # internal accumulation
    # if this is set to true, the kernel dtype has to match all other dtypes
    # but input dtypes might be arbitrary (are autocasted)
    enable_automatic_mixed_precision: bool = True
    trainable_r: Union[list[bool], bool] = True
    trainable_b: Union[list[bool], bool] = True
    # initial value for each state
    initial_val: Union[float, Sequence[float]] = 0.0

    _internal_dtype_size: int = 2
    _internal_acc_dtype_size: int = 4

    @property
    def input_dim(self):
        return self.num_gates_w * self.hidden_dim

    @property
    def torch_dtype(self) -> torch.dtype:
        return DTYPE_DICT[self.dtype]

    @property
    def torch_dtype_b(self) -> torch.dtype:
        return DTYPE_DICT[self.dtype_b]

    @property
    def torch_dtype_r(self) -> torch.dtype:
        return DTYPE_DICT[self.dtype_r]

    @property
    def torch_dtype_w(self) -> torch.dtype:
        return DTYPE_DICT[self.dtype_w]

    @property
    def torch_dtype_s(self) -> torch.dtype:
        return DTYPE_DICT[self.dtype_s]

    def __post_init__(self):
        if self.dtype_acc != "float32":
            assert self.dtype_acc == "float16" and self.dtype == "float16"

        if self.dtype_acc == "float32":
            self._internal_acc_dtype_size = 4
        else:
            self._internal_acc_dtype_size = 2
        if self.dtype == "float32":
            self._internal_dtype_size = 4
        else:
            self._internal_dtype_size = 2

        if self.num_heads <= 0 and self.head_dim <= 0:
            self.num_heads = 1
        if self.num_heads <= 0:
            self.num_heads = self.hidden_dim // self.head_dim
        elif self.head_dim <= 0:
            self.head_dim = self.hidden_dim // self.num_heads
        if self.hidden_dim <= 0:
            self.hidden_dim = self.num_heads * self.head_dim
        if self.num_gates_t < 0:
            self.num_gates_t = self.num_gates_r
        if self.dtype_b is None:
            self.dtype_b = self.dtype
        if self.dtype_a is None:
            self.dtype_a = self.dtype_b
        if self.dtype_r is None:
            self.dtype_r = self.dtype
        if self.dtype_w is None:
            self.dtype_w = self.dtype
        if self.dtype_s is None:
            self.dtype_s = self.dtype_w
        if self.dtype_g is None:
            self.dtype_g = self.dtype_r

        assert (
            self.function in rnn_function_registry
        ), f"RNN function {self.function} not in registry"
        (
            self.num_gates_r,
            self.num_gates_w,
            self.num_gates_i,
            self.num_gates_t,
            self.num_states,
        ) = rnn_function_registry[self.function]["gates_states"]
        # TODO fix this by padding the recurrent matrix
        if self.function == "gru" and self.backend == "cuda_fused":
            LOGGER.info(
                "Fixing cuda_fused to cuda kernel as recurrent gates do not divide 8 which is problematic for kernels"
            )
            self.backend = "cuda"
        if self.constants is None:
            self.constants = rnn_function_registry[self.function]["constants"]
        self.simple_agg = rnn_function_registry[self.function]["simple_agg"]
        if "initial_val" in rnn_function_registry[self.function]:
            self.initial_val = rnn_function_registry[self.function]["initial_val"]

        if self.backend == "vanilla":
            self._internal_input_shape = "TBGHD"
            self._internal_bias_shape = "GHD"
            self._internal_recurrent_shape = "HPGD"
            self._internal_output_shape = "TBSHD"
        elif self.backend == "vanilla_fwbw":
            self._internal_input_shape = "BTGHD"
            self._internal_recurrent_shape = "GHDP"
            self._internal_bias_shape = (
                "GHD"  # TODO should be HGD but permute does not work
            )
            self._internal_output_shape = "TSBHD"
        elif self.backend == "cuda":
            self._internal_input_shape = "TBHGD"
            self._internal_recurrent_shape = "HPGD"
            self._internal_bias_shape = "HGD"
            self._internal_output_shape = "STBHD"
        elif self.backend == "cuda_fused":
            self._internal_input_shape = "TBHDG"
            self._internal_recurrent_shape = "HDGP"
            self._internal_bias_shape = "HDG"
            self._internal_output_shape = "STBHD"
            if self.dtype == "float32":
                self.batch_size = 16
        elif self.backend == "triton_fused":
            self._internal_input_shape = "BTGHD"
            self._internal_recurrent_shape = "GHDP"
            self._internal_bias_shape = (
                "GHD"  # TODO should be HGD but permute does not work
            )
            self._internal_output_shape = "TSBHD"

        self._internal_input_permutation = permute_to(
            self.input_shape, self._internal_input_shape
        )
        self._internal_output_permutation = permute_to(
            self._internal_output_shape, self.output_shape
        )
        self._internal_recurrent_permutation = permute_to(
            self.recurrent_shape, self._internal_recurrent_shape
        )
        self._internal_bias_permutation = permute_to(
            self.bias_shape, self._internal_bias_shape
        )
        self._internal_output_backward_permutation = permute_to(
            self.output_shape, self._internal_output_shape
        )

    @property
    def defines(self):
        return (
            [
                f"-DFLASHRNN_HIDDEN_SIZE={self.hidden_dim}",
                f"-DFLASHRNN_BATCH_SIZE={self.batch_size}",
                f"-DFLASHRNN_NUM_HEADS={self.num_heads}",
                f"-DFLASHRNN_NUM_STATES={self.num_states}",
                f"-DFLASHRNN_DTYPE={_python_dtype_to_cuda_dtype[self.dtype]}",
                f"-DFLASHRNN_DTYPE_B={_python_dtype_to_cuda_dtype[self.dtype_b]}",
                f"-DFLASHRNN_DTYPE_R={_python_dtype_to_cuda_dtype[self.dtype_r]}",
                f"-DFLASHRNN_DTYPE_W={_python_dtype_to_cuda_dtype[self.dtype_w]}",
                f"-DFLASHRNN_DTYPE_G={_python_dtype_to_cuda_dtype[self.dtype_g]}",
                f"-DFLASHRNN_DTYPE_S={_python_dtype_to_cuda_dtype[self.dtype_s]}",
                f"-DFLASHRNN_DTYPE_A={_python_dtype_to_cuda_dtype[self.dtype_a]}",
                f"-DFLASHRNN_NUM_GATES_R={self.num_gates_r}",
                f"-DFLASHRNN_NUM_GATES_W={self.num_gates_w}",
                f"-DFLASHRNN_NUM_GATES_I={self.num_gates_i}",
                f"-DFLASHRNN_NUM_GATES_T={self.num_gates_t}",
                f"-DFLASHRNN_SIMPLE_AGG={'true' if self.simple_agg else 'false'}",
            ]
            + [f"-DFLASHRNN_USE_DTYPE_{self.dtype.upper()}=1"]
            + (
                [
                    "-DFLASHRNN_GRADIENT_RECURRENT_CLIPVAL_VALID=true",
                    f"-DFLASHRNN_GRADIENT_RECURRENT_CLIPVAL={self.gradient_recurrent_clipval}",
                ]
                if self.gradient_recurrent_clipval is not None
                else [
                    "-DFLASHRNN_GRADIENT_RECURRENT_CLIPVAL_VALID=false",
                    "-DFLASHRNN_GRADIENT_RECURRENT_CLIPVAL=0.0",
                ]
            )
            + (
                [
                    "-DFLASHRNN_FORWARD_CLIPVAL_VALID=true",
                    f"-DFLASHRNN_FORWARD_CLIPVAL={self.gradient_recurrent_clipval}",
                ]
                if self.gradient_recurrent_clipval is not None
                else [
                    "-DFLASHRNN_FORWARD_CLIPVAL_VALID=false",
                    "-DFLASHRNN_FORWARD_CLIPVAL=0.0",
                ]
            )
        )


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
            # config = (
            #     config_list[i]
            #     if config_list
            #     else _get_config(
            #         Wx_list[i],
            #         R_list[i],
            #         b_list[i],
            #         function="lstm",
            #         backend="cuda_fused",
            #         dtype=dtype_str,
            #     )
            # )

            self.layers.append(
                _FlashRNNTritonFusedLayer(Wx_list[i], R_list[i], b_list[i], config)
            )

        # self.config = self.layers[0].config  # use first layer's config
        self.config = config  # use first layer's config

    def forward(self, states=None):
        if states is None:
            states = _zero_state(self.config, self.layers[0].Wx)
        # states = _permute_output_backward(self.config, states)

        out = states
        h = None
        last_h = None
        for layer in self.layers:
            # 可能存在问题 是传 h 还是 last_h
            h, last_h = layer(out)
            out = last_h
        return h, last_h


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
