import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional, Sequence, Union

import torch

LOGGER = logging.getLogger(__name__)

_flashrnn_function_to_num_states = {
    "gru": 1,
    "lstm": 2,
    "slstm": 2,
}

DTYPE_DICT = {
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
    "float32": torch.float32,
    "float64": torch.float64,
}

# str到torch精度到字典
DTYPE_DICT_REV = {
    torch.bfloat16: "bfloat16",
    torch.float: "float32",
    torch.float16: "float16",
    torch.float64: "float64",
}

DTYPES = Literal["bfloat16", "float16", "float32"]


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
    layer_id: int = 0
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

    # B = batch, T time/sequence dim, P num heads, S state dimension, P previous D dimension (R matrix)
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
        # if self.function == "gru" and self.backend == "cuda_fused":
        #     LOGGER.info(
        #         "Fixing cuda_fused to cuda kernel as recurrent gates do not divide 8 which is problematic for kernels"
        #     )
        #     self.backend = "cuda"
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
            self._internal_recurrent_shape = (
                "HDGP"  # 特别注意，虽然D=P，但是还是要permute交换维度
            )
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
        num_states=_flashrnn_function_to_num_states[function],
        backend=backend,
        dtype=dtype if dtype is not None else "bfloat16",
        dtype_w=DTYPE_DICT_REV[Wx.dtype],
        dtype_r=DTYPE_DICT_REV[R.dtype],
        dtype_b=DTYPE_DICT_REV[b.dtype],
    )
