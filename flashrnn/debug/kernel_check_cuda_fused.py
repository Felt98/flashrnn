import os
import torch
from tqdm import tqdm

import sys

sys.path.append("..")
from flashrnn.flashrnn_fused import (
    flashrnn_fused,
    _FlashRNNCudaFusedLayer,
    build_flashrnn_stack,
)
from flashrnn.flashrnn_fused import _get_config
from flashrnn.flashrnn_multi_layer import flashrnn_multi_layer, generate_parameter_list

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"


os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

# logfile = setup_logger()


device = "cuda"
dtype = torch.bfloat16
dtype_str = "bfloat16"
B = 64  # batch size
T = 1024  # sequence length
NG = 3  # number of gates (NGI == NGR)
NH = 12  # number of heads
DH = 32  # input/hidden (embedding) dimension gru需要按16对齐
# NS = 1  # number of states (c, h)
LAYER = 8
requires_grad = True
###
ITERS = 3
function = "gru"

# rnn = _FlashRNNCudaFusedLayer(
#     Wx=Wx_mtr, R=R_mtr, b=b_mtr, config=config, dtype="bfloat16"
# )
# print(config.defines)
# for _ in tqdm(range(WARMUP_ITERS), desc="Warmup - CUDA fused", file=sys.stdout):
#     out = rnn()[0][0].sum().backward()  # 默认 zero_state

# for _ in tqdm(range(ITERS), desc="Test - CUDA fused", file=sys.stdout):
#     out = rnn()[0][0].sum().backward()

Wx_list, R_list, x_list, b_list = generate_parameter_list(
    B,
    T,
    NG,
    NH,
    DH,
    LAYER,
    device=device,
    dtype=dtype,
    requires_grad=requires_grad,
)
config = _get_config(
    Wx_list[0], R_list[0], b_list[0], function, "cuda_fused", dtype=dtype_str
)

rnn = build_flashrnn_stack(
    B,
    T,
    NG,
    NH,
    DH,
    num_layers=LAYER,
    config=config,
    function=function,
)
# for _ in tqdm(range(ITERS), desc="Test - CUDA fused multi layers", file=sys.stdout):
#     h_frnn, hlast_frnn, _ = rnn()
#     print("h_frnn.shape:", h_frnn.shape)
#     h_frnn[0].sum().backward()

for _ in tqdm(range(ITERS), desc="Test - CUDA fused multi layers GRU", file=sys.stdout):
    h_frnn, hlast_frnn = flashrnn_multi_layer(
        Wx_list,
        R_list,
        b_list,
        num_layers=LAYER,
        states=None,
        function=config.function,
        backend=config.backend,
        dtype=dtype_str,
    )
    h_frnn[0].sum().backward()
# for _ in tqdm(
#     range(ITERS), desc="Test - CUDA fused multi layers LSTM", file=sys.stdout
# ):
#     h_frnn, hlast_frnn = flashrnn_multi_layer(
#         Wx_list,
#         R_list,
#         b_list,
#         num_layers=LAYER,
#         states=None,
#         function=config.function,
#         backend=config.backend,
#         dtype=dtype_str,
#     )
#     h_frnn[0].sum().backward()
