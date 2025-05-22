import os
import torch
from tqdm import tqdm

import sys

sys.path.append("..")
sys.path.append(os.path.abspath(os.path.join(__file__, "../../..")))
from flashrnn.flashrnn_fused import (
    flashrnn_fused,
    _FlashRNNCudaFusedLayer,
    build_flashrnn_stack,
)
from flashrnn.config import _get_config

from flashrnn.frameworks.cuda_alternating.lstm import LSTMCuda

from flashrnn.flashrnn_multi_layer import generate_parameter_list, flashrnn_multi_layer


os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

# logfile = setup_logger()


device = "cuda"
dtype = torch.float32
dtype_str = "float32"
B = 16  # batch size
T = 512  # sequence length
NG = 3  # number of gates (NGI == NGR)
NH = 12  # number of heads
DH = 32  # input/hidden (embedding) dimension gru需要按16对齐
H = NH * DH
# NS = 1  # number of states (c, h)
LAYER = 8
requires_grad = True
###
ITERS = 10


lstm = LSTMCuda(128, H, NH, LAYER).to(device=device, dtype=dtype)
input = torch.randn(
    [B, T, 128],
    device=device,
    dtype=dtype,
    requires_grad=requires_grad,
)
for _ in tqdm(range(ITERS), desc="Test - CUDA multi layers", file=sys.stdout):
    out_my, (hn_my, cn_my) = lstm(input)
    print("h_frnn.shape:", out_my.shape)
    out_my[0].sum().backward()

# rnn = build_gru_stack(
#     B,
#     T,
#     NG,
#     NH,
#     DH,
#     num_layers=LAYER,
#     config=config,
#     function=function,
# )
# for _ in tqdm(range(ITERS), desc="Test - CUDA fused multi layers", file=sys.stdout):
#     h_frnn, hlast_frnn, _ = rnn()
#     print("h_frnn.shape:", h_frnn.shape)
#     h_frnn[0].sum().backward()

# for _ in tqdm(range(ITERS), desc="Test - CUDA multi layers GRU", file=sys.stdout):
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
