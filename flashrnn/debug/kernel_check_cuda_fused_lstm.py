import os
import torch
from tqdm import tqdm

import sys

sys.path.append("..")
sys.path.append(os.path.abspath(os.path.join(__file__, "../../..")))
# from flashrnn.flashrnn_fused import (
#     flashrnn_fused,
#     _FlashRNNCudaFusedLayer,
#     build_flashrnn_stack,
# )
from flashrnn.flashrnn_fused import _get_config
from flashrnn.frameworks.cuda_fused.lstm import LSTMFused

from flashrnn.flashrnn_multi_layer import generate_parameter_list


os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

# logfile = setup_logger()


device = "cuda"
dtype = torch.float32
dtype_str = "float32"
B = 16  # batch size
T = 512  # sequence length
NG = 3  # number of gates (NGI == NGR)
NH = 1  # number of heads
DH = 64  # input/hidden (embedding) dimension
H = NH * DH
# NS = 1  # number of states (c, h)
LAYER = 8
requires_grad = True
###
ITERS = 10

lstm = LSTMFused(128, H, NH, LAYER).to(device=device, dtype=dtype)

h0 = torch.randn(LAYER, B, H, device="cuda", requires_grad=True)
c0 = torch.randn(LAYER, B, H, device="cuda", requires_grad=True)

# Clone inputs for reference
h0_ref = h0.detach().clone().requires_grad_()
c0_ref = c0.detach().clone().requires_grad_()
input = torch.randn(
    [B, T, 128],
    device=device,
    dtype=dtype,
    requires_grad=requires_grad,
)

for _ in tqdm(range(ITERS), desc="Test - CUDA fused multi layers", file=sys.stdout):
    h_frnn, (hlast_frnn, _) = lstm(input, (h0_ref, c0_ref))
    print("h_frnn.shape:", h_frnn.shape)
    h_frnn[0].sum().backward()

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

# for _ in tqdm(range(ITERS), desc="Test - CUDA fused multi layers GRU", file=sys.stdout):
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
