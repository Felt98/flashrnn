import os
import torch
from tqdm import tqdm
import sys

sys.path.append("..")
from models.lstm_cuda import LSTMCuda


os.environ["CUDA_LAUNCH_BLOCKING"] = "1"


device = "cuda"
dtype = torch.bfloat16
B = 16  # batch size
T = 512  # sequence length
NG = 4  # number of gates (NGI == NGR)
NH = 12  # number of heads
DH = 64  # head hidden (embedding) dimension gru fused需要按16对齐
H = NH * DH  # head dimension
# NS = 2  # number of states (c, h)
LAYER = 8  # number of layers
requires_grad = True
ITERS = 10

# Create LSTMCuda
lstm = LSTMCuda(128, H, NH, LAYER).to(device=device, dtype=dtype)

# Create inputs
h0 = torch.randn(LAYER, B, H, device=device, requires_grad=requires_grad)
c0 = torch.randn(LAYER, B, H, device=device, requires_grad=requires_grad)

h0_ref = h0.detach().clone().requires_grad_()
c0_ref = c0.detach().clone().requires_grad_()
input = torch.randn(
    [B, T, 128],
    device=device,
    dtype=dtype,
    requires_grad=requires_grad,
)

for _ in tqdm(range(ITERS), desc="Test - LSTMCuda multi layers", file=sys.stdout):
    out_my, (hn_my, cn_my) = lstm(input, (h0_ref, c0_ref))
    print("h_frnn.shape:", out_my.shape)
    out_my[0].sum().backward()
