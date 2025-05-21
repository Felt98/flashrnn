import os
import torch
from tqdm import tqdm

import sys

sys.path.append("..")

from models.gru_fused import GRUFused


os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

# logfile = setup_logger()


device = "cuda"
dtype = torch.bfloat16
B = 16  # batch size
T = 512  # sequence length
NG = 3  # number of gates (NGI == NGR)
NH = 12  # number of heads
DH = 64  # head hidden (embedding) dimension
H = NH * DH  # head dimension
# NS = 1  # number of states (h)
LAYER = 2  # number of layers
requires_grad = True
###
ITERS = 10

# Create GRUFused
gru = GRUFused(128, H, NH, LAYER).to(device=device, dtype=dtype)

# Create inputs
h0 = torch.randn(LAYER, B, H, device=device, requires_grad=requires_grad)
input = torch.randn(
    [B, T, 128],
    device=device,
    dtype=dtype,
    requires_grad=requires_grad,
)
for _ in tqdm(range(ITERS), desc="Test - GRUFused multi layers", file=sys.stdout):
    h_frnn, hlast_frnn = gru(input, h0)
    print("h_frnn.shape:", h_frnn.shape)
    h_frnn[0].sum().backward()
