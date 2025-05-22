import os
import torch
from tqdm import tqdm

import sys

sys.path.append("..")
from flashrnn.flashrnn import flashrnn
from flashrnn.flashrnn import _get_config


os.environ["CUDA_LAUNCH_BLOCKING"] = "1"


device = "cuda"
dtype = torch.float32
TGT_DTYPE = torch.bfloat16
B = 16  # batch size
T = 32  # sequence length
NG = 3  # number of gates (NGI == NGR)
NH = 1  # number of heads
D = 768  # input/hidden (embedding) dimension
NS = 1  # number of states (c, h)
function = "gru"
###
WARMUP_ITERS = 1
ITERS = 4

Wx = torch.randn([B, T, NG, NH, D], device=device, dtype=dtype)
R = torch.randn([NG, NH, D, D], device=device, dtype=dtype)
b = torch.randn([NG, NH, D], device=device, dtype=dtype)
states_initial = torch.randn([NS, B, 1, NH, D], device=device, dtype=dtype)

Wx_mpt = Wx.clone().to(TGT_DTYPE).detach().requires_grad_(False)
R_mpt = R.clone().to(TGT_DTYPE).detach().requires_grad_(False)
b_mpt = b.clone().to(TGT_DTYPE).detach().requires_grad_(False)
states_initial_mpt = states_initial.clone().to(TGT_DTYPE).detach().requires_grad_(False)

Wx_mtr = Wx.clone().to(TGT_DTYPE).detach().requires_grad_(True)
R_mtr = R.clone().to(TGT_DTYPE).detach().requires_grad_(True)
b_mtr = b.clone().to(TGT_DTYPE).detach().requires_grad_(True)
states_initial_mtr = states_initial.clone().to(TGT_DTYPE).detach().requires_grad_(True)


config = _get_config(Wx_mtr, R_mtr, b_mtr, function, "cuda", dtype="bfloat16")
config.batch_size = 16
print(config.defines)
for _ in tqdm(range(WARMUP_ITERS), desc="Warmup - CUDA alternating"):
    out = (
        flashrnn(
            Wx=Wx_mtr,
            R=R_mtr,
            b=b_mtr,
            states=states_initial_mtr,
            function=function,
            dtype="bfloat16",
            backend="cuda",
            config=config,
        )[0][0]
        .sum()
        .backward()
    )
    print("Wx.grad:", Wx.grad)
    print("states_initial_mtr.grad:", states_initial_mtr.grad)

# for _ in tqdm(range(ITERS), desc="Warmup - CUDA alternating"):
#     out = (
#         flashrnn(
#             Wx=Wx_mtr,
#             R=R_mtr,
#             b=b_mtr,
#             function=function,
#             dtype="bfloat16",
#             backend="cuda",
#             config=config,
#         )[0][0]
#         .sum()
#         .backward()
#     )
