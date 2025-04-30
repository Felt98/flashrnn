import os
import torch
from tqdm import tqdm

# from logutils.logger import setup_logger
import logging
import sys

sys.path.append("..")
from flashrnn_fused import flashrnn_fused, _FlashRNNCudaFusedLayer, build_flashrnn_stack
from flashrnn_fused import _get_config


os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

# logfile = setup_logger()


device = "cuda"
dtype = torch.float32
TGT_DTYPE = torch.bfloat16
B = 64  # batch size
T = 1024  # sequence length
NG = 4  # number of gates (NGI == NGR)
NH = 1  # number of heads
D = 1024  # input/hidden (embedding) dimension
NS = 2  # number of states (c, h)

###
WARMUP_ITERS = 1
ITERS = 30

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


config = _get_config(Wx_mtr, R_mtr, b_mtr, "lstm", "cuda_fused", dtype="bfloat16")
config.batch_size = 16
# rnn = _FlashRNNCudaFusedLayer(
#     Wx=Wx_mtr, R=R_mtr, b=b_mtr, config=config, dtype="bfloat16"
# )
# print(config.defines)
# for _ in tqdm(range(WARMUP_ITERS), desc="Warmup - CUDA fused", file=sys.stdout):
#     out = rnn()[0][0].sum().backward()  # 默认 zero_state

# for _ in tqdm(range(ITERS), desc="Test - CUDA fused", file=sys.stdout):
#     out = rnn()[0][0].sum().backward()


rnn = build_flashrnn_stack(B, T, NG, NH, D, num_layers=8, config=config)
for _ in tqdm(range(ITERS), desc="Test - CUDA fused multi layers", file=sys.stdout):
    out = rnn()[0][0].sum().backward()
