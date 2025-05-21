# Copyright 2024 NXAI GmbH
# Korbinian Poeppel


from ..config import (
    FlashRNNConfig,
    DTYPE_DICT_REV,
)
import torch
from torch import nn
from .gru_build import GRUFuncGenerator


class GRUCuda(nn.Module):
    def __init__(
        self,
        input_size,
        hidden_size,
        num_heads=1,
        num_layers=1,
    ):
        """
        Initialize the parameters of the GRU Alternating layer.

        Arguments:
        input_size: int, the feature dimension of the input.
        hidden_size: int, the feature dimension of the output.
            input_size (int): The number of expected features in the input (input feature dimension).
            hidden_size (int): The number of features in the hidden state (output feature dimension).
            num_heads (int, optional): Number of hidden state heads .
            num_layers (int, optional): Number of stacked LSTM layers. Default is 1.
        """
        super().__init__()
        assert (
            hidden_size % num_heads == 0
        ), f"hidden_size {hidden_size} must be divisible by num_heads {num_heads}"
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.head_dim = hidden_size // num_heads
        self.num_gates = 3
        self.linear = nn.Linear(input_size, self.num_gates * hidden_size)
        self.linear_Wx = nn.Linear(self.head_dim, self.num_gates * self.head_dim)

        self.config = FlashRNNConfig(
            head_dim=self.head_dim,
            num_heads=self.num_heads,
            function="gru",
            num_states=1,
            backend="cuda",
        )

        # 为每一层分配参数R、b
        self.recurrents = nn.ParameterList()
        self.biases = nn.ParameterList()
        for layer in range(num_layers):
            R = nn.Parameter(
                torch.empty(num_heads, self.head_dim, self.num_gates, self.head_dim)
            )
            b = nn.Parameter(torch.empty(num_heads, self.num_gates, self.head_dim))
            self.recurrents.append(R)
            self.biases.append(b)

        self.reset_parameters()

    def reset_parameters(self):
        for R, b in zip(self.recurrents, self.biases):
            nn.init.normal_(R, mean=0.0, std=1.0 / self.head_dim**0.5)
            nn.init.normal_(b, mean=0.0, std=1.0)

    def forward(self, input, states=None):
        batch_size = input.shape[0]
        seq_len = input.shape[1]
        Wx = self.linear(input)
        Wx = Wx.reshape(
            batch_size,
            seq_len,
            self.num_gates,
            self.num_heads,
            self.head_dim,
        )
        Wx = Wx.permute(1, 0, 3, 2, 4)  # [T,B,NH,NG,D]

        self._set_config_type(input.dtype)
        self.config.batch_size = batch_size

        if states is None or states.size(0) != self.num_layers:
            # print("=============new==========")
            h_0 = input.new_zeros(
                self.num_layers,
                batch_size,
                self.hidden_size,
                requires_grad=True,
            )

        else:
            h_0 = states

        output = None
        hn_list = []
        last_h = None

        for layer in range(self.num_layers):
            recurrent = self.recurrents[layer]
            recurrent = recurrent.permute(0, 3, 2, 1)
            bias = self.biases[layer]

            hx = h_0[layer].reshape(batch_size, self.num_heads, self.head_dim)
            state = torch.stack([hx], dim=0).unsqueeze(1)  # [1, 1, B, NH, D]

            output = self._impl(Wx, recurrent, bias, state)  # [1,T+1,B,NH,D]

            last_h = output[:, -1:].permute(
                0, 2, 1, 3, 4
            )  # [NS, 1, B, NH, D] ->[NS, B, 1, NH, D]
            h = output[:, 1:].permute(
                0, 2, 1, 3, 4
            )  # [NS, T, B, NH, D] ->[NS, B, T, NH, D]
            s_out_last = last_h[0].squeeze(1)  # [B, NH, D]
            h = h[0]

            Wx = h.permute(1, 0, 2, 3)  # [T,B,NH,D] 需要扩充为 [...,D,4]
            Wx = self._reshape_Wx(Wx)

            hn_list.append(s_out_last)  # T-1 时刻的 hidden 状态

        h = h.reshape(batch_size, seq_len, self.hidden_size)
        hn = torch.stack(hn_list).reshape(self.num_layers, batch_size, self.hidden_size)
        return h, hn

    def _impl(self, Wx, R, b, state):
        kernel = GRUFuncGenerator(torch.is_grad_enabled(), self.config)

        state = kernel.apply(
            torch.is_grad_enabled(),
            Wx.contiguous(),
            state[:, 0].contiguous(),
            R.contiguous(),
            b.contiguous(),
        )

        return state

    def _reshape_Wx(self, Wx):
        seq_len, batch_size, _, _ = Wx.shape
        Wx = Wx.reshape(-1, self.head_dim)
        Wx = self.linear_Wx(Wx)
        return Wx.view(
            seq_len, batch_size, self.num_heads, self.num_gates, self.head_dim
        )

    def _set_config_type(self, dtype):
        self.config.dtype = self.config.dtype_w = self.config.dtype_r = (
            self.config.dtype_s
        ) = self.config.dtype_b = self.config.dtype_g = DTYPE_DICT_REV[dtype]
