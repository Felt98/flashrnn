import torch
from torch import nn
import sys

sys.path.append("..")

from .lstm_build import LSTMFuncGeneratorFused
from ..config import (
    FlashRNNConfig,
    DTYPE_DICT_REV,
)


class LSTMFused(nn.Module):
    def __init__(
        self,
        input_size,
        hidden_size,
        num_heads=1,
        num_layers=1,
    ):
        """
        Initialize the parameters of the LSTM Fused layer.

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
        self.num_gates = 4
        self.linear = torch.nn.Linear(input_size, self.num_gates * hidden_size)
        self.linear_Wx = nn.Linear(self.head_dim, self.num_gates * self.head_dim)

        self.config = FlashRNNConfig(
            head_dim=self.head_dim,
            num_heads=self.num_heads,
            function="lstm",
            num_states=2,
            backend="cuda_fused",
        )

        # 为每一层分配参数R、b
        self.recurrents = nn.ParameterList()
        self.biases = nn.ParameterList()
        for layer in range(num_layers):
            R = nn.Parameter(
                torch.empty(num_heads, self.head_dim, self.num_gates, self.head_dim)
            )
            b = nn.Parameter(torch.empty(num_heads, self.head_dim, self.num_gates))
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

        # Wx不能直接reshape成目标形状 ，需要先reshape成下面排布，再permute
        Wx = Wx.reshape(
            batch_size,
            seq_len,
            self.num_gates,
            self.num_heads,
            self.head_dim,
        )
        Wx = Wx.permute(1, 0, 3, 4, 2)

        self._set_config_type(input.dtype)
        self.config.batch_size = batch_size

        if states is None or states[0].size(0) != self.num_layers:
            h_0 = input.new_zeros(self.num_layers, batch_size, self.hidden_size)
            c_0 = input.new_zeros(self.num_layers, batch_size, self.hidden_size)
        else:
            h_0, c_0 = states
        output = None
        hn_list, cn_list = [], []
        for layer in range(self.num_layers):
            recurrent = self.recurrents[layer]
            bias = self.biases[layer]

            hx = h_0[layer].reshape(batch_size, self.num_heads, self.head_dim)
            cx = c_0[layer].reshape(batch_size, self.num_heads, self.head_dim)
            state = torch.stack([hx, cx], dim=0).unsqueeze(1)

            # output shape: [NS,T+1,B,NH,D]
            output = self._impl(Wx, recurrent, bias, state)
            output = output.permute(
                0, 2, 1, 3, 4
            )  # [NS, T, B, NH, D] ->[NS, B, T, NH, D]

            last_h = output[:, :, -1:]
            h = output[0, :, 1:]
            s_out_last = last_h[0].squeeze(1)  # [B, NH, D]
            c_out_last = last_h[1].squeeze(1)  # [B, NH, D]
            assert s_out_last.shape == (batch_size, self.num_heads, self.head_dim)

            Wx = h.permute(1, 0, 2, 3)  # [T,B,NH,D] 需要扩充为 [...,D,4]
            Wx = self._reshape_Wx(Wx)

            hn_list.append(s_out_last)  # T-1 时刻的 hidden 状态
            cn_list.append(c_out_last)  # T-1 时刻的 cell 状态

        h = h.reshape(batch_size, seq_len, self.hidden_size)
        hn = torch.stack(hn_list).reshape(self.num_layers, batch_size, self.hidden_size)
        cn = torch.stack(cn_list).reshape(self.num_layers, batch_size, self.hidden_size)

        return h, (hn, cn)

    def _impl(self, Wx, R, b, state):
        kernel = LSTMFuncGeneratorFused(torch.is_grad_enabled(), self.config)

        states = kernel.apply(
            torch.is_grad_enabled(),
            Wx.contiguous(),
            state[:, 0].contiguous(),
            R.contiguous(),
            b.contiguous(),
        )
        return states

    def _reshape_Wx(self, Wx):
        seq_len, batch_size, _, _ = Wx.shape
        Wx = Wx.reshape(-1, self.head_dim)
        Wx = self.linear_Wx(Wx)
        return Wx.view(
            seq_len, batch_size, self.num_heads, self.head_dim, self.num_gates
        )

    def _set_config_type(self, dtype):

        self.config.dtype = self.config.dtype_w = self.config.dtype_r = (
            self.config.dtype_s
        ) = self.config.dtype_b = self.config.dtype_g = DTYPE_DICT_REV[dtype]
