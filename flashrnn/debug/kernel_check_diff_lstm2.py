import os
import torch
from torch import nn
from tqdm import tqdm

import sys

sys.path.append("..")
sys.path.append(os.path.abspath(os.path.join(__file__, "../../..")))


from flashrnn.frameworks.cuda_alternating.lstm import LSTMCuda
from flashrnn.frameworks.cuda_fused.lstm import LSTMFused
import pandas as pd


os.environ["CUDA_LAUNCH_BLOCKING"] = "1"


device = "cuda"

dtype = torch.bfloat16
dtype_str = "bfloat16"
###
# Config
input_size = 512
hidden_size = 512
batch = 64
num_layers = 1
num_head = 1
num_gate = 4
seq_len = 1
requires_grad = True
csvfilename = f"diff_lstm_cuda&fused_{dtype_str}_T{seq_len}_H{hidden_size}_B{batch}.csv"

# lstm = GRUCuda(input_size, H, NH, LAYER).to(device=device, dtype=dtype)
# input = torch.randn(
#     [B, T, input_size],
#     device=device,
#     dtype=dtype,
#     requires_grad=requires_grad,
# )


def initialize_ref_lstm_constant(ref_lstm, value=1.0):
    with torch.no_grad():
        for name, param in ref_lstm.named_parameters():
            param.fill_(value)


def sync_from_pytorch_lstm(my_lstm, ref_lstm, fused: bool):
    """
    同步 nn.LSTM 的第一层权重到自定义的 LSTMFused。
    要求：
    - my_lstm.num_heads == 1
    - my_lstm.num_layers == 1
    - ref_lstm.num_layers == 1，单向
    """
    assert my_lstm.num_heads == 1, "只能同步 num_heads == 1 的模型"
    assert my_lstm.num_layers == 1, "只能同步单层模型"
    assert ref_lstm.num_layers == 1, "只支持同步单层单向 LSTM"

    H = my_lstm.hidden_size
    I = my_lstm.linear.in_features  # 输入维度

    with torch.no_grad():
        ref_lstm.biases[0].zero_()
        ref_lstm.linear.bias.zero_()
        # ========== 1. 同步 Linear 权重 ==========
        # ref: weight_ih_l0: [4H, I]
        my_lstm.linear.weight.copy_(ref_lstm.linear.weight)  # [4H, I]
        my_lstm.linear.bias.copy_(ref_lstm.linear.bias)  # [4H]

        # ========== 2. 同步 Recurrent 权重 R ==========
        weight_hh = ref_lstm.recurrents[0]  # shape [NH,P,NG,D]
        # gates = torch.split(weight_hh, H, dim=0)  # 4 tensors of shape [H, H]
        # stacked = torch.stack(gates, dim=0)  # [4, H, H]
        # R = stacked.unsqueeze(0).permute(0, 2, 1, 3).contiguous()  # [1, H, 4, H]
        R = weight_hh.permute(0, 3, 2, 1)  # shape [NH,D,NG,P]
        my_lstm.recurrents[0].copy_(R)

        # ========== 3. 同步 bias ==========
        total_bias = ref_lstm.biases[0]  # shape [4H]
        # gates_b = torch.split(total_bias, H, dim=0)  # 4 tensors of shape [H]
        b_stacked = total_bias.permute(0, 2, 1)
        my_lstm.biases[0].copy_(b_stacked)

        # # ========== 验证是否同步成功 ==========
        # [4H, I]
        diff_w = (my_lstm.linear.weight - ref_lstm.linear.weight).abs().max()
        print(f"[Check] Linear weight max abs diff: {diff_w:.2e}")

        # [4H]
        diff_b = (my_lstm.linear.bias - ref_lstm.linear.bias).abs().max()
        print(f"[Check] Linear bias   max abs diff: {diff_b:.2e}")

        # [1, H, 4, H] -> [4, H, H]
        R = my_lstm.recurrents[0].permute(0, 3, 2, 1)
        diff_R = (R - ref_lstm.recurrents[0]).abs().max()
        print(f"[Check] Recurrent weight max abs diff: {diff_R:.2e}")

        # [1, H, 4] -> [4H]
        b_my = my_lstm.biases[0].permute(0, 2, 1)
        b_ref = ref_lstm.biases[0]
        diff_bias = (b_my - b_ref).abs().max()
        print(f"[Check] Bias max abs diff: {diff_bias:.2e}")
    print("[✓] LSTMFused 参数成功同步。")


def sync_grads_R_b(my_lstm, ref_lstm, fused, layer=0):

    H = my_lstm.hidden_size
    weight_hh_grad = ref_lstm.recurrents[0]  # [4H, H]
    if weight_hh_grad is None:
        print(f"[R] ref_lstm.recurrents[0].grad is None")
        return

    # reshape PyTorch GRU's weight_hh gradient to match custom format
    # gates = torch.split(weight_hh_grad, H, dim=0)  # 4 tensors of shape [H, H]
    # ref_grad_R = torch.stack(gates, dim=0)  # [4, H, H]
    ref_grad_R = weight_hh_grad.permute(0, 3, 2, 1)  # shape [NH,D,NG,P]

    my_grad_R = my_lstm.recurrents[0].grad  # [1, H, 4, H]
    if my_grad_R is None:
        print(f"[R] my_lstm.recurrents[{0}].grad is None")
        return
    if my_grad_R.shape != ref_grad_R.shape:
        print(f"[b] Shape mismatch: my={my_grad_b.shape}, ref={ref_grad_b.shape}")
        return
    # ====== 提取bias 的 grad ======
    bias_grad = ref_lstm.biases[0].grad  # shape: [4H]
    if bias_grad is None:
        print("[b] ref_lstm.bias_hh_l0.grad is None")
        return
    ref_grad_b = bias_grad.permute(0, 2, 1)

    # ====== 提取自定义实现中的 bias.grad ======
    my_grad_b = my_lstm.biases[0].grad  # shape: [1, H, 4] or [1, 4, H]
    if my_grad_b is None:
        print("[b] my_lstm.biases[0].grad is None")

    # ====== 确保两个形状一致 ======
    if my_grad_b.shape != ref_grad_b.shape:
        print(f"[b] Shape mismatch: my={my_grad_b.shape}, ref={ref_grad_b.shape}")
        return
    return my_grad_R, ref_grad_R, my_grad_b, ref_grad_b


def generate_diff_csv(tensor_pairs: dict, filename: str):
    """
    比较多个 tensor 对之间的误差，并写入 CSV。

    参数:
        tensor_pairs: dict[str, Tuple[tensor, tensor]]
            例如 {"x": (out_my, out_ref), "x grad": (x.grad, x_ref.grad)}

        filename: str
            要保存的 CSV 文件名
    """
    # 定义误差函数
    metric_funcs = {
        "max abs": max_abs_diff,
        "mean abs": mean_abs_diff,
        "max ref": max_ref_diff,
        "mean ref": mean_ref_diff,
    }

    records = []
    for metric, func in metric_funcs.items():
        for name, (a, b) in tensor_pairs.items():
            try:
                value = func(a, b)
            except Exception as e:
                value = float("nan")  # 如果出错则写 NaN
                print(f"[Warning] Failed to compute {metric} for '{name}': {e}")
            records.append({"metric": metric, "target": name, "value": value})

    df = pd.DataFrame(records)
    column_order = list(tensor_pairs.keys())
    pivot_df = df.pivot(index="metric", columns="target", values="value")
    pivot_df = pivot_df[column_order]

    pivot_df.to_csv(filename, float_format="%.2e")
    print(f"[Info] Saved diff summary to {filename}")


def max_abs_diff(a, b):
    return (a - b).abs().max().item()


def mean_abs_diff(a, b):
    return (a - b).abs().mean().item()


def max_ref_diff(a, b, eps=1e-8):
    return ((a - b).abs() / (b.abs() + eps)).max().item()


def mean_ref_diff(a, b, eps=1e-8):
    return ((a - b).abs() / (b.abs() + eps)).mean().item()


def print_lstm_all_params(lstm: nn.LSTM):
    with torch.no_grad():
        for name, param in lstm.named_parameters():
            print("name :", name)
            print("param:", param)


def compare_models():

    torch.manual_seed(0)

    # Models
    ref_lstm = LSTMFused(input_size, hidden_size, num_layers).to(
        device=device, dtype=dtype
    )
    my_lstm = LSTMCuda(input_size, hidden_size, num_layers).to(
        device=device, dtype=dtype
    )
    fused = False
    # initialize_ref_lstm_constant(ref_lstm)
    # print_lstm_all_params(ref_lstm)
    sync_from_pytorch_lstm(my_lstm, ref_lstm, fused)  # 同步权重

    # Inputs
    # x = torch.randn(batch, seq_len, input_size, device="cuda", requires_grad=False)
    x = torch.randn(
        batch, seq_len, input_size, device="cuda", requires_grad=True, dtype=dtype
    )
    h0 = torch.zeros(
        num_layers, batch, hidden_size, device="cuda", requires_grad=True, dtype=dtype
    )
    c0 = torch.zeros(
        num_layers, batch, hidden_size, device="cuda", requires_grad=True, dtype=dtype
    )

    # Clone inputs for reference
    x_ref = x.detach().clone().requires_grad_()
    h0_ref = h0.detach().clone().requires_grad_()
    c0_ref = c0.detach().clone().requires_grad_()

    # Forward
    # out_ref: [B,T,H]
    # hn_ref \ cn_ref: [num_layers,B,H]
    out_ref, (hn_ref, cn_ref) = ref_lstm(x_ref, (h0_ref, c0_ref))
    out_my, (hn_my, cn_my) = my_lstm(x, (h0, c0))
    # out_ref, (hn_ref, cn_ref) = ref_lstm(x_ref)
    # out_my, (hn_my, cn_my) = my_lstm(x)
    # print("out my: ", out_my)
    # print("out ref:", out_ref)
    print("out_my shape: ", out_my.shape)
    print("out_ref shape: ", out_ref.shape)
    # Backward
    loss_my = out_my.sum()
    loss_ref = out_ref.sum()
    loss_my.backward()
    loss_ref.backward()

    print("\n[Grad Check] ---------------------")
    tracked = {
        "x": x,
        "h0": h0,
        "c0": c0,
        "out_my": out_my,
        "hn_my": hn_my,
    }

    # 也可以加上 recurrent 权重和 bias
    for i, (R, b) in enumerate(zip(my_lstm.recurrents, my_lstm.biases)):
        tracked[f"R_{i}"] = R
        tracked[f"b_{i}"] = b
    for name, tensor in tracked.items():
        if not isinstance(tensor, torch.Tensor):
            continue
        grad = tensor.grad
        print(
            f"{name:10s}: "
            f"requires_grad={tensor.requires_grad}, "
            f"is_leaf={tensor.is_leaf}, "
            f"grad={'None' if grad is None else grad.shape}"
        )
    # Output comparison
    print(f"[Forward] Output max abs diff: {max_abs_diff(out_my, out_ref):.2e}")
    print(f"[Forward] hn     max abs diff: {max_abs_diff(hn_my, hn_ref):.2e}")
    print(f"[Forward] cn     max abs diff: {max_abs_diff(cn_my, cn_ref):.2e}")
    print(f"[Forward] Output mean abs diff: {mean_abs_diff(out_my, out_ref):.2e}")
    print(f"[Forward] hn     mean abs diff: {mean_abs_diff(hn_my, hn_ref):.2e}")
    print(f"[Forward] cn     mean abs diff: {mean_abs_diff(cn_my, cn_ref):.2e}")
    print(f"[Forward] Output max ref diff: {max_ref_diff(out_my, out_ref):.2e}")
    print(f"[Forward] hn     max ref diff: {max_ref_diff(hn_my, hn_ref):.2e}")
    print(f"[Forward] cn     max ref diff: {max_ref_diff(cn_my, cn_ref):.2e}")
    print(f"[Forward] Output mean ref diff: {mean_ref_diff(out_my, out_ref):.2e}")
    print(f"[Forward] hn     mean ref diff: {mean_ref_diff(hn_my, hn_ref):.2e}")
    print(f"[Forward] cn     mean ref diff: {mean_ref_diff(cn_my, cn_ref):.2e}")

    # Gradients
    # Gradients
    print(
        f"[Grad] Input x     grad max abs diff: {max_abs_diff(x.grad, x_ref.grad):.2e}"
    )
    print(
        f"[Grad] Input x     grad mean abs diff: {mean_abs_diff(x.grad, x_ref.grad):.2e}"
    )
    print(
        f"[Grad] Input x     grad max ref diff: {max_ref_diff(x.grad, x_ref.grad):.2e}"
    )
    print(
        f"[Grad] Input x     grad mean ref diff: {mean_ref_diff(x.grad, x_ref.grad):.2e}"
    )

    print(
        f"[Grad] h0          grad max abs diff: {max_abs_diff(h0.grad, h0_ref.grad):.2e}"
    )
    print(
        f"[Grad] c0          grad max abs diff: {max_abs_diff(c0.grad, c0_ref.grad):.2e}"
    )
    print(
        f"[Grad] h0          grad mean abs diff: {mean_abs_diff(h0.grad, h0_ref.grad):.2e}"
    )
    print(
        f"[Grad] c0          grad mean abs diff: {mean_abs_diff(c0.grad, c0_ref.grad):.2e}"
    )
    print(
        f"[Grad] h0          grad max ref diff: {max_ref_diff(h0.grad, h0_ref.grad):.2e}"
    )
    print(
        f"[Grad] c0          grad max ref diff: {max_ref_diff(c0.grad, c0_ref.grad):.2e}"
    )
    print(
        f"[Grad] h0          grad mean ref diff: {mean_ref_diff(h0.grad, h0_ref.grad):.2e}"
    )
    print(
        f"[Grad] c0          grad mean ref diff: {mean_ref_diff(c0.grad, c0_ref.grad):.2e}"
    )
    my_grad_R, ref_grad_R, my_grad_b, ref_grad_b = sync_grads_R_b(
        my_lstm, ref_lstm, fused, layer=0
    )

    generate_diff_csv(
        {
            "hn": (hn_my, hn_ref),
            "cn": (cn_my, cn_ref),
            "x grad": (x.grad, x_ref.grad),
            "h0 grad": (h0.grad, h0_ref.grad),
            "c0 grad": (c0.grad, c0_ref.grad),
            "R grad": (my_grad_R, ref_grad_R),
            "b grad": (my_grad_b, ref_grad_b),
        },
        csvfilename,
    )
    # for (n1, p1), (n2, p2) in zip(
    #     my_lstm.named_parameters(), ref_lstm.named_parameters()
    # ):
    #     if p1.grad is not None and p2.grad is not None:
    #         diff = max_abs_diff(p1.grad, p2.grad)
    #         print(f"[Grad] Param {n1:20s} grad diff: {diff:.2e}")


if __name__ == "__main__":
    compare_models()
