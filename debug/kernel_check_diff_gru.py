import os
import torch
from torch import nn
from tqdm import tqdm
import pandas as pd

import sys

sys.path.append("..")
sys.path.append(os.path.abspath(os.path.join(__file__, "../../..")))


from flashrnn.frameworks.cuda_alternating.gru import GRUCuda
from flashrnn.frameworks.cuda_fused.gru import GRUFused


os.environ["CUDA_LAUNCH_BLOCKING"] = "1"


device = "cuda"
dtype = torch.float32
dtype_str = "float32"
###
# Config
input_size = 64
hidden_size = 64
seq_len = 1
batch = 64
num_layers = 1
num_head = 1
num_gate = 3
requires_grad = True
csvfilename = f"diff_gru_fused_{dtype_str}_T{seq_len}_H{512}_B{batch}.csv"


# gru = GRUCuda(input_size, H, NH, LAYER).to(device=device, dtype=dtype)
# input = torch.randn(
#     [B, T, input_size],
#     device=device,
#     dtype=dtype,
#     requires_grad=requires_grad,
# )
def initialize_ref_gru_constant(ref_gru, value=1.0):
    with torch.no_grad():
        for name, param in ref_gru.named_parameters():
            param.fill_(value)


def sync_from_pytorch_gru(my_gru: GRUFused, ref_gru: nn.GRU, fused: bool):
    """
    - my_gru.num_heads == 1
    - my_gru.num_layers == 1
    - ref_gru.num_layers == 1，单向
    """
    assert my_gru.num_heads == 1, "只能同步 num_heads == 1 的模型"
    assert my_gru.num_layers == 1, "只能同步单层模型"
    assert ref_gru.num_layers == 1 and not ref_gru.bidirectional

    H = my_gru.hidden_size
    I = my_gru.linear.in_features  # 输入维度

    with torch.no_grad():
        ref_gru.bias_ih_l0.zero_()
        ref_gru.bias_hh_l0.zero_()
        # ========== 1. 同步 Linear 权重 ==========
        # ref: weight_ih_l0: [4H, I]
        my_gru.linear.weight.copy_(ref_gru.weight_ih_l0)  # [4H, I]
        my_gru.linear.bias.copy_(ref_gru.bias_ih_l0)  # [4H]

        # ========== 2. 同步 Recurrent 权重 R ==========
        weight_hh = ref_gru.weight_hh_l0  # shape [4H, H]
        gates = torch.split(weight_hh, H, dim=0)  # 4 tensors of shape [H, H]
        stacked = torch.stack(gates, dim=0)  # [4, H, H]
        R = stacked.unsqueeze(0).permute(0, 2, 1, 3).contiguous()  # [1, H, 4, H]
        my_gru.recurrents[0].copy_(R)

        # ========== 3. 同步 bias ==========
        if fused:
            total_bias = ref_gru.bias_hh_l0  # shape [4H]
            gates_b = torch.split(total_bias, H, dim=0)  # 4 tensors of shape [H]
            b_stacked = (
                torch.stack(gates_b, dim=0).unsqueeze(0).permute(0, 2, 1)
            )  # [1, H, 4]
            my_gru.biases[0].copy_(b_stacked)
        else:
            total_bias = ref_gru.bias_hh_l0  # shape [4H]
            gates_b = torch.split(total_bias, H, dim=0)  # 4 tensors of shape [H]
            b_stacked = (
                torch.stack(gates_b, dim=0).unsqueeze(0).permute(0, 1, 2)
            )  # [1, H, 4]
            my_gru.biases[0].copy_(b_stacked)

        # ========== 验证是否同步成功 ==========
        # [4H, I]
        diff_w = (my_gru.linear.weight - ref_gru.weight_ih_l0).abs().max()
        print(f"[Check] Linear weight max abs diff: {diff_w:.2e}")

        # [4H]
        expected_bias = ref_gru.bias_ih_l0
        diff_b = (my_gru.linear.bias - expected_bias).abs().max()
        print(f"[Check] Linear bias   max abs diff: {diff_b:.2e}")

        # [1, H, 4, H] -> [4, H, H]
        R = my_gru.recurrents[0].permute(2, 1, 3, 0).squeeze(3)  # [4, H, H]
        R_flat = torch.cat([R[i] for i in range(3)], dim=0)  # [4H, H]
        diff_R = (R_flat - ref_gru.weight_hh_l0).abs().max()
        print(f"[Check] Recurrent weight max abs diff: {diff_R:.2e}")

        # [1, H, 4] -> [4H]
        b_my = my_gru.biases[0].permute(2, 1, 0).reshape(-1)  # [4H]
        b_ref = ref_gru.bias_hh_l0
        diff_bias = (b_my - b_ref).abs().max()
        print(f"[Check] Bias max abs diff: {diff_bias:.2e}")
    print("GRUFused 参数成功同步自 nn.GRU。")


def sync_grads_R_b(my_gru, ref_gru, fused, layer=0):

    H = my_gru.hidden_size
    weight_hh_grad = getattr(ref_gru, f"weight_hh_l{0}").grad  # [4H, H]
    if weight_hh_grad is None:
        print(f"[R] ref_gru.weight_hh_l0.grad is None")
        return

    # reshape PyTorch GRU's weight_hh gradient to match custom format
    gates = torch.split(weight_hh_grad, H, dim=0)  # 4 tensors of shape [H, H]
    ref_grad_R = torch.stack(gates, dim=0)  # [4, H, H]
    ref_grad_R = (
        ref_grad_R.unsqueeze(0).permute(0, 2, 1, 3).contiguous()
    )  # [1, H, 4, H]

    my_grad_R = my_gru.recurrents[0].grad  # [1, H, 4, H]
    if my_grad_R is None:
        print(f"[R] my_gru.recurrents[{0}].grad is None")
        return
    # ====== 提取bias 的 grad ======
    bias_grad = ref_gru.bias_hh_l0.grad  # shape: [4H]
    if bias_grad is None:
        print("[b] ref_gru.bias_hh_l0.grad is None")
        return

    H = my_gru.hidden_size
    gates_b = torch.split(bias_grad, H, dim=0)  # list of 4 [H]

    if fused:
        # 自定义的 bias shape 是 [1, H, 4]，对应 permute(0, 2, 1)
        ref_grad_b = (
            torch.stack(gates_b, dim=0).unsqueeze(0).permute(0, 2, 1).contiguous()
        )  # [1, H, 4]
    else:
        ref_grad_b = (
            torch.stack(gates_b, dim=0).unsqueeze(0).permute(0, 1, 2).contiguous()
        )  # [1, 4, H]

    # ====== 提取自定义实现中的 bias.grad ======
    my_grad_b = my_gru.biases[0].grad  # shape: [1, H, 4] or [1, 4, H]
    if my_grad_b is None:
        print("[b] my_gru.biases[0].grad is None")

    # ====== 确保两个形状一致 ======
    if my_grad_b.shape != ref_grad_b.shape:
        print(f"[b] Shape mismatch: my={my_grad_b.shape}, ref={ref_grad_b.shape}")
        return
    return my_grad_R, ref_grad_R, my_grad_b, ref_grad_b


def max_abs_diff(a, b):
    return (a - b).abs().max().item()


def mean_abs_diff(a, b):
    return (a - b).abs().mean().item()


def max_ref_diff(a, b, eps=1e-8):
    return ((a - b).abs() / (b.abs() + eps)).max().item()


def mean_ref_diff(a, b, eps=1e-8):
    return ((a - b).abs() / (b.abs() + eps)).mean().item()


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


def compare_models():
    torch.manual_seed(0)

    # Models
    ref_gru = nn.GRU(
        input_size,
        hidden_size,
        num_layers,
        bias=True,
        batch_first=True,
        bidirectional=False,
    ).to(device=device, dtype=dtype)
    my_gru = GRUFused(input_size, hidden_size, num_layers).to(
        device=device, dtype=dtype
    )
    fused = True
    # initialize_ref_gru_constant(ref_gru)
    sync_from_pytorch_gru(my_gru, ref_gru, fused)
    print(ref_gru.weight_hh_l0)
    print(my_gru.recurrents[0])
    # Inputs
    x = torch.randn(
        batch, seq_len, input_size, device="cuda", requires_grad=True, dtype=dtype
    )
    h0 = torch.zeros(
        num_layers, batch, hidden_size, device="cuda", requires_grad=True, dtype=dtype
    )

    # Clone inputs for reference
    x_ref = x.detach().clone().requires_grad_()
    h0_ref = h0.detach().clone().requires_grad_()

    # Forward
    out_ref, hn_ref = ref_gru(x_ref, h0_ref)
    # h0.retain_grad()
    out_my, hn_my = my_gru(x, h0)
    # hn_my.retain_grad()

    print("out_my shape: ", out_my.shape)
    print("out_ref shape: ", out_ref.shape)
    print("hn_my shape: ", hn_my.shape)
    print("hn_ref shape: ", hn_ref.shape)
    # Backward
    loss_my = out_my.sum()
    loss_ref = out_ref.sum()
    loss_my.backward()
    loss_ref.backward()

    # for i, (R, b) in enumerate(zip(my_gru.recurrents, my_gru.biases)):
    #     tracked[f"R_{i}"] = R
    #     tracked[f"b_{i}"] = b
    # for name, tensor in tracked.items():
    #     if not isinstance(tensor, torch.Tensor):
    #         continue
    #     grad = tensor.grad
    #     print(
    #         f"{name:10s}: "
    #         f"requires_grad={tensor.requires_grad}, "
    #         f"is_leaf={tensor.is_leaf}, "
    #         f"grad={'None' if grad is None else grad.shape}"
    #     )
    # Output comparison
    print(f"[Forward] Output max abs diff: {max_abs_diff(out_my, out_ref):.2e}")
    print(f"[Forward] hn     max abs diff: {max_abs_diff(hn_my, hn_ref):.2e}")
    print(f"[Forward] Output mean abs diff: {mean_abs_diff(out_my, out_ref):.2e}")
    print(f"[Forward] hn     mean abs diff: {mean_abs_diff(hn_my, hn_ref):.2e}")
    print(f"[Forward] Output max ref diff: {max_ref_diff(out_my, out_ref):.2e}")
    print(f"[Forward] hn     max ref diff: {max_ref_diff(hn_my, hn_ref):.2e}")
    print(f"[Forward] Output mean ref diff: {mean_ref_diff(out_my, out_ref):.2e}")
    print(f"[Forward] hn     mean ref diff: {mean_ref_diff(hn_my, hn_ref):.2e}")

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

    print(f"[Grad] h0          grad diff: {max_abs_diff(h0.grad, h0_ref.grad):.2e}")

    my_grad_R, ref_grad_R, my_grad_b, ref_grad_b = sync_grads_R_b(
        my_gru, ref_gru, fused, layer=0
    )

    generate_diff_csv(
        {
            "hn": (hn_my, hn_ref),
            "x grad": (x.grad, x_ref.grad),
            "h0 grad": (h0.grad, h0_ref.grad),
            "R grad": (my_grad_R, ref_grad_R),
            "b grad": (my_grad_b, ref_grad_b),
        },
        csvfilename,
    )


if __name__ == "__main__":
    compare_models()
