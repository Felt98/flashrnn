from dataclasses import dataclass
from typing import Literal
import torch
import triton
from dataclasses import dataclass
from typing import Literal, Callable
from kernel_speed_benchmark import create_kernel2style_mapping
import sys

sys.path.append(
    ".."
)  # 将上一级目录添加到了 Python 的模块搜索路径中，不然会找不到flashrnn包
from flashrnn.flashrnn import flashrnn

from torch import nn


_flashrnn_function_to_num_gates = {
    "lstm": 4,
    "slstm": 4,
}


@dataclass
class KernelSpec:
    function: str
    backend: str
    fwbw: bool
    use_torch_compile: bool

    @staticmethod
    def parse_from_string(kernel_specifier: str):
        parts_minus = kernel_specifier.split("--")
        parts_plus = parts_minus[1].split("++")
        function = parts_minus[0]
        backend = parts_plus[0]
        fwbw = "fwbw" in parts_plus[1]
        if len(parts_plus) > 2:
            use_torch_compile = "compile" in parts_plus[2]
        else:
            use_torch_compile = False

        return KernelSpec(function, backend, fwbw, use_torch_compile)


@dataclass
class KernelSpeedBenchmarkConfig:
    benchmark_name: str
    kernel_specifiers: list[str]
    warmup: int = 500
    rep: int = 2000
    dtype: Literal["float16", "bfloat16", "float32"] = "bfloat16"


def generate_parameter_list(
    batch_size,
    seq_size,
    num_gate,
    num_heads=1,
    hidden_dim=768,
    num_layers=1,
    device="cuda",
    dtype=torch.bfloat16,
    requires_grad=True,
):
    Wx_list = []
    R_list = []
    x_list = []
    b_list = []
    # config_list = []

    # 生成各层网络的输入（Wx，R，b）
    for _ in range(num_layers):
        # Wx shape: [B, T, NG, NH, D]
        Wx = torch.randn(
            [batch_size, seq_size, num_gate, num_heads, hidden_dim],
            device=device,
            dtype=dtype,
            requires_grad=requires_grad,
        )

        # R shape: [NG, NH, D, D]
        R = torch.randn(
            [num_gate, num_heads, hidden_dim, hidden_dim],
            device=device,
            dtype=dtype,
            requires_grad=requires_grad,
        ) / (hidden_dim**0.5)
        x_only = torch.randn(
            [batch_size, seq_size, num_heads * hidden_dim],
            device=device,
            dtype=dtype,
            requires_grad=requires_grad,
        )
        # b shape: [NG, NH, D]
        b = torch.randn(
            [num_gate, num_heads, hidden_dim],
            device=device,
            dtype=dtype,
            requires_grad=requires_grad,
        )
        Wx_mtr = Wx.clone().to(dtype=dtype).detach().requires_grad_(requires_grad)
        R_mtr = R.clone().to(dtype=dtype).detach().requires_grad_(requires_grad)
        b_mtr = b.clone().to(dtype=dtype).detach().requires_grad_(requires_grad)
        x_only_mtr = (
            x_only.clone().to(dtype=dtype).detach().requires_grad_(requires_grad)
        )
        Wx_list.append(Wx_mtr)
        x_list.append(x_only_mtr)
        R_list.append(R_mtr)
        b_list.append(b_mtr)
    return Wx_list, R_list, x_list, b_list


def create_multi_layer_configs(
    benchmark_config: KernelSpeedBenchmarkConfig,
    # dh_nh_pairs: list[tuple[int, int]] = [(64, 12), (768, 1)],
) -> list[triton.testing.Benchmark]:
    """
    We vary the batch size and use these (DH, NH) pairs:
    (64, 12), (768, 1)
    Note in this experiment cuda_fused can be faster for smaller batch sizes
    if the batch_size in the config is set to 8 instead of 16.
    """
    configs = []
    kernels_to_benchmark = benchmark_config.kernel_specifiers
    B = 16
    T = 256  # 256

    configs.append(
        triton.testing.Benchmark(
            x_names=["LAYER", "DH", "NH"],
            x_vals=[
                (2, 4, 16),
                (2, 64, 1),
                (4, 32, 16),
                (4, 512, 1),
                (8, 64, 16),
                (8, 1024, 1),
            ],
            line_arg="provider",
            line_vals=kernels_to_benchmark,
            line_names=kernels_to_benchmark,
            styles=create_kernel2style_mapping(kernels_to_benchmark),
            ylabel="ms",
            plot_name=f"{benchmark_config.benchmark_name}--batch-{B}--T-{T}--dtype-{benchmark_config.dtype}",
            args={
                "B": B,
                "T": T,
            },
        )
    )
    return configs


def get_flashrnn_muli_layer_kernel_benchmark_fn(kernel_spec: KernelSpec) -> Callable:
    def kernel_fn(
        Wx: torch.Tensor,
        R: torch.Tensor,
        b: torch.Tensor,
        LAYER: int,
        dtype: str,
        gate_linear: nn.Module = None,
        x_only: torch.Tensor = None,
    ):
        if kernel_spec.use_torch_compile:
            flashrnn_fn = torch.compile(flashrnn)
        else:
            flashrnn_fn = flashrnn

        if gate_linear is not None:
            Wx = gate_linear(x_only)
            Wx = Wx.reshape(
                Wx.shape[0], Wx.shape[1], R.shape[0], R.shape[1], R.shape[2]
            )
        # kernel在这启动
        h_frnn, hlast_frnn = flashrnn_fn(
            Wx=Wx,
            R=R,
            b=b,
            num_layer=LAYER,
            states=None,
            function=kernel_spec.function,
            backend=kernel_spec.backend,
            dtype=dtype,
        )
        if kernel_spec.fwbw:
            # run the backward pass
            h_frnn[0].sum().backward()

    return kernel_fn


# 启动benchmark
def get_runnable_benchmark(
    run_configs: list[triton.testing.Benchmark],
    benchmark_config: KernelSpeedBenchmarkConfig,
):
    @triton.testing.perf_report(run_configs)
    def bench_flashrnn(
        B: int,
        NH: int,
        T: int,
        DH: int,
        LAYER: int,
        provider: str,
        bench_config: KernelSpeedBenchmarkConfig = benchmark_config,
        device: str = "cuda",
    ):
        dtype = getattr(torch, bench_config.dtype)

        # select kernel
        kernel_spec = KernelSpec.parse_from_string(kernel_specifier=provider)

        requires_grad = (
            kernel_spec.fwbw
        )  # if we are running the backward pass, we need to compute the gradients

        if kernel_spec.function == "nn.LSTM":
            nn_lstm_dtype_str = kernel_spec.backend.split("-")[-1]
            assert nn_lstm_dtype_str in [
                "bfloat16",
                "float32",
                "float16",
            ], f"Invalid dtype for nn.LSTM, got {nn_lstm_dtype_str}"
            nn_lstm_dtype = getattr(torch, nn_lstm_dtype_str)
            torch_lstm = torch.nn.LSTM(
                DH,
                DH,
                num_layers=LAYER,
                bias=True,
                batch_first=True,
                bidirectional=False,
            ).to(device=device, dtype=nn_lstm_dtype)

            pt_in = (
                torch.randn([B, T, DH], device=device, dtype=nn_lstm_dtype)
                .clone()
                .detach()
                .requires_grad_(True)
            )

            def run_kernel_fn():
                out = torch_lstm(pt_in)
                if kernel_spec.fwbw:
                    out[0].sum().backward()

        # cuda 和 triton的输入
        else:
            num_gates = _flashrnn_function_to_num_gates[kernel_spec.function]
            Wx_list, R_list, x_list, b_list = generate_parameter_list(
                B,
                T,
                num_gates,
                NH,
                DH,
                LAYER,
                device=device,
                dtype=dtype,
                requires_grad=requires_grad,
            )

            # check if we need to add a torch nn.Linear layer to compute the gate preactivations
            kernel_backend_split = kernel_spec.backend.split("-")
            if len(kernel_backend_split) > 1:
                assert len(kernel_backend_split) == 2, "Invalid kernel backend"
                backend_name = kernel_backend_split[0]
                if "withlinear" in kernel_backend_split[1]:
                    with_linear = True
                else:
                    with_linear = False
            else:
                backend_name = kernel_spec.backend
                with_linear = False

            kernel_spec.backend = backend_name

            if with_linear:
                gate_linear = torch.nn.Linear(NH * DH, num_gates * NH * DH).to(
                    device=device, dtype=dtype
                )
            else:
                gate_linear = None

            # get the benchmark function
            # flashrnn的kernel在这里封装
            kernel_benchmark_fn = get_flashrnn_muli_layer_kernel_benchmark_fn(
                kernel_spec
            )

            # 启动kernel
            def run_kernel_fn():
                kernel_benchmark_fn(
                    Wx_list,
                    R_list,
                    b_list,
                    LAYER,
                    bench_config.dtype,
                    gate_linear=gate_linear,
                    x_only=x_list,
                )

        print(
            f"[NEW CONFIGURATION] Running speedtest for {provider}, with batch size {B}, num heads {NH}, context size {T}, head dim {DH}, dtype {bench_config.dtype}"
        )
        try:
            # warmup = 热身，不记录时间
            # rep = 正式运行并计时的重复次数
            # ms = rep次数运行时间的平均耗时
            ms = triton.testing.do_bench(
                run_kernel_fn, warmup=bench_config.warmup, rep=bench_config.rep
            )
        except Exception as e:
            print(f"Error: {e}")
            ms = float("nan")
        return ms

    return bench_flashrnn


# 额外的batch_size实验
def paper_plot_experiments_additional():
    OUTPUT_DIR = "./outputs_speed_exps_add_v2"
    ### head dimension experiment
    print("====================================")
    print("BATCH SIZE ADDITIONAL EXPERIMENT")
    print("====================================")
    batch_size_add_benchmark_config = KernelSpeedBenchmarkConfig(
        benchmark_name="batch_size_exp_additional",
        kernel_specifiers=[
            ## lstm
            # fw
            # "lstm--vanilla++fw",
            # "lstm--vanilla_fwbw++fw",
            # "lstm--triton_fused++fw",
            "lstm--cuda_fused++fw",
            "lstm--cuda++fw",
            # "lstm--triton_fused-withlinear++fw",
            "lstm--cuda_fused-withlinear++fw",
            "lstm--cuda-withlinear++fw",
            # fwbw
            # "lstm--vanilla_fwbw++fwbw",
            # "lstm--vanilla++fwbw",
            # "lstm--triton_fused++fwbw",
            "lstm--cuda_fused++fwbw",
            "lstm--cuda++fwbw",
            # "lstm--triton_fused-withlinear++fwbw",
            "lstm--cuda_fused-withlinear++fwbw",
            "lstm--cuda-withlinear++fwbw",
            ## baselines
            "nn.LSTM--pytorch-float32++fw",
            "nn.LSTM--pytorch-float32++fwbw",
            "nn.LSTM--pytorch-float16++fw",
            "nn.LSTM--pytorch-float16++fwbw",
        ],
        warmup=25,
        rep=1000,
        dtype="bfloat16",
    )
    #
    # B = 16, T = 256
    # [NLayer,DH,NH]: (2, 4, 16), (2, 64, 1), (4, 32, 16), (4, 512, 1), (8, 64, 16), (8, 1024, 1),
    batch_size_run_configs_additional = create_multi_layer_configs(
        batch_size_add_benchmark_config
    )

    # 启动benchmark
    batch_size_add_benchmark_fn = get_runnable_benchmark(
        batch_size_run_configs_additional, batch_size_add_benchmark_config
    )

    batch_size_add_benchmark_fn.run(
        save_path=f"{OUTPUT_DIR}/{batch_size_add_benchmark_config.benchmark_name}",
        print_data=True,
    )
    ### =================
