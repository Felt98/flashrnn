# Copyright 2024 NXAI GmbH
# Korbinian Poeppel

'''
封装torch.utils.cpp_extension.load(JIT动态编译和加载 C++/CUDA 扩展模块) 到load
'''
import os
from typing import Sequence, Union
import logging

import time
import random

import torch
from torch.utils.cpp_extension import load as _load

# print("INCLUDE:", torch.utils.cpp_extension.include_paths(cuda=True))
# print("C++ compat", torch.utils.cpp_extension.check_compiler_abi_compatibility("g++"))
# print("C compat", torch.utils.cpp_extension.check_compiler_abi_compatibility("gcc"))

LOGGER = logging.getLogger(__name__)


def defines_to_cflags(
    defines=Union[dict[str, Union[int, str]], Sequence[tuple[str, Union[str, int]]]],
):
    cflags = []
    LOGGER.info("Compiling definitions: ", defines)
    if isinstance(defines, dict):
        defines = defines.items()
    for key, val in defines:
        cflags.append(f"-D{key}={str(val)}")
    return cflags


curdir = os.path.dirname(__file__)

if torch.cuda.is_available():
    from packaging import version

    if version.parse(torch.__version__) >= version.parse("2.6.0"):
        os.environ["CUDA_LIB"] = os.path.join(
            os.path.split(torch.utils.cpp_extension.include_paths(device_type="cuda")[-1])[0], "lib"
        )
    else:
        os.environ["CUDA_LIB"] = os.path.join(
            os.path.split(torch.utils.cpp_extension.include_paths(cuda=True)[-1])[0], "lib"
        )


EXTRA_INCLUDE_PATHS = () + (
    tuple(os.environ["FLASHRNN_EXTRA_INCLUDE_PATHS"].split(":")) if "FLASHRNN_EXTRA_INCLUDE_PATHS" in os.environ else ()
)
if "CONDA_PREFIX" in os.environ:
    # This enforces adding the correct include directory from the CUDA installation via torch. If you use the system
    # installation, you might have to add the cflags yourself.
    from pathlib import Path
    from packaging import version
    import glob

    if version.parse(torch.__version__) >= version.parse("2.6.0"):
        matching_dirs = glob.glob(f"{os.environ['CONDA_PREFIX']}/targets/**", recursive=True)
        EXTRA_INCLUDE_PATHS = (
            EXTRA_INCLUDE_PATHS
            + tuple(map(str, (Path(os.environ["CONDA_PREFIX"]) / "targets").glob("**/include/")))[:1]
        )


'''
    name: 模块名，load会自动使用-DVALUE中的参数作为后缀
    sources： .cpp .cu的路径，如 sources=["src/add.cpp", "src/add.cu"]
    编译参数extra_cflags，extra_cuda_cflags，kwargs会被整合到 myargs
'''
def load(*, name, sources, extra_cflags=(), extra_cuda_cflags=(), **kwargs):
    suffix = ""     # 模块后缀名,_f、_h等

    # 处理extra_cflags
    for flag in extra_cflags:
        pref = [st[0] for st in flag[2:].split("=")[0].split("_")]
        if len(pref) > 1:
            pref = pref[1:]
        suffix += "".join(pref)
        value = flag[2:].split("=")[1].replace("-", "m").replace(".", "d")
        value_map = {
            "float": "f",
            "__half": "h",
            "__nv_bfloat16": "b",
            "true": "1",
            "false": "0",
        }
        if value in value_map:
            value = value_map[value]
        suffix += value
    if suffix:
        suffix = "_" + suffix
    suffix = suffix[:64]

    # 取消默认 CUDA 限制的一些 flags
    extra_cflags = list(extra_cflags) + [
        "-U__CUDA_NO_HALF_OPERATORS__",
        "-U__CUDA_NO_HALF_CONVERSIONS__",
        "-U__CUDA_NO_BFLOAT16_OPERATORS__",
        "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
        "-U__CUDA_NO_BFLOAT162_OPERATORS__",
        "-U__CUDA_NO_BFLOAT162_CONVERSIONS__",
    ]
    for eip in EXTRA_INCLUDE_PATHS:
        extra_cflags.append("-isystem")
        extra_cflags.append(eip)

    #  支持 conda 环境：将 conda 环境下的 include/ 添加到头文件路径
    conda_prefix = os.getenv("CONDA_PREFIX")
    if conda_prefix:
        conda_prefix_include = os.path.join(conda_prefix, "include")
        extra_cflags.append(f"-I{conda_prefix_include}")
        extra_cuda_cflags = list(extra_cuda_cflags) + [f"-I{conda_prefix_include}"]

    # 编译参数组装
    myargs = {
        "verbose": True,
        "with_cuda": True,
        "extra_ldflags": [f"-L{os.environ['CUDA_LIB']}", "-lcublas"],   # 链接 CUDA 和 cuBLAS 库
        "extra_cflags": [*extra_cflags],
        "extra_cuda_cflags": [
            # "-gencode",
            # "arch=compute_70,code=compute_70",
            # "-dbg=1",
            '-Xptxas="-v"',
            "-gencode",
            "arch=compute_80,code=compute_80",
            "-res-usage",
            "--use_fast_math",
            "-O3",
            "-Xptxas -O3",
            "--extra-device-vectorization",
            *extra_cflags,
            *extra_cuda_cflags,
        ],
    }
    LOGGER.info("Kernel compilation arguments", myargs)
    myargs.update(**kwargs)     # 将传入load的额外参数 **kwargs 也加入到 myargs

    # add random waiting time to minimize deadlocks because of badly managed multicompile of pytorch ext
    # 随机等待防止死锁，避免多个并发 PyTorch 扩展编译时卡死。
    time.sleep(random.random() * 10)
    LOGGER.info(f"Before compilation and loading of {name}.")
    
    mod = _load(name + suffix, sources, **myargs)
    LOGGER.info(f"After compilation and loading of {name}.")

    # 加载后的模块mod，供 Python 调用
    return mod
