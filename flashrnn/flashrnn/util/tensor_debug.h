#pragma once
#include <cuda_runtime.h>
#include <fstream>
#include <iostream>
#include <memory>
#include <string>
#include <vector>
#include <torch/extension.h>

namespace tensor_debug
{
template <typename T>
void dump_cuda_tensor(const T *device_ptr, size_t num_elements, const std::string &bin_path,
                      const std::string &meta_path, std::vector<int64_t> shape, bool verbose = true)
{
    // T* host_buffer=new T[num_elements];
    std::unique_ptr<T[]> host_buffer(new T[num_elements]);
    cudaError_t err = cudaMemcpy(host_buffer.get(), device_ptr, sizeof(T) * num_elements,
                                 cudaMemcpyDeviceToHost) ;
    if (err != cudaSuccess)
    {
        std::cerr << "[DUMP] cudaMemcpy failed: " << cudaGetErrorString(err) << std::endl;
        return;
    }
    std::ofstream fout(bin_path, std::ios::binary);
    if (!fout)
    {
        std::cerr << "[DUMP] Failed to open file: " << bin_path << std::endl;
        return;
    }
    fout.write(reinterpret_cast<char *>(host_buffer.get()), sizeof(T) * num_elements);
    fout.close();

    std::ofstream meta(meta_path);
    for (size_t i = 0; i < shape.size(); ++i)
    {
        meta << shape[i];
        if (i < shape.size() - 1)
            meta << " ";
    }
    meta << "\n";
    meta.close();

    if (verbose)
    {
        std::cout << "[DUMP] Saved tensor to " << bin_path << " with shape ";
        for (auto s : shape)
            std::cout << s << " ";
        std::cout << std::endl;
    }
}

template<typename T>
void dump_tensor(torch::Tensor & tensor, const std::string& bin_path, const std::string& meta_path, bool verbose = true){
    TORCH_CHECK(tensor.is_cuda(), "[DUMP] Tensor must be on CUDA device.");
    const T* device_ptr = tensor.data_ptr<T>();
    size_t num_elements = tensor.numel();
    std::vector<int64_t> shape(tensor.sizes().begin(), tensor.sizes().end());
    dump_cuda_tensor<T>(device_ptr, num_elements, bin_path, meta_path, shape, verbose);
}

} // namespace tensor_debug