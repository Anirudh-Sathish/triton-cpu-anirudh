import torch 
import triton
import triton.language as tl
import triton.compiler as tc
from triton.backends.compiler import GPUTarget

CPU_BLOCK_SIZE = 4

import torch

import triton
import triton.language as tl
import triton.compiler as tc
from triton.backends.compiler import GPUTarget

GPU_BLOCK_SIZE = 1024
CPU_BLOCK_SIZE = 4
USE_GPU = False

@triton.jit
def copy_kernel_2d(x_ptr, output_ptr,M,N, BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr):
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)
    block_start_m = pid_m * BLOCK_SIZE_M
    block_start_n = pid_n * BLOCK_SIZE_N
    offsets_m = block_start_m + tl.arange(0, BLOCK_SIZE_M)[:, None]
    offsets_n = block_start_n + tl.arange(0, BLOCK_SIZE_N)[None, :]
    x = tl.load(x_ptr + offsets_m * N + offsets_n)
    tl.store(output_ptr + offsets_m * N + offsets_n, x)

def store_2d(x: torch.Tensor, output: torch.Tensor):
    if output is None:
        output = torch.empty_like(x)
    
    n_elements_m, n_elements_n = output.shape
    grid = lambda meta: (
        triton.cdiv(n_elements_m, meta['BLOCK_SIZE_M']),
        triton.cdiv(n_elements_n, meta['BLOCK_SIZE_N']),
    )
    
    copy_kernel_2d[grid](x, output,n_elements_m, n_elements_n, BLOCK_SIZE_M=CPU_BLOCK_SIZE, BLOCK_SIZE_N=CPU_BLOCK_SIZE)
    return output

# src = tc.ASTSource(
#     fn=copy_kernel_2d,
#     constants={"BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 16},
#     signature="*fp32,*fp32",
# )

# # Compile the source for a specific GPU target
# ret = triton.compile(src, target=GPUTarget("cuda", 80, 32))

# # Print the compiled Triton IR
# print(ret.asm["ttir"])

torch.manual_seed(0)
M, N = 8, 8
x = torch.rand((M, N), device='cpu')
y = torch.rand((M, N), device='cpu')

output_triton_cpu = store_2d(x, y)

print("Input Tensor (x):")
print(x)
print("Output Tensor after Triton Kernel (output_triton_cpu):")
print(output_triton_cpu)
assert torch.equal(output_triton_cpu, x)