import torch

import triton
import triton.language as tl
import triton.compiler as tc
from triton.backends.compiler import GPUTarget

GPU_BLOCK_SIZE = 1024
CPU_BLOCK_SIZE = 4
USE_GPU = False

@triton.jit
def copy_kernel(x_ptr, 
           output_ptr,
           BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis = 0)
    block_start = pid*BLOCK_SIZE
    offsets= block_start + tl.arange(0, BLOCK_SIZE)
    x = tl.load(x_ptr+offsets)
    tl.store(output_ptr+offsets,x)


def store(x: torch.Tensor, output: torch.Tensor):
    if output is None:
        output = torch.empty_like(x)
    n_elements = output.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']), )
    copy_kernel[grid](x, output , BLOCK_SIZE=CPU_BLOCK_SIZE)
    return output

# src = tc.ASTSource(
#     fn=copy_kernel,
#     constants={"BLOCK_SIZE": 16},
#     signature="*fp32,*fp32",
# )

# ret = triton.compile(src, target=GPUTarget("cuda",80,32))
# print(ret.asm["ttir"])

torch.manual_seed(0)
size = 16
triton.runtime.driver.set_active_to_cpu()
x = torch.rand(size, device='cpu')
output_triton_cpu = store(x, None)
print(output_triton_cpu)
print(x)
assert torch.equal(output_triton_cpu, x)