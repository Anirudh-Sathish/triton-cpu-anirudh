
"""
Code for 1D elementwise addition
"""
import torch

import triton
import triton.language as tl
import triton.compiler as tc
from triton.backends.compiler import GPUTarget

GPU_BLOCK_SIZE = 1024
CPU_BLOCK_SIZE = 4
USE_GPU = False


@triton.jit
def add_kernel(x_ptr,  # *Pointer* to first input vector.
               y_ptr,  # *Pointer* to second input vector.
               output_ptr,  # *Pointer* to output vector.
               n_elements,  # Size of the vector.
               BLOCK_SIZE: tl.constexpr,  # Number of elements each program should process.
               # NOTE: `constexpr` so it can be used as a shape value.
               ):
    pid = tl.program_id(axis=0) 
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    output = x + y
    tl.store(output_ptr + offsets, output, mask=mask)

def add(x: torch.Tensor, y: torch.Tensor, output: torch.Tensor):
    if output is None:
        output = torch.empty_like(x)
    n_elements = output.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']), )
    add_kernel[grid](x, y, output, n_elements, BLOCK_SIZE=CPU_BLOCK_SIZE)
    return output


# src = tc.ASTSource(
#     fn=add_kernel,
#     constants={"BLOCK_SIZE": 16},
#     signature="*fp32,*fp32,*fp32,i32",
# )

# ret = triton.compile(src, target=GPUTarget("cuda",80,32))
# print(ret.asm["ttir"])

torch.manual_seed(0)
size = 16
triton.runtime.driver.set_active_to_cpu()
x = torch.rand(size, device='cpu')
y = torch.rand(size, device='cpu')
print("X: ")
print(x)
print("Y: ")
print(y)
output_triton_cpu = add(x, y, None)
required = x+y
print("Triton Output after pass: ")
print(output_triton_cpu)
print("Required output: ")
print(required)
assert torch.equal(output_triton_cpu, required)