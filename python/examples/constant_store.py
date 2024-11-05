import torch 
import triton
import triton.language as tl
import triton.compiler as tc
from triton.backends.compiler import GPUTarget


GPU_BLOCK_SIZE = 1024
CPU_BLOCK_SIZE = 4
USE_GPU = False

@triton.jit
def constant_store(x_ptr, 
           y_ptr,
           output_ptr,
           n_elements,
           BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis = 0)
    block_start = pid*BLOCK_SIZE
    offsets= block_start + tl.arange(0, BLOCK_SIZE)
    output = float(3.0)
    tl.store(output_ptr+offsets,output)

def store(x: torch.Tensor, y: torch.Tensor, output: torch.Tensor):
    if output is None:
        output = torch.empty_like(x)
    n_elements = output.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']), )
    constant_store[grid](x, y, output, n_elements, BLOCK_SIZE=CPU_BLOCK_SIZE)
    return output

# src = tc.ASTSource(
#     fn=constant_store,
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
output_triton_cpu = store(x, y, None)
required = torch.full((size,), 3.0, device='cpu')
print(output_triton_cpu)
print(required)
assert torch.equal(output_triton_cpu, required)