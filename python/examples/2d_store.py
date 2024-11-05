import torch 
import triton
import triton.language as tl
import triton.compiler as tc
from triton.backends.compiler import GPUTarget

CPU_BLOCK_SIZE = 4

@triton.jit
def constant_store_2d(x_ptr, y_ptr, output_ptr, M, N, BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr):
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)
    block_start_m = pid_m * BLOCK_SIZE_M
    block_start_n = pid_n * BLOCK_SIZE_N
    
    offsets_m = block_start_m + tl.arange(0, BLOCK_SIZE_M)
    offsets_n = block_start_n + tl.arange(0, BLOCK_SIZE_N)
    # Create 2D grid of offsets
    offset_grid = offsets_m[:, None] * N + offsets_n[None, :]
    # Set the output to 3.0
    output = tl.full((BLOCK_SIZE_M, BLOCK_SIZE_N), 3.0, dtype=tl.float32)
    
    # Store the output in memory
    tl.store(output_ptr + offset_grid, output)
    # return offset_grid

def store_2d(x: torch.Tensor, y: torch.Tensor, output: torch.Tensor):
    if output is None:
        output = torch.empty_like(x)
    
    M, N = output.shape
    grid = lambda meta: (triton.cdiv(M, meta['BLOCK_SIZE_M']), triton.cdiv(N, meta['BLOCK_SIZE_N']))
    constant_store_2d[grid](x, y, output, M, N, BLOCK_SIZE_M=CPU_BLOCK_SIZE, BLOCK_SIZE_N=CPU_BLOCK_SIZE)
    return output

# src = tc.ASTSource(
#     fn=constant_store_2d,
#     constants={"BLOCK_SIZE_M": 4, "BLOCK_SIZE_N": 4},
#     signature="*fp32,*fp32,*fp32,i32,i32"
# )

# ret = triton.compile(src, target=GPUTarget("cuda",80,32))
# print(ret.asm["ttir"])

torch.manual_seed(0)
M, N = 16, 16
triton.runtime.driver.set_active_to_cpu()
x = torch.rand((M, N), device='cpu')
y = torch.rand((M, N), device='cpu')
output_triton_cpu = store_2d(x, y, None)
required = torch.full((M, N), 3.0, device='cpu')
print("Output after pass")
print(output_triton_cpu)
print("Expected Output")
print(required)
assert torch.equal(output_triton_cpu, required), "Output does not match expected result."
