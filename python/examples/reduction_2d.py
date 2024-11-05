"""
Code for 2D Reduction 
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
def reduction_kernel_2d(x_ptr,
                        output_ptr,
                        n_cols,
                        n_rows,
                        BLOCK_SIZE: tl.constexpr
                        ):
    pid = tl.program_id(axis=0)
    row_start = pid * n_cols
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_cols
    x = tl.load(x_ptr + row_start + offsets, mask=mask, other=0.0)
    partial_sum = tl.sum(x, axis=0)
    tl.store(output_ptr + pid, partial_sum)

def reduction_2d(x: torch.Tensor):
    output = torch.zeros(x.shape[0], device='cpu', dtype=x.dtype)
    n_rows, n_cols = x.shape
    grid = lambda meta: (n_rows, )
    reduction_kernel_2d[grid](x, output, n_cols, n_rows, BLOCK_SIZE=16)
    return output

# src = tc.ASTSource(
#     fn=reduction_kernel_2d,
#     constants={"BLOCK_SIZE": 16},
#     signature="*fp32,*fp32,i32,i32",
# )

# ret = triton.compile(src, target=GPUTarget("cuda", 80, 32))

# # Print the compiled Triton IR
# print(ret.asm["ttir"])


# Example usage:
torch.manual_seed(0)
n_rows = 4
n_cols = 16
triton.runtime.driver.set_active_to_cpu()

x = torch.rand((n_rows, n_cols), device='cpu')
print("X: ")
print(x)

output_triton_cpu = reduction_2d(x)
print(f"Obtained Reduction result (sum along columns): {output_triton_cpu}")

output_torch = torch.sum(x, dim=1)
print(f"Expected Reduction result (sum along columns): {output_torch}")

assert torch.allclose(output_triton_cpu, output_torch), "Obtained and Expected results do not match!"

print("The results match!")