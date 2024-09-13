"""
Code for 1D reduction using triton
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
def reduction_kernel(x_ptr,  # *Pointer* to input vector.
                     output_ptr,  # *Pointer* to output scalar (final sum).
                     n_elements,  # Size of the vector.
                     BLOCK_SIZE: tl.constexpr  # Number of elements each program processes.
                     ):
    pid = tl.program_id(axis=0)  # Program ID (for block-level parallelism).
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    
    # Mask for boundaries of the input tensor.
    mask = offsets < n_elements
    
    # Load elements of the input tensor.
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    
    # Perform a block-wide reduction (sum within the block).
    partial_sum = tl.sum(x, axis=0)
    
    tl.atomic_add(output_ptr, partial_sum)


def reduction(x: torch.Tensor):
    # Prepare output tensor (final result, scalar).
    output = torch.zeros(1, device='cpu', dtype=x.dtype)
    
    n_elements = x.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']), )
    
    # Launch the Triton kernel.
    reduction_kernel[grid](x, output, n_elements, BLOCK_SIZE=16)
    
    return output.item()


src = tc.ASTSource(
    fn=reduction_kernel,
    constants={"BLOCK_SIZE": 16},
    signature="*fp32,*fp32,i32",
)

# # # Compile the source for a specific GPU target
# ret = triton.compile(src, target=GPUTarget("cuda", 80, 32))

# # Print the compiled Triton IR
# print(ret.asm["ttir"])

# Example usage:
torch.manual_seed(0)
size = 16
triton.runtime.driver.set_active_to_cpu()

x = torch.rand(size, device='cpu')
print("X: ")
print(x)

output_triton_cpu = reduction(x)
print(f"Obtained Reduction result(sum): {output_triton_cpu}")

output_torch = torch.sum(x).item()
print(f"Expected Reduction result (sum): {output_torch}")

assert abs(output_triton_cpu - output_torch) < 1e-6, "Obtained and Expected results do not match!"

print("The results match!")