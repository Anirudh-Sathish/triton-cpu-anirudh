#ifndef TRITONTOSCF_LOWERING_PASSES
#define TRITONTOSCF_LOWERING_PASSES

include "mlir/Pass/PassBase.td"

def LowerTritonToSCF : Pass<"lower-triton-to-scf", "mlir::ModuleOp"> {
    let summary = "Lower ttir to to scf";
    let description = [{

    }];
    let constructor = "mlir::triton::cpu::createLowerTritonToSCF()";

    let dependentDialects = ["mlir::arith::ArithDialect",
                             "mlir::memref::MemRefDialect",
                             "mlir::vector::VectorDialect",
                             "mlir::triton::TritonDialect",
                             "mlir::triton::cpu::TritonCPUDialect"];
}


#endif
