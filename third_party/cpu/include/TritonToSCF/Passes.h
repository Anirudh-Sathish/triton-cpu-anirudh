#ifndef TRITONTOSCF_CONVERSION_PASSES_H
#define TRITONTOSCF_CONVERSION_PASSES_H

#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

#include <memory>

namespace mlir {

class ModuleOp;
template <typename T> class OperationPass;

namespace triton {
namespace cpu {

#define GEN_PASS_DECL
#include "cpu/include/TritonToSCF/Passes.h.inc"

std::unique_ptr<OperationPass<ModuleOp>> createLowerTritonToSCF();

#define GEN_PASS_REGISTRATION
#include "cpu/include/TritonToSCF/Passes.h.inc"

} // namespace cpu
} // namespace triton

} // namespace mlir

#endif
