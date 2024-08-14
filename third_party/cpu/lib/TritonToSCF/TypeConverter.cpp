#include "TypeConverter.h"

#include "triton/Dialect/TritonCPU/IR/Dialect.h"

using namespace mlir;
using namespace mlir::triton;
using namespace mlir::triton::cpu;

TritonToSCFTypeConverter::TritonToSCFTypeConverter() {
  addConversion([](Type type) { return type; });
  addConversion([this](RankedTensorType tensorTy) -> Type {
    Type elemTy = convertType(tensorTy.getElementType());
    if (isa<triton::PointerType>(elemTy))
      elemTy = IntegerType::get(tensorTy.getContext(), 64);
    return MemRefType::get(tensorTy.getShape(), elemTy);
  });
}
