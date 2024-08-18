#include "TypeConverter.h"

#include "triton/Dialect/TritonCPU/IR/Dialect.h"

using namespace mlir;
using namespace mlir::triton;
using namespace mlir::triton::cpu;
using namespace llvm;
TritonToSCFTypeConverter::TritonToSCFTypeConverter() {
  addConversion([](mlir::Type type) { return type; });
  // Converted ops produce vectors instead of tensors. Provide conversion
  // here for users.
  addSourceMaterialization([&](OpBuilder &builder, Type type, ValueRange inputs,
                               Location loc) -> std::optional<Value> {
    return builder.create<UnrealizedConversionCastOp>(loc, type, inputs)
        .getResult(0);
  });

  // Provide conversion for vector users.
  addTargetMaterialization([&](OpBuilder &builder, Type type, ValueRange inputs,
                               Location loc) -> std::optional<Value> {
    if (isa<VectorType>(type))
      return builder.create<UnrealizedConversionCastOp>(loc, type, inputs)
          .getResult(0);
    llvm_unreachable("Unexpected target materizalization");
  });
}
