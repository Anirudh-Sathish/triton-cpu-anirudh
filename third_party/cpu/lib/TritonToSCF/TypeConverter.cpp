#include "TypeConverter.h"

#include "triton/Dialect/TritonCPU/IR/Dialect.h"

using namespace mlir;
using namespace mlir::triton;
using namespace mlir::triton::cpu;
using namespace llvm;
TritonToSCFTypeConverter::TritonToSCFTypeConverter() {
  addConversion([](mlir::Type type) { return type; });
}
