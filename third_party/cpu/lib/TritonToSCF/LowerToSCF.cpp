/*
File to lower triton ir to scf

*/

#include "cpu/include/Analysis/TensorPtrShapeInfo.h"
#include "cpu/include/TritonToSCF/Passes.h"
#include "TypeConverter.h"


#include "mlir/Analysis/DataFlowFramework.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Index/IR/IndexDialect.h"
#include "mlir/Dialect/Index/IR/IndexOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Utils/IndexingUtils.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "triton/Analysis/Allocation.h"
#include "triton/Analysis/AxisInfo.h"
#include "triton/Analysis/Membar.h"
#include "triton/Conversion/TritonGPUToLLVM/Utility.h"
#include "triton/Dialect/Triton/IR/Dialect.h"


#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinDialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Support/TypeID.h"


#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include <algorithm>
#include <cstdint>
#include <functional>
#include <memory>
#include <utility>


namespace mlir {
namespace triton {
#define GEN_PASS_DEF_LOWERTRITONTOSCF
#include "cpu/include/TritonToSCF/Passes.h.inc"
} // namespace triton
} // namespace mlir

using namespace mlir;
using namespace mlir::triton;
using namespace mlir::triton::cpu;
using namespace llvm;
using scf::LoopNest;  

class PtrConversionTarget : public ConversionTarget {
public:
  explicit PtrConversionTarget(MLIRContext &ctx, TypeConverter &converter)
      : ConversionTarget(ctx) {
    addLegalDialect<arith::ArithDialect>();
    addLegalDialect<memref::MemRefDialect>();
    addLegalDialect<scf::SCFDialect>();  
    addLegalOp<mlir::UnrealizedConversionCastOp>();

    addDynamicallyLegalOp<triton::PtrToIntOp>(
        [](triton::PtrToIntOp op) { return op.getType().isInteger(); });
    addDynamicallyLegalOp<triton::IntToPtrOp>([](triton::IntToPtrOp op) {
      return op.getSrc().getType().isInteger();
    });
  }
};


struct MakeRangeOpConversion : public OpConversionPattern<triton::MakeRangeOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(triton::MakeRangeOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    int64_t start = static_cast<int64_t>(op.getStart());
    int64_t end = static_cast<int64_t>(op.getEnd());
    assert(end >= start);
    int64_t size = end - start;
    auto memrefType = MemRefType::get({size}, rewriter.getIntegerType(64));
    auto alloc = rewriter.create<memref::AllocOp>(op.getLoc(), memrefType);
    auto lowerBound = rewriter.create<arith::ConstantIndexOp>(op.getLoc(), start);
    auto upperBound = rewriter.create<arith::ConstantIndexOp>(op.getLoc(), end);
    auto step = rewriter.create<arith::ConstantIndexOp>(op.getLoc(), 1);
    rewriter.create<scf::ForOp>(
      op.getLoc(), lowerBound, upperBound, step, ValueRange{}, 
      [&](OpBuilder &nestedBuilder, Location loc, mlir::Value iv, ValueRange /*args*/) {
        auto value = nestedBuilder.create<arith::ConstantIntOp>(loc, start, 64);
        auto increment = nestedBuilder.create<arith::IndexCastOp>(loc, nestedBuilder.getIntegerType(64), iv);
        auto finalValue = nestedBuilder.create<arith::AddIOp>(loc, value, increment);
        nestedBuilder.create<memref::StoreOp>(loc, finalValue, alloc, iv);
        nestedBuilder.create<scf::YieldOp>(loc);
      });
    rewriter.replaceOp(op, alloc.getResult());
    return success();
  }
};

namespace{
    struct LowerTritonToSCF
    : public triton::impl::LowerTritonToSCFBase<LowerTritonToSCF> {
  using LowerTritonToSCFBase::LowerTritonToSCFBase;

  LowerTritonToSCF() : LowerTritonToSCFBase() {}

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    ModuleOp mod = getOperation();
    TritonToSCFTypeConverter typeConverter;
    PtrConversionTarget convTarget(*context, typeConverter);
    RewritePatternSet patterns(context);
    patterns.add<MakeRangeOpConversion>(typeConverter, context);
    if (failed(applyPartialConversion(mod, convTarget, std::move(patterns))))
      return signalPassFailure();
  }
};

}



namespace mlir {
namespace triton {
namespace cpu {

std::unique_ptr<OperationPass<ModuleOp>> createLowerTritonToSCF() {
  return std::make_unique<LowerTritonToSCF>();
}

} // namespace cpu
} // namespace triton
} // namespace mlir
