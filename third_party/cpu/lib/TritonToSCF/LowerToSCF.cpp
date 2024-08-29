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
#include "mlir/Dialect/Vector/IR/VectorOps.h"

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
    addLegalDialect<TritonCPUDialect>();
    addLegalDialect<vector::VectorDialect>();

    addDynamicallyLegalOp<triton::PtrToIntOp>(
        [](triton::PtrToIntOp op) { return op.getType().isInteger(); });
    addDynamicallyLegalOp<triton::IntToPtrOp>([](triton::IntToPtrOp op) {
      return op.getSrc().getType().isInteger();
    });
  }
};

struct SplatOpConversion : public OpConversionPattern<triton::SplatOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(triton::SplatOp op, OpAdaptor adaptor,
                 ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto splatValue = adaptor.getSrc(); 
    auto splatType = splatValue.getType(); 
    auto resultType = op.getType();
    mlir::Type pointee;
    bool isPointer = false;
     if (auto ptrType =dyn_cast<mlir::triton::PointerType>(splatType)) {
      isPointer = true;
      pointee = ptrType.getPointeeType();
    }
    int64_t resSize = 0;
    llvm::ArrayRef<int64_t> resShape;
    if (auto shapedType = dyn_cast<mlir::ShapedType>(resultType)) {
      resShape = shapedType.getShape();
      resSize = resShape.size();
    } else {
      llvm::errs() << "Result type is not a ShapedType.\n";
      return failure();
    }
    MemRefType memrefType;
    if(isPointer)
      memrefType = MemRefType::get(resShape, pointee);
    else
      memrefType = MemRefType::get(resShape, splatValue.getType());
    auto alloc = rewriter.create<memref::AllocOp>(loc, memrefType);
    auto lowerBound = rewriter.create<arith::ConstantIndexOp>(loc, 0);
    auto upperBound = rewriter.create<arith::ConstantIndexOp>(loc, resShape[0]);
    auto step = rewriter.create<arith::ConstantIndexOp>(loc, 1);
    rewriter.create<scf::ForOp>(
      loc, lowerBound, upperBound, step, ValueRange{},
      [&](OpBuilder &nestedBuilder, Location loc, mlir::Value iv, ValueRange ) {
        if(isPointer){
          auto cast = rewriter.create<UnrealizedConversionCastOp>(loc, pointee, splatValue).getResult(0);
          nestedBuilder.create<memref::StoreOp>(loc, cast, alloc, iv);
        }
        else{
          nestedBuilder.create<memref::StoreOp>(loc, splatValue, alloc, iv);
        }
        nestedBuilder.create<scf::YieldOp>(loc);
        });
    rewriter.replaceOp(op, alloc.getResult());
    return success();
  }
};

struct LoadOpConversion : public OpConversionPattern<triton::LoadOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(triton::LoadOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    mlir::Value ptr = op.getPtr();
    mlir::Type resultType = op.getResult().getType();
    mlir::Type ptrType = ptr.getType();
    llvm::ArrayRef<int64_t> ptrShape;
    mlir::Type pointeeType;
    auto loc = op.getLoc();
    if (auto shapedType = dyn_cast<mlir::ShapedType>(resultType)) {
      ptrShape = shapedType.getShape();
    } else {
      llvm::errs() << "Result type is not a ShapedType.\n";
      return failure();
    }

    if (auto tensorType = dyn_cast<RankedTensorType>(ptrType)) {
      auto elemTy = tensorType.getElementType();
      if (auto ptrType = dyn_cast<mlir::triton::PointerType>(elemTy)) {
        pointeeType = ptrType.getPointeeType();
      } else {
        llvm::errs() << "elemTy is not a pointer.\n";
        return failure();
      }
    } else {
      llvm::errs() << "Not a Tensor Type.\n";
      return failure();
    }
    auto memrefType = MemRefType::get(ptrShape, pointeeType);
    auto castPtr = rewriter.create<UnrealizedConversionCastOp>(
        loc, memrefType, ptr);
    auto lowerBound = rewriter.create<arith::ConstantIndexOp>(loc, 0);
    auto upperBound = rewriter.create<arith::ConstantIndexOp>(loc, ptrShape[0]);
    auto step = rewriter.create<arith::ConstantIndexOp>(loc, 1);
    auto alloc = rewriter.create<memref::AllocOp>(loc, memrefType);
    rewriter.create<scf::ForOp>(
        loc, lowerBound, upperBound, step, ValueRange{},
        [&](OpBuilder &nestedBuilder, Location loc, mlir::Value iv,
            ValueRange ) {
          auto loadedValue = nestedBuilder.create<memref::LoadOp>(
              loc, castPtr.getResult(0), iv);
          nestedBuilder.create<memref::StoreOp>(loc, loadedValue,
                                                alloc, iv);
          nestedBuilder.create<scf::YieldOp>(loc);
        });
    rewriter.replaceOp(op, alloc.getResult());

    return success();
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
      [&](OpBuilder &nestedBuilder, Location loc, mlir::Value iv, ValueRange ) {
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

struct StoreOpConversion : public OpConversionPattern<triton::StoreOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(triton::StoreOp op, OpAdaptor adaptor,
  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto ptrs = rewriter.getRemappedValue(op.getPtr());
    auto vals = rewriter.getRemappedValue(op.getValue());
    llvm::ArrayRef<int64_t> ptrShape;
    mlir::Value ptr = rewriter.create<vector::ExtractOp>(loc, ptrs, 0);
    auto tensorTy = dyn_cast<RankedTensorType>(op.getPtr().getType());
    auto ptrTy = tensorTy.getElementType();
    auto updatedPtr = rewriter.create<IntToPtrOp>(loc, ptrTy, ptr);
    mlir::Type resTy = getTypeConverter()->convertType(vals.getType());
    mlir::Type elemTy;
    if (auto vecTy =dyn_cast<mlir::VectorType>(resTy))
    {
      elemTy = vecTy.getElementType();
    }
    if (auto shapedType = dyn_cast<mlir::ShapedType>(vals.getType())) {
      ptrShape = shapedType.getShape();
    }
    mlir::Type memRefTy = MemRefType::get(ptrShape, elemTy);
    mlir::Value memRef =rewriter.create<triton::cpu::PtrToMemRefOp>(loc, memRefTy, updatedPtr);
    auto lowerBound = rewriter.create<arith::ConstantIndexOp>(loc, 0);
    auto upperBound =rewriter.create<arith::ConstantIndexOp>(loc, ptrShape[0]);
    auto step = rewriter.create<arith::ConstantIndexOp>(loc, 1);
    auto forOp = rewriter.create<scf::ForOp>(
      loc, lowerBound, upperBound, step, ValueRange{},
      [&](OpBuilder &nestedBuilder, Location loc, mlir::Value iv, ValueRange ) {
        auto constantValue = rewriter.create<vector::ExtractOp>(loc, vals, 0);
        nestedBuilder.create<memref::StoreOp>(loc,constantValue, memRef, iv);
        nestedBuilder.create<scf::YieldOp>(loc);
        });
    rewriter.eraseOp(op);
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
    // patterns.add<MakeRangeOpConversion>(typeConverter, context);
    // patterns.add<SplatOpConversion>(typeConverter, context);
    // patterns.add<LoadOpConversion>(typeConverter, context);
    patterns.add<StoreOpConversion>(typeConverter, context);
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
