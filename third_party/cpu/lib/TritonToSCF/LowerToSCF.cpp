/*
File to lower triton ir to scf

*/

#include "OpTypeConversion.h"
#include "TypeConverter.h"
#include "cpu/include/Analysis/TensorPtrShapeInfo.h"
#include "cpu/include/TritonToSCF/Passes.h"

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
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Transforms/DialectConversion.h"
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
    addDynamicallyLegalDialect<math::MathDialect>(
        [&](Operation *op) -> std::optional<bool> {
          return converter.isLegal(op);
        });
    addDynamicallyLegalDialect<arith::ArithDialect>(
        [&](Operation *op) -> std::optional<bool> {
          return converter.isLegal(op);
        });

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
    if (auto ptrType = dyn_cast<mlir::triton::PointerType>(splatType)) {
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
    if (isPointer)
      memrefType = MemRefType::get(resShape, pointee);
    else
      memrefType = MemRefType::get(resShape, splatValue.getType());
    auto alloc = rewriter.create<memref::AllocOp>(loc, memrefType);
    auto lowerBound = rewriter.create<arith::ConstantIndexOp>(loc, 0);
    auto upperBound = rewriter.create<arith::ConstantIndexOp>(loc, resShape[0]);
    auto step = rewriter.create<arith::ConstantIndexOp>(loc, 1);
    rewriter.create<scf::ForOp>(
        loc, lowerBound, upperBound, step, ValueRange{},
        [&](OpBuilder &nestedBuilder, Location loc, mlir::Value iv,
            ValueRange) {
          if (isPointer) {
            auto cast = rewriter
                            .create<UnrealizedConversionCastOp>(loc, pointee,
                                                                splatValue)
                            .getResult(0);
            nestedBuilder.create<memref::StoreOp>(loc, cast, alloc, iv);
          } else {
            nestedBuilder.create<memref::StoreOp>(loc, splatValue, alloc, iv);
          }
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
    auto lowerBound =
        rewriter.create<arith::ConstantIndexOp>(op.getLoc(), start);
    auto upperBound = rewriter.create<arith::ConstantIndexOp>(op.getLoc(), end);
    auto step = rewriter.create<arith::ConstantIndexOp>(op.getLoc(), 1);
    rewriter.create<scf::ForOp>(
        op.getLoc(), lowerBound, upperBound, step, ValueRange{},
        [&](OpBuilder &nestedBuilder, Location loc, mlir::Value iv,
            ValueRange) {
          auto value =
              nestedBuilder.create<arith::ConstantIntOp>(loc, start, 64);
          auto increment = nestedBuilder.create<arith::IndexCastOp>(
              loc, nestedBuilder.getIntegerType(64), iv);
          auto finalValue =
              nestedBuilder.create<arith::AddIOp>(loc, value, increment);
          nestedBuilder.create<memref::StoreOp>(loc, finalValue, alloc, iv);
          nestedBuilder.create<scf::YieldOp>(loc);
        });
    rewriter.replaceOp(op, alloc.getResult());
    return success();
  }
};

// This conversion pattern handles the transformation of `triton::StoreOp`
// operations. It converts the Triton StoreOp to SCF dialect operations by first
// extracting the pointer and value, converting the pointer to a MemRef type,
// and then storing the value into the MemRef using a loop. The original Triton
// StoreOp is erased and replaced with the new operations.
struct StoreOpConversion : public OpConversionPattern<triton::StoreOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(triton::StoreOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto ptrs = rewriter.getRemappedValue(op.getPtr());
    auto vals = rewriter.getRemappedValue(op.getValue());
    llvm::ArrayRef<int64_t> ptrShape;
    auto tensorTy = dyn_cast<RankedTensorType>(op.getPtr().getType());
    auto ptrTy = tensorTy.getElementType();
    mlir::Type resTy = getTypeConverter()->convertType(vals.getType());
    mlir::Type elemTy;
    if (auto vecTy = dyn_cast<mlir::VectorType>(resTy)) {
      elemTy = vecTy.getElementType();
    }
    if (auto shapedType = dyn_cast<mlir::ShapedType>(vals.getType())) {
      ptrShape = shapedType.getShape();
    }
    if (ptrShape.size() == 2) {
      mlir::Value row, ptr, updatedPtr, memRef, newIdx, zeroIdx;
      mlir::Type memRefTy = MemRefType::get(ptrShape, elemTy);
      auto cast =
          rewriter.create<UnrealizedConversionCastOp>(loc, memRefTy, vals)
              .getResult(0);
      auto lowerBound1 = rewriter.create<arith::ConstantIndexOp>(loc, 0);
      auto upperBound1 =
          rewriter.create<arith::ConstantIndexOp>(loc, ptrShape[1]);
      auto step1 = rewriter.create<arith::ConstantIndexOp>(loc, 1);
      for (int i = 0; i < ptrShape[0]; i++) {
        row = rewriter.create<vector::ExtractOp>(loc, ptrs, i);
        ptr = rewriter.create<vector::ExtractOp>(loc, row, 0);
        updatedPtr = rewriter.create<IntToPtrOp>(loc, ptrTy, ptr);
        memRef = rewriter.create<triton::cpu::PtrToMemRefOp>(loc, memRefTy,
                                                             updatedPtr);
        rewriter.create<scf::ForOp>(
            loc, lowerBound1, upperBound1, step1, ValueRange{},
            [&](OpBuilder &nestedBuilder, Location loc, mlir::Value iv1,
                ValueRange) {
              newIdx = rewriter.create<arith::ConstantIndexOp>(loc, i);
              zeroIdx = rewriter.create<arith::ConstantIndexOp>(loc, 0);
              auto value = nestedBuilder.create<memref::LoadOp>(
                  loc, cast, ValueRange{newIdx, iv1});
              nestedBuilder.create<memref::StoreOp>(loc, value, memRef,
                                                    ValueRange{zeroIdx, iv1});
              nestedBuilder.create<scf::YieldOp>(loc);
            });
      }
    } else {
      mlir::Value ptr = rewriter.create<vector::ExtractOp>(loc, ptrs, 0);
      auto updatedPtr = rewriter.create<IntToPtrOp>(loc, ptrTy, ptr);
      mlir::Type memRefTy = MemRefType::get(ptrShape, elemTy);
      mlir::Value memRef = rewriter.create<triton::cpu::PtrToMemRefOp>(
          loc, memRefTy, updatedPtr);
      auto cast =
          rewriter.create<UnrealizedConversionCastOp>(loc, memRefTy, vals)
              .getResult(0);
      auto lowerBound = rewriter.create<arith::ConstantIndexOp>(loc, 0);
      auto upperBound =
          rewriter.create<arith::ConstantIndexOp>(loc, ptrShape[0]);
      auto step = rewriter.create<arith::ConstantIndexOp>(loc, 1);
      auto forOp = rewriter.create<scf::ForOp>(
          loc, lowerBound, upperBound, step, ValueRange{},
          [&](OpBuilder &nestedBuilder, Location loc, mlir::Value iv,
              ValueRange) {
            auto constantValue =
                nestedBuilder.create<memref::LoadOp>(loc, cast, iv);
            nestedBuilder.create<memref::StoreOp>(loc, constantValue, memRef,
                                                  iv);
            nestedBuilder.create<scf::YieldOp>(loc);
          });
    }
    rewriter.eraseOp(op);
    return success();
  }
};

// This conversion pattern handles the transformation of `triton::LoadOp`
// operations. It converts the Triton LoadOp to SCF dialect operations by
// extracting the pointer, converting it to a MemRef type, and then loading the
// value from the MemRef. The result is cast to the appropriate vector type. The
// original Triton LoadOp is replaced with the new vector type loaded from
// memory.
struct LoadOpConversion : public OpConversionPattern<triton::LoadOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(triton::LoadOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    mlir::Value ptrs = rewriter.getRemappedValue(op.getPtr());
    llvm::ArrayRef<int64_t> ptrShape;
    mlir::Type pointeeType;
    mlir::Value updatedPtr, row, ptr, memRef, newIdx, zeroIdx;
    auto tensorTy = dyn_cast<RankedTensorType>(op.getPtr().getType());
    auto ptrTy = tensorTy.getElementType();
    if (auto elemType = dyn_cast<mlir::triton::PointerType>(ptrTy)) {
      pointeeType = elemType.getPointeeType();
    }
    mlir::Type resTy = getTypeConverter()->convertType(ptrs.getType());
    if (auto shapedType = dyn_cast<mlir::ShapedType>(ptrs.getType())) {
      ptrShape = shapedType.getShape();
    }
    auto memRefTy = MemRefType::get(ptrShape, pointeeType);
    if (ptrShape.size() == 2) {
      auto alloc = rewriter.create<memref::AllocOp>(loc, memRefTy);
      auto lowerBound1 = rewriter.create<arith::ConstantIndexOp>(loc, 0);
      auto upperBound1 =
          rewriter.create<arith::ConstantIndexOp>(loc, ptrShape[1]);
      auto step1 = rewriter.create<arith::ConstantIndexOp>(loc, 1);
      for (int i = 0; i < ptrShape[0]; i++) {
        row = rewriter.create<vector::ExtractOp>(loc, ptrs, i);
        ptr = rewriter.create<vector::ExtractOp>(loc, row, 0);
        updatedPtr = rewriter.create<IntToPtrOp>(loc, ptrTy, ptr);
        memRef = rewriter.create<triton::cpu::PtrToMemRefOp>(loc, memRefTy,
                                                             updatedPtr);
        rewriter.create<scf::ForOp>(
            loc, lowerBound1, upperBound1, step1, ValueRange{},
            [&](OpBuilder &nestedBuilder, Location loc, mlir::Value iv1,
                ValueRange) {
              newIdx = rewriter.create<arith::ConstantIndexOp>(loc, i);
              zeroIdx = rewriter.create<arith::ConstantIndexOp>(loc, 0);
              auto value = nestedBuilder.create<memref::LoadOp>(
                  loc, memRef, ValueRange{zeroIdx, iv1});
              nestedBuilder.create<memref::StoreOp>(loc, value, alloc,
                                                    ValueRange{newIdx, iv1});
              nestedBuilder.create<scf::YieldOp>(loc);
            });
      }
      auto vectorResultType = mlir::VectorType::get(ptrShape, pointeeType);
      auto cast = rewriter
                      .create<UnrealizedConversionCastOp>(loc, vectorResultType,
                                                          alloc.getResult())
                      .getResult(0);
      rewriter.replaceOp(op, cast);
    } else {
      mlir::Value ptr = rewriter.create<vector::ExtractOp>(loc, ptrs, 0);
      updatedPtr = rewriter.create<IntToPtrOp>(loc, ptrTy, ptr);
      mlir::Value memRef = rewriter.create<triton::cpu::PtrToMemRefOp>(
          loc, memRefTy, updatedPtr);
      auto vectorResultType = mlir::VectorType::get(ptrShape, pointeeType);
      auto cast =
          rewriter
              .create<UnrealizedConversionCastOp>(loc, vectorResultType, memRef)
              .getResult(0);
      rewriter.replaceOp(op, cast);
    }

    return success();
  }
};

// This conversion pattern handles the transformation of `arith::ConstantOp`
// operations. It converts constant values defined in tensor form into memory
// reference allocations. The pattern creates a memory allocation, initializes
// it with the constant values using an scf loop, and then converts the
// allocated memory reference to a vector type. The original arith::ConstantOp
// is replaced with the new vector type initialized with constants.
struct ConstantOpConversion : public OpConversionPattern<arith::ConstantOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(arith::ConstantOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    assert(isa<RankedTensorType>(op.getType()));
    llvm::ArrayRef<int64_t> shape;
    auto resTy = dyn_cast<mlir::VectorType>(
        getTypeConverter()->convertType(op.getType()));
    assert(resTy);
    auto loc = op.getLoc();
    if (auto shapedType = dyn_cast<mlir::ShapedType>(op.getType())) {
      shape = shapedType.getShape();
    }
    auto elemTy = resTy.getElementType();
    if (auto denseAttr = dyn_cast<DenseElementsAttr>(op.getValueAttr())) {
      auto memrefType = MemRefType::get(shape, elemTy);
      auto alloc = rewriter.create<memref::AllocOp>(loc, memrefType);
      if (shape.size() == 2) {
        auto lowerBound0 = rewriter.create<arith::ConstantIndexOp>(loc, 0);
        auto upperBound0 =
            rewriter.create<arith::ConstantIndexOp>(loc, shape[0]);
        auto step0 = rewriter.create<arith::ConstantIndexOp>(loc, 1);
        auto lowerBound1 = rewriter.create<arith::ConstantIndexOp>(loc, 0);
        auto upperBound1 =
            rewriter.create<arith::ConstantIndexOp>(loc, shape[1]);
        auto step1 = rewriter.create<arith::ConstantIndexOp>(loc, 1);
        if (denseAttr.isSplat()) {
          auto splatType = denseAttr.getElementType();
          mlir::Value newCst;
          if (splatType.isF32()) {
            auto splatValue = denseAttr.getSplatValue<float>();
            newCst = rewriter.create<arith::ConstantOp>(
                loc, rewriter.getF32Type(),
                rewriter.getF32FloatAttr(splatValue));
          } else if (splatType.isF64()) {
            auto splatValue = denseAttr.getSplatValue<double>();
            newCst = rewriter.create<arith::ConstantOp>(
                loc, rewriter.getF64Type(),
                rewriter.getF64FloatAttr(splatValue));
          } else {
            llvm::errs() << "Unsupported element type\n";
          }
          rewriter.create<scf::ForOp>(
              loc, lowerBound0, upperBound0, step0, ValueRange{},
              [&](OpBuilder &nestedBuilder, Location loc, mlir::Value iv0,
                  ValueRange) {
                nestedBuilder.create<scf::ForOp>(
                    loc, lowerBound1, upperBound1, step1, ValueRange{},
                    [&](OpBuilder &nestedBuilder, Location loc, mlir::Value iv1,
                        ValueRange) {
                      nestedBuilder.create<memref::StoreOp>(
                          loc, newCst, alloc, ValueRange{iv0, iv1});
                      nestedBuilder.create<scf::YieldOp>(loc);
                    });
                nestedBuilder.create<scf::YieldOp>(loc);
              });
        } else {
          return failure();
        }
      } else {
        auto lowerBound = rewriter.create<arith::ConstantIndexOp>(loc, 0);
        auto upperBound =
            rewriter.create<arith::ConstantIndexOp>(loc, shape[0]);
        auto step = rewriter.create<arith::ConstantIndexOp>(loc, 1);
        auto splatType = denseAttr.getElementType();
        mlir::Value newCst;
        if (splatType.isF32()) {
          auto splatValue = denseAttr.getSplatValue<float>();
          newCst = rewriter.create<arith::ConstantOp>(
              loc, rewriter.getF32Type(), rewriter.getF32FloatAttr(splatValue));
        } else if (splatType.isF64()) {
          auto splatValue = denseAttr.getSplatValue<double>();
          newCst = rewriter.create<arith::ConstantOp>(
              loc, rewriter.getF64Type(), rewriter.getF64FloatAttr(splatValue));
        } else {
          llvm::errs() << "Unsupported element type\n";
        }
        auto forOp = rewriter.create<scf::ForOp>(
            loc, lowerBound, upperBound, step, ValueRange{},
            [&](OpBuilder &nestedBuilder, Location loc, mlir::Value iv,
                ValueRange) {
              nestedBuilder.create<memref::StoreOp>(loc, newCst, alloc, iv);
              nestedBuilder.create<scf::YieldOp>(loc);
            });
      }

      auto vecType = mlir::VectorType::get(shape, elemTy);
      auto cast = rewriter
                      .create<UnrealizedConversionCastOp>(loc, vecType,
                                                          alloc.getResult())
                      .getResult(0);
      rewriter.replaceOp(op, cast);
    } else {
      llvm_unreachable("Unexpected constant attribute");
    }
    return success();
  }
};

// This pattern converts `arith::AddFOp` from tensor operands to memref
// allocations. It allocates memory, performs element-wise addition using a
// loop, stores the result, and casts the allocated memory to a vector type
// before replacing the original operation.
struct AddOpConversion : public OpConversionPattern<arith::AddFOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(arith::AddFOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto lhs = rewriter.getRemappedValue(op.getLhs());
    auto rhs = rewriter.getRemappedValue(op.getRhs());
    auto tensorTy = dyn_cast<RankedTensorType>(op.getLhs().getType());
    auto ptrTy = tensorTy.getElementType();
    llvm::ArrayRef<int64_t> shape;
    if (auto shapedType = dyn_cast<mlir::ShapedType>(op.getLhs().getType())) {
      shape = shapedType.getShape();
    }
    auto memRefTy = MemRefType::get(shape, ptrTy);
    auto alloc = rewriter.create<memref::AllocOp>(loc, memRefTy);
    auto castLhs =
        rewriter.create<UnrealizedConversionCastOp>(loc, memRefTy, lhs)
            .getResult(0);
    auto castRhs =
        rewriter.create<UnrealizedConversionCastOp>(loc, memRefTy, rhs)
            .getResult(0);
    auto lowerBound = rewriter.create<arith::ConstantIndexOp>(loc, 0);
    auto upperBound = rewriter.create<arith::ConstantIndexOp>(loc, shape[0]);
    auto step = rewriter.create<arith::ConstantIndexOp>(loc, 1);
    auto forOp = rewriter.create<scf::ForOp>(
        loc, lowerBound, upperBound, step, ValueRange{},
        [&](OpBuilder &nestedBuilder, Location loc, mlir::Value iv,
            ValueRange) {
          auto constantLhs =
              nestedBuilder.create<memref::LoadOp>(loc, castLhs, iv);
          auto constantRhs =
              nestedBuilder.create<memref::LoadOp>(loc, castRhs, iv);
          auto summationValue = nestedBuilder.create<arith::AddFOp>(
              loc, constantLhs, constantRhs);
          nestedBuilder.create<memref::StoreOp>(loc, summationValue, alloc, iv);
          nestedBuilder.create<scf::YieldOp>(loc);
        });
    auto vectorResultType = mlir::VectorType::get(shape, ptrTy);
    auto cast = rewriter
                    .create<UnrealizedConversionCastOp>(loc, vectorResultType,
                                                        alloc.getResult())
                    .getResult(0);
    rewriter.replaceOp(op, cast);
    return success();
  }
};

// `ReductionOpConversion` lowers `triton::ReduceOp` to an SCF loop for
// reduction. It converts the input tensor to a `MemRefType`, and sets up an SCF
// `for` loop that iterates over the reduction dimension (set to 0). The loop
// accumulates the sum of the tensor's elements using `memref::LoadOp` and
// `arith::AddFOp`, yielding the result at the end.
struct ReductionOpConversion : public OpConversionPattern<triton::ReduceOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(triton::ReduceOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    auto src = rewriter.getRemappedValue(op.getOperand(0));
    auto tensorTy = dyn_cast<RankedTensorType>(op.getOperand(0).getType());
    if (!tensorTy)
      return failure();
    auto ptrTy = tensorTy.getElementType();
    llvm::ArrayRef<int64_t> shape = tensorTy.getShape();
    auto memRefTy = MemRefType::get(shape, ptrTy);
    auto cast = rewriter.create<UnrealizedConversionCastOp>(loc, memRefTy, src)
                    .getResult(0);

    int64_t reduceDim = 0;
    int64_t reduceSize = shape[reduceDim];
    auto lowerBound = rewriter.create<arith::ConstantIndexOp>(loc, 0);
    auto upperBound = rewriter.create<arith::ConstantIndexOp>(loc, reduceSize);
    auto step = rewriter.create<arith::ConstantIndexOp>(loc, 1);
    mlir::Value sumInit =
        rewriter.create<arith::ConstantOp>(loc, rewriter.getF32FloatAttr(0.0));
    auto forOp = rewriter.create<scf::ForOp>(
        loc, lowerBound, upperBound, step, sumInit,
        [&](OpBuilder &builder, Location loc, mlir::Value iv,
            mlir::ValueRange iterArgs) {
          mlir::Value sum = iterArgs[0];
          auto loadValue = builder.create<memref::LoadOp>(loc, cast, iv);
          mlir::Value updatedSum =
              builder.create<arith::AddFOp>(loc, sum, loadValue);
          builder.create<scf::YieldOp>(loc, ValueRange{updatedSum});
        });
    rewriter.replaceOp(op, forOp.getResult(0));
    return success();
  }
};

namespace {
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
    patterns.add<AddOpConversion>(typeConverter, context);
    patterns.add<LoadOpConversion>(typeConverter, context);
    patterns.add<StoreOpConversion>(typeConverter, context);
    patterns.add<OpTypeConversion<arith::AddIOp>>(typeConverter, context);
    patterns.add<ConstantOpConversion>(typeConverter, context);
    patterns.add<OpTypeConversion<arith::MulFOp>>(typeConverter, context);
    patterns.add<OpTypeConversion<arith::MulIOp>>(typeConverter, context);
    patterns.add<OpTypeConversion<arith::CmpFOp>>(typeConverter, context);
    patterns.add<OpTypeConversion<arith::CmpIOp>>(typeConverter, context);
    patterns.add<ReductionOpConversion>(typeConverter, context);
    if (failed(applyPartialConversion(mod, convTarget, std::move(patterns))))
      return signalPassFailure();
  }
};

} // namespace

namespace mlir {
namespace triton {
namespace cpu {

std::unique_ptr<OperationPass<ModuleOp>> createLowerTritonToSCF() {
  return std::make_unique<LowerTritonToSCF>();
}

} // namespace cpu
} // namespace triton
} // namespace mlir
