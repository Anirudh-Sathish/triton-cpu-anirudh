// RUN: triton-opt %s -split-input-file -lower-triton-to-scf | FileCheck %s
module {
  tt.func public @add_kernel(%arg0: !tt.ptr<f32>, %arg1: !tt.ptr<f32>, %arg2: !tt.ptr<f32>, %arg3: i32) attributes {noinline = false} {
    %cst = arith.constant dense<3.000000e+00> : tensor<16xf32>
    %c16_i32 = arith.constant 16 : i32
    %0 = tt.get_program_id x : i32 
    %1 = arith.muli %0, %c16_i32 : i32 
    %2 = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32> 
    %3 = tt.splat %1 : i32 -> tensor<16xi32> 
    %4 = arith.addi %3, %2 : tensor<16xi32> 
    %5 = tt.splat %arg2 : !tt.ptr<f32> -> tensor<16x!tt.ptr<f32>> 
    %6 = tt.addptr %5, %4 : tensor<16x!tt.ptr<f32>>, tensor<16xi32> 
    tt.store %6, %cst : tensor<16x!tt.ptr<f32>> 
    tt.return 
  } 
} 
// CHECK-LABEL: add_kernel
// CHECK: arith.constant dense<3.000000e+00> : tensor<16xf32>
// CHECK: arith.constant 16 : i32
// CHECK: tt.get_program_id x : i32
// CHECK: arith.muli {{.*}} : i32
// CHECK: memref.alloc() : memref<16xi64>
// CHECK: scf.for {{.*}} to {{.*}} step {{.*}} {
// CHECK:   arith.constant 0 : i64
// CHECK:   arith.index_cast {{.*}} : index to i64
// CHECK:   arith.addi {{.*}} : i64
// CHECK:   memref.store {{.*}} : memref<16xi64>
// CHECK: }
// CHECK: builtin.unrealized_conversion_cast {{.*}} : memref<16xi64> to tensor<16xi32>
// CHECK: memref.alloc() : memref<16xi32>
// CHECK: scf.for {{.*}} to {{.*}} step {{.*}} {
// CHECK:   memref.store {{.*}} : memref<16xi32>
// CHECK: }
// CHECK: builtin.unrealized_conversion_cast {{.*}} : memref<16xi32> to tensor<16xi32>
// CHECK: arith.addi {{.*}} : tensor<16xi32>
// CHECK: memref.alloc() : memref<16xf32>
// CHECK: scf.for {{.*}} to {{.*}} step {{.*}} {
// CHECK:   builtin.unrealized_conversion_cast {{.*}} : !tt.ptr<f32> to f32
// CHECK:   memref.store {{.*}} : memref<16xf32>
// CHECK: }
// CHECK: builtin.unrealized_conversion_cast {{.*}} : memref<16xf32> to tensor<16x!tt.ptr<f32>>
// CHECK: tt.addptr {{.*}} : tensor<16x!tt.ptr<f32>>, tensor<16xi32>
// CHECK: builtin.unrealized_conversion_cast {{.*}} : tensor<16x!tt.ptr<f32>> to memref<16xf32>
// CHECK: scf.for {{.*}} to {{.*}} step {{.*}} {
// CHECK:   memref.load {{.*}} : memref<16xf32>
// CHECK:   memref.store {{.*}} : memref<16xf32>
// CHECK: }
// CHECK: tt.return
