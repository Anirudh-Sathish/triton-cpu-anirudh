// RUN: triton-opt %s -split-input-file -lower-triton-to-scf | FileCheck %s
module {
  tt.func public @constant_store(%arg0: !tt.ptr<f32>, %arg1: !tt.ptr<f32>, %arg2: !tt.ptr<f32>, %arg3: i32) attributes {noinline = false} {
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
// CHECK-LABEL: constant_store
// CHECK: arith.constant 0 : index
// CHECK: arith.constant 16 : index
// CHECK: arith.constant 1 : index
// CHECK: arith.constant 3.000000e+00 : f32
// CHECK: scf.for {{.*}} to {{.*}} step {{.*}} {
// CHECK:   memref.store {{.*}} : memref<16xf32>
// CHECK: }
// CHECK: builtin.unrealized_conversion_cast {{.*}} : memref<16xf32> to vector<16xf32>
// CHECK: arith.constant 16 : i32
// CHECK: tt.get_program_id x : i32
// CHECK: arith.muli {{.*}} : i32
// CHECK: tt.make_range {{.*}} : tensor<16xi32>
// CHECK: builtin.unrealized_conversion_cast {{.*}} : tensor<16xi32> to vector<16xi32>
// CHECK: tt.splat {{.*}} : i32 -> tensor<16xi32>
// CHECK: builtin.unrealized_conversion_cast {{.*}} : tensor<16xi32> to vector<16xi32>
// CHECK: arith.addi {{.*}} : vector<16xi32>
// CHECK: builtin.unrealized_conversion_cast {{.*}} : vector<16xi32> to tensor<16xi32>
// CHECK: tt.splat {{.*}} : !tt.ptr<f32> -> tensor<16x!tt.ptr<f32>>
// CHECK: tt.addptr {{.*}} : tensor<16x!tt.ptr<f32>>, tensor<16xi32>
// CHECK: builtin.unrealized_conversion_cast {{.*}} : tensor<16x!tt.ptr<f32>> to vector<16xi64>
// CHECK: vector.extract {{.*}} : i64 from vector<16xi64>
// CHECK: tt.int_to_ptr {{.*}} : i64 -> !tt.ptr<f32>
// CHECK: triton_cpu.ptr_to_memref {{.*}} : <f32> -> memref<16xf32>
// CHECK: builtin.unrealized_conversion_cast {{.*}} : vector<16xf32> to memref<16xf32>
// CHECK: scf.for {{.*}} to {{.*}} step {{.*}} {
// CHECK:   memref.load {{.*}} : memref<16xf32>
// CHECK:   memref.store {{.*}} : memref<16xf32>
// CHECK: }
// CHECK: tt.return