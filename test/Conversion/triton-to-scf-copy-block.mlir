// RUN: triton-opt %s -split-input-file -lower-triton-to-scf | FileCheck %s
module {
  tt.func public @copy_kernel(%arg0: !tt.ptr<f32>, %arg1: !tt.ptr<f32>) attributes {noinline = false} {
    %c16_i32 = arith.constant 16 : i32 
    %0 = tt.get_program_id x : i32 
    %1 = arith.muli %0, %c16_i32 : i32 
    %2 = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32> 
    %3 = tt.splat %1 : i32 -> tensor<16xi32> 
    %4 = arith.addi %3, %2 : tensor<16xi32> 
    %5 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<16x!tt.ptr<f32>> 
    %6 = tt.addptr %5, %4 : tensor<16x!tt.ptr<f32>>, tensor<16xi32> 
    %7 = tt.load %6 : tensor<16x!tt.ptr<f32>> 
    %8 = tt.splat %arg1 : !tt.ptr<f32> -> tensor<16x!tt.ptr<f32>> 
    %9 = tt.addptr %8, %4 : tensor<16x!tt.ptr<f32>>, tensor<16xi32> 
    tt.store %9, %7 : tensor<16x!tt.ptr<f32>> 
    tt.return 
  }
} 

// CHECK-LABEL: copy_kernel
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
// CHECK: builtin.unrealized_conversion_cast {{.*}} : memref<16xf32> to vector<16xf32>
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
