// RUN: @tara @file --dump-ast
// CHECK: 0
// CHECK: 1

const A = a + b * c - d / e;
