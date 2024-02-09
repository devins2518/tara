// RUN: @tara @file --dump-ast
// CHECK: 0
// CHECK: 1

const S = struct {
    const A = bool;
    const B = struct {
        const C = u1;
    };

    d: u1,
};
