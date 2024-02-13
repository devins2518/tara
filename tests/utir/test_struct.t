// RUN: @tara @file --dump-utir
// CHECK: 0

const S = struct {
    const A = bool;
    const B = struct {
        const C = u1;
    };

    d: u1,
};
