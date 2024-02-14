// RUN: @tara @file --dump-utir
// CHECK: test

const S = struct {
    const A = bool;

    pub fn a(b: A, c: A) bool {
        return b and c;
    }
};
