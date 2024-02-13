// RUN: @tara @file --dump-utir
// CHECK: 0

const S = struct {
    const A = bool;

    pub fn a(b: A, c: A) bool {
        return b and c;
    }
};
