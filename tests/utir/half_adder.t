// RUN: @tara @file --dump-utir
// CHECK: 0

const HalfAdder = module {
    pub comb sum(a: &u1, b: &u1) u1 {
        return a ^ b;
    }
    pub comb carry(a: &u1, b: &u1) u1 {
        return a & b;
    }
};
