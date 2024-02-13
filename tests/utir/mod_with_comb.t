// RUN: @tara @file --dump-utir
// CHECK: 0

const Mod = module {
    const A = bool;

    pub comb a(b: A, c: A) bool {
        return b and c;
    }
};
