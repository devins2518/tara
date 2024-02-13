// RUN: @tara @file --dump-utir
// CHECK: 0

const Mod = module {
    const A = bool;
    const B = module {
        const C = u1;
    };

    d: &u1,
};
