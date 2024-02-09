// RUN: @tara @file --dump-ast
// CHECK: 0
// CHECK: 1

const Mod = module {
    const A = bool;
    const B = module {
        const C = u1;
    };

    d: &u1,
};
