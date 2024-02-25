// RUN: @tara @file --dump-tir
// CHECK: 

const Top = module {
    const A = bool;

    pub comb a(b: A, c: A) bool {
        return b and c;
    }

    pub comb top() void {
        const a = Top{};
        a.a();
    }
};
