// RUN: @tara @file --dump-tir
// CHECK: 

const Mod = module {
    pub comb add(a: &u1, b: &u1) u1 {
        return a ^ b;
    }
};

const Top = module {
    const A = bool;

    pub comb a(b: A, c: A) bool {
        return b and c;
    }

    pub comb top() void {
        const b = Mod{};
        b.add();
    }
};
