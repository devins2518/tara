// RUN: @tara @file
// CHECK: 0
// CHECK: 1

const Mod = module {
    const A = bool;

    pub comb a(b: A, c: A) bool {
        return b and c;
    }
}
