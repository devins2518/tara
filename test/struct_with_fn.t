const S = struct {
    const A = bool;

    pub fn a(b: A, c: A) bool {
        return b and c;
    }
}
