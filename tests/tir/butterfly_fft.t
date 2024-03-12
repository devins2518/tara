pub const Top = module {
    pub comb top(x0: u8, x1: u8) struct { y0: u8, y1: u8 } {
        return .{ .y0 = x0 + x1, .y1 = x0 - x1 };
    }
};
