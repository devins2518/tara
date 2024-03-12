const Top = module {
    pub comb top(clk: clock, nrst: reset, max: u8) struct { out: u8, rollover: bool } {
        const out: u8 = @reg(clk, reset, .{ .reset = 0 });
        out.next = if (out == max) 0 else out + 1;
        return .{ .out = out, .rollover = out == max };
    }
};
