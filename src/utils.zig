pub const Location = struct {
    start: usize,
    end: usize,
};

pub fn unionPayloadPtr(comptime T: type, union_ptr: anytype) ?T {
    const U = @typeInfo(@TypeOf(union_ptr));
    inline for (U.Union.fields, 0..) |field, i| {
        if (field.type != T)
            continue;
        if (@intFromEnum(union_ptr) == i)
            return @field(union_ptr, field.name);
    }
    return null;
}

pub fn u32s(comptime T: type) comptime_int {
    return @sizeOf(T) / @sizeOf(u32);
}
