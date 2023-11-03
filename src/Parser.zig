const std = @import("std");
const Allocator = std.mem.Allocator;
const tokenize = @import("std");
const Token = tokenize.Token;

allocator: Allocator,
buf: []const u8,
tokens: []const Token,
