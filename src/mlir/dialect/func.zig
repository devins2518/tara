const std = @import("std");
const c = @import("../c.zig");
const ir = @import("../ir.zig");
const support = @import("../support.zig");
const Context = ir.Context;
const Location = ir.Location;
const Attribute = ir.Attribute;
const Operation = ir.Operation;
const OperationState = ir.OperationState;
const NamedAttribute = ir.NamedAttribute;
const StringRef = support.StringRef;

pub const Call = struct {
    op: Operation,
};
