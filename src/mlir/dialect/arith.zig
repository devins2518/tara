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

pub fn dialect() ir.DialectHandle {
    return ir.DialectHandle.fromRaw(c.mlirGetDialectHandle__arith__());
}

pub const Constant = struct {
    op: ir.Operation,

    pub fn init(context: Context, value: Attribute, location: Location) Constant {
        var state = OperationState.get("arith.constant", location);
        state.addAttributes(&.{NamedAttribute.initFromNameAndAttr(context, "value", value)});
        state.enableResultTypeInference();
        return Constant{ .op = Operation.init(&state) };
    }
};

test {
    std.testing.refAllDeclsRecursive(@This());
}
