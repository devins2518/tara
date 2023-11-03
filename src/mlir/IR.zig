const c = @import("../c.zig");

pub const MLIRContext = struct {
    inner: c.MlirContext,

    /// Creates an MLIR context and transfers its ownership to the caller.
    /// This sets the default multithreading option (enabled).
    pub fn init() MLIRContext {
        return .{ .innter = c.mlirContextCreate() };
    }
};
