const std = @import("std");
const assert = std.debug.assert;
const c = @import("c.zig");

//===----------------------------------------------------------------------===//
// MlirStringRef.
//===----------------------------------------------------------------------===//
/// A pointer to a sized fragment of a string, not necessarily null-terminated.
/// Does not own the underlying string. This is equivalent to llvm::StringRef.
pub const StringRef = struct {
    _: c.MlirStringRef,

    pub fn fromRaw(raw: c.MlirStringRef) StringRef {
        assert(raw.data != null);
        return .{ ._ = raw };
    }

    pub fn getRaw(str: StringRef) c.MlirStringRef {
        return str._;
    }

    /// Constructs a string reference from the pointer and length. The pointer need
    /// not reference to a null-terminated string.
    pub fn init(str: []const u8) StringRef {
        return StringRef.fromRaw(c.mlirStringRefCreate(str.ptr, str.len));
    }

    /// Constructs a string reference from a null-terminated C string. Prefer
    /// mlirStringRefCreate if the length of the string is known.
    pub fn initRaw(str: [*:0]const u8, len: usize) StringRef {
        return StringRef.fromRaw(c.mlirStringRefCreate(str, len));
    }

    /// Returns true if two string references are equal, false otherwise.
    pub fn eql(string: StringRef, other: StringRef) bool {
        return c.mlirStringRefEqual(string.getRaw(), other.getRaw());
    }

    /// A callback for returning string references.
    ///
    /// This function is called back by the functions that need to return a
    /// reference to the portion of the string with the following arguments:
    ///  - an MlirStringRef representing the current portion of the string
    ///  - a pointer to user data forwarded from the printing call.
    pub const StringCallback = c.MlirStringCallback;
};

//===----------------------------------------------------------------------===//
// MlirLogicalResult.
//===----------------------------------------------------------------------===//
/// A logical result value, essentially a boolean with named states. LLVM
/// convention for using boolean values to designate success or failure of an
/// operation is a moving target, so MLIR opted for an explicit class.
/// Instances of MlirLogicalResult must only be inspected using the associated
/// functions.
pub const LogicalResult = struct {
    _: c.MlirLogicalResult,

    pub fn fromRaw(raw: c.MlirLogicalResult) LogicalResult {
        return .{ ._ = raw };
    }

    pub fn getRaw(res: LogicalResult) c.MlirLogicalResult {
        return res._;
    }

    /// Checks if the given logical result represents a success.
    pub fn isSuccess(res: LogicalResult) bool {
        return c.mlirLogicalResultIsSuccess(res.getRaw());
    }

    /// Checks if the given logical result represents a failure.
    pub fn isFailure(res: LogicalResult) bool {
        return c.mlirLogicalResultIsFailure(res.getRaw());
    }
    /// Creates a logical result representing a success.
    pub fn initSuccess() LogicalResult {
        return LogicalResult.fromRaw(c.mlirLogicalResultSuccess());
    }

    /// Creates a logical result representing a failure.
    pub fn initFailure() LogicalResult {
        return LogicalResult.fromRaw(c.mlirLogicalResultFailure());
    }
};

//===----------------------------------------------------------------------===//
// MlirLlvmThreadPool.
//===----------------------------------------------------------------------===//
pub const LlvmThreadPool = struct {
    _: c.MlirLlvmThreadPool,

    pub fn fromRaw(raw: c.MlirLlvmThreadPool) LlvmThreadPool {
        return .{ ._ = raw };
    }

    pub fn getRaw(pool: LlvmThreadPool) c.MlirLlvmThreadPool {
        return pool._;
    }

    /// Create an LLVM thread pool. This is reexported here to avoid directly
    /// pulling in the LLVM headers directly.
    pub fn init() LlvmThreadPool {
        return LlvmThreadPool.fromRaw(c.mlirLlvmThreadPoolCreate());
    }

    /// Destroy an LLVM thread pool.
    pub fn deinit(pool: LlvmThreadPool) void {
        c.mlirLlvmThreadPoolDestroy(pool.getRaw());
    }
};

//===----------------------------------------------------------------------===//
// TypeID API.
//===----------------------------------------------------------------------===//
pub const TypeID = struct {
    _: c.MlirTypeID,

    pub fn fromRaw(raw: c.MlirTypeID) TypeID {
        return .{ ._ = raw };
    }

    pub fn getRaw(type_id: TypeID) c.MlirTypeID {
        return type_id._;
    }

    /// `ptr` must be 8 byte aligned and unique to a type valid for the duration of
    /// the returned type id's usage
    pub fn init(ptr: *align(8) anyopaque) TypeID {
        return TypeID.fromRaw(c.mlirTypeIDCreate(ptr));
    }

    /// Checks whether a type id is null.
    pub fn isNull(type_id: TypeID) bool {
        return c.mlirTypeIDIsNull(type_id.getRaw());
    }

    /// Checks if two type ids are equal.
    pub fn eql(type_id: TypeID, other: TypeID) bool {
        return c.mlirTypeIDEqual(type_id.getRaw(), other.getRaw());
    }

    /// Returns the hash value of the type id.
    pub fn hash(type_id: TypeID) usize {
        return c.mlirTypeIDHashValue(type_id.getRaw());
    }
};

//===----------------------------------------------------------------------===//
// TypeIDAllocator API.
//===----------------------------------------------------------------------===//
pub const TypeIDAllocator = struct {
    _: c.MlirTypeIDAllocator,

    pub fn fromRaw(raw: c.MlirTypeIDAllocator) TypeIDAllocator {
        return .{ ._ = raw };
    }

    pub fn getRaw(allocator: TypeIDAllocator) c.MlirTypeIDAllocator {
        return allocator._;
    }

    /// Creates a type id allocator for dynamic type id creation
    pub fn init() TypeIDAllocator {
        return TypeIDAllocator.fromRaw(c.mlirTypeIDAllocatorCreate());
    }

    /// Deallocates the allocator and all allocated type ids
    pub fn deinit(allocator: TypeIDAllocator) void {
        c.mlirTypeIDAllocatorDestroy(allocator.getRaw());
    }

    /// Allocates a type id that is valid for the lifetime of the allocator
    pub fn allocateTypeId(allocator: TypeIDAllocator) TypeID {
        return TypeID.fromRaw(c.mlirTypeIDAllocatorAllocateTypeID(allocator.getRaw()));
    }
};
