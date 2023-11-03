pub fn DefineZigStruct(storage: type) type {
    return extern struct { ptr: ?*storage };
}

pub const MlirLlvmThreadPool = DefineZigStruct(anyopaque);
pub const MlirTypeID = DefineZigStruct(anyopaque);
pub const MlirTypeIDAllocator = DefineZigStruct(anyopaque);

//===----------------------------------------------------------------------===//
// MlirStringRef.
//===----------------------------------------------------------------------===//

/// A pointer to a sized fragment of a string, not necessarily null-terminated.
/// Does not own the underlying string. This is equivalent to llvm::StringRef.
pub const MlirStringRef = extern struct {
    data: [*c]u8,
    length: usize,

    /// Constructs a string reference from the pointer and length. The pointer need
    /// not reference to a null-terminated string.
    pub inline fn mlirStringRefCreate(str: [*]u8, length: usize) MlirStringRef {
        return .{ .data = str, .length = length };
    }

    /// Constructs a string reference from a null-terminated C string. Prefer
    /// mlirStringRefCreate if the length of the string is known.
    pub extern fn mlirStringRefCreateFromCString(str: [*c]u8) MlirStringRef;

    /// Returns true if two string references are equal, false otherwise.
    pub extern fn mlirStringRefEqual(string: MlirStringRef, other: MlirStringRef) bool;

    /// A callback for returning string references.
    ///
    /// This function is called back by the functions that need to return a
    /// reference to the portion of the string with the following arguments:
    ///  - an MlirStringRef representing the current portion of the string
    ///  - a pointer to user data forwarded from the printing call.
    pub const MlirStringCallback = *const fn (*anyopaque) void;
};

//===----------------------------------------------------------------------===//
// MlirLogicalResult.
//===----------------------------------------------------------------------===//

/// A logical result value, essentially a boolean with named states. LLVM
/// convention for using boolean values to designate success or failure of an
/// operation is a moving target, so MLIR opted for an explicit class.
/// Instances of MlirLogicalResult must only be inspected using the associated
/// functions.
pub const MlirLogicalResult = extern struct {
    value: i8,

    /// Checks if the given logical result represents a success.
    pub inline fn mlirLogicalResultIsSuccess(res: MlirLogicalResult) bool {
        return res.value != 0;
    }

    /// Checks if the given logical result represents a failure.
    pub inline fn mlirLogicalResultIsFailure(res: MlirLogicalResult) bool {
        return res.value == 0;
    }

    /// Creates a logical result representing a success.
    pub inline fn mlirLogicalResultSuccess() MlirLogicalResult {
        return .{ .value = 1 };
    }

    /// Creates a logical result representing a failure.
    pub inline fn mlirLogicalResultFailure() MlirLogicalResult {
        return .{ .value = 0 };
    }
};

//===----------------------------------------------------------------------===//
// MlirLlvmThreadPool.
//===----------------------------------------------------------------------===//

/// Create an LLVM thread pool. This is reexported here to avoid directly
/// pulling in the LLVM headers directly.
pub extern fn mlirLlvmThreadPoolCreate(void) MlirLlvmThreadPool;

/// Destroy an LLVM thread pool.
pub extern fn mlirLlvmThreadPoolDestroy(pool: MlirLlvmThreadPool) void;

//===----------------------------------------------------------------------===//
// TypeID API.
//===----------------------------------------------------------------------===//

/// `ptr` must be 8 byte aligned and unique to a type valid for the duration of
/// the returned type id's usage
pub extern fn mlirTypeIDCreate(ptr: *const anyopaque) MlirTypeID;

/// Checks whether a type id is null.
pub inline fn mlirTypeIDIsNull(typeID: MlirTypeID) bool {
    return !typeID.ptr;
}

/// Checks if two type ids are equal.
pub extern fn mlirTypeIDEqual(typeID1: MlirTypeID, typeID2: MlirTypeID) bool;

/// Returns the hash value of the type id.
pub extern fn mlirTypeIDHashValue(typeID: MlirTypeID) usize;

//===----------------------------------------------------------------------===//
// TypeIDAllocator API.
//===----------------------------------------------------------------------===//

/// Creates a type id allocator for dynamic type id creation
pub extern fn mlirTypeIDAllocatorCreate(void) MlirTypeIDAllocator;

/// Deallocates the allocator and all allocated type ids
pub extern fn mlirTypeIDAllocatorDestroy(allocator: MlirTypeIDAllocator) void;

/// Allocates a type id that is valid for the lifetime of the allocator
pub extern fn mlirTypeIDAllocatorAllocateTypeID(allocator: MlirTypeIDAllocator) MlirTypeID;
