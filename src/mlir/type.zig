const std = @import("std");
const c = @import("c.zig");
const attribute = @import("attribute.zig");
const ir = @import("ir.zig");
const Attribute = ir.Attribute;
const Context = ir.Context;
const Location = ir.Location;
const NamedAttribute = ir.NamedAttribute;
const Type = ir.Type;
const support = @import("support.zig");
const StringRef = support.StringRef;
const TypeID = support.TypeID;

//===----------------------------------------------------------------------===//
// Integer types.
//===----------------------------------------------------------------------===//
const IntegerType = struct {
    type: Type,

    /// Creates a signless integer type of the given bitwidth in the context. The
    /// type is owned by the context.
    pub fn initSignless(ctx: Context, bitwidth: u16) IntegerType {
        return .{ .type = c.mlirIntegerTypeGet(ctx.getRaw(), bitwidth) };
    }

    /// Creates a signed integer type of the given bitwidth in the context. The type
    /// is owned by the context.
    pub fn initSigned(ctx: Context, bitwidth: u16) IntegerType {
        return .{ .type = c.mlirIntegerTypeSignedGet(ctx.getRaw(), bitwidth) };
    }

    /// Creates an unsigned integer type of the given bitwidth in the context. The
    /// type is owned by the context.
    pub fn initUnsigned(ctx: Context, bitwidth: u16) IntegerType {
        return .{ .type = c.mlirIntegerTypeUnsignedGet(ctx.getRaw(), bitwidth) };
    }

    /// Returns the bitwidth of an integer type.
    pub fn getWidth(@"type": IntegerType) u16 {
        return c.mlirIntegerTypeGetWidth(@"type".type.getRaw());
    }

    /// Checks whether the given integer type is signless.
    pub fn isSignless(@"type": IntegerType) bool {
        return c.mlirIntegerTypeIsSignless(@"type".type.getRaw());
    }

    /// Checks whether the given integer type is signed.
    pub fn isSigned(@"type": IntegerType) bool {
        return c.mlirIntegerTypeIsSigned(@"type".type.getRaw());
    }

    /// Checks whether the given integer type is unsigned.
    pub fn isUnsigned(@"type": IntegerType) bool {
        return c.mlirIntegerTypeIsUnsigned(@"type".type.getRaw());
    }

    /// Returns the typeID of an Integer type.
    pub fn getTypeID() TypeID {
        return TypeID.fromRaw(c.mlirIntegerTypeGetTypeID());
    }
};

//===----------------------------------------------------------------------===//
// Index type.
//===----------------------------------------------------------------------===//
const IndexType = struct {
    type: Type,

    /// Creates an index type in the given context. The type is owned by the
    /// context.
    pub fn init(ctx: Context) IndexType {
        return .{ .type = c.mlirIndexTypeGet(ctx.getRaw()) };
    }

    /// Returns the typeID of an Index type.
    pub fn getTypeID() TypeID {
        return TypeID.fromRaw(c.mlirIndexTypeGetTypeID());
    }
};

//===----------------------------------------------------------------------===//
// Floating-point types.
//===----------------------------------------------------------------------===//
const FloatType = struct {
    type: Type,

    pub const Float8E4M3B11FNUZType = struct {
        type: Type,

        /// Creates an f8E4M3B11FNUZ type in the given context. The type is owned by the
        /// context.
        pub fn init(ctx: Context) Float8E4M3B11FNUZType {
            return .{ .type = c.mlirFloat8E5M2TypeGet(ctx.getRaw()) };
        }

        /// Returns the typeID of an Float8E4M3B11FNUZ type.
        pub fn getTypeID() TypeID {
            return TypeID.fromRaw(c.mlirFloat8E4M3B11FNUZTypeGetTypeID());
        }
    };

    pub const Float8E4M3FNType = struct {
        type: Type,

        /// Creates an f8E4M3FN type in the given context. The type is owned by the
        /// context.
        pub fn init(ctx: Context) Float8E4M3FNType {
            return .{ .type = Type.fromRaw(c.mlirFloat8E4M3FNTypeGet(ctx.getRaw())) };
        }

        /// Returns the typeID of an Float8E4M3FN type.
        pub fn getTypeID() TypeID {
            return TypeID.fromRaw(c.mlirFloat8E4M3FNTypeGetTypeID());
        }
    };

    pub const Float8E4M3FNUZType = struct {
        type: Type,

        /// Creates an f8E4M3FNUZ type in the given context. The type is owned by the
        /// context.
        pub fn init(ctx: Context) Float8E4M3FNUZType {
            return .{ .type = Type.fromRaw(c.mlirFloat8E4M3FNUZTypeGet(ctx.getRaw())) };
        }

        /// Returns the typeID of an Float8E4M3FNUZ type.
        pub fn getTypeID() TypeID {
            return TypeID.fromRaw(c.mlirFloat8E4M3FNUZTypeGetTypeID());
        }
    };

    pub const Float8E5M2Type = struct {
        type: Type,

        /// Creates an f8E5M2 type in the given context. The type is owned by the
        /// context.
        pub fn init(ctx: Context) Float8E5M2Type {
            return .{ .type = c.mlirFloat8E5M2TypeGet(ctx.getRaw()) };
        }

        /// Returns the typeID of an Float8E5M2 type.
        pub fn getTypeID() TypeID {
            return TypeID.fromRaw(c.mlirFloat8E5M2TypeGetTypeID());
        }
    };

    pub const Float8E5M2FNUZType = struct {
        type: Type,

        /// Creates an f8E5M2FNUZ type in the given context. The type is owned by the
        /// context.
        pub fn init(ctx: Context) Float8E5M2FNUZType {
            return .{ .type = c.mlirFloat8E5M2FNUZTypeGet(ctx.getRaw()) };
        }

        /// Returns the typeID of an Float8E5M2FNUZ type.
        pub fn getTypeID() TypeID {
            return TypeID.fromRaw(c.mlirFloat8E5M2FNUZTypeGetTypeID());
        }
    };

    pub const Float16Type = struct {
        type: Type,

        /// Creates an f16 type in the given context. The type is owned by the
        /// context.
        pub fn init(ctx: Context) Float16Type {
            return .{ .type = Type.fromRaw(c.mlirF16TypeGet(ctx.getRaw())) };
        }

        /// Returns the typeID of an Float16 type.
        pub fn getTypeID() TypeID {
            return c.mlirFloat16TypeGetTypeID();
        }
    };

    pub const Float32Type = struct {
        type: Type,

        /// Creates an f32 type in the given context. The type is owned by the
        /// context.
        pub fn init(ctx: Context) Float32Type {
            return .{ .type = Type.fromRaw(c.mlirF32TypeGet(ctx.getRaw())) };
        }

        /// Returns the typeID of an Float32 type.
        pub fn getTypeID() TypeID {
            return c.mlirFloat32TypeGetTypeID();
        }
    };

    pub const Float64Type = struct {
        type: Type,

        /// Creates a f64 type in the given context. The type is owned by the
        /// context.
        pub fn init(ctx: Context) Float64Type {
            return .{ .type = Type.fromRaw(c.mlirF64TypeGet(ctx.getRaw())) };
        }

        /// Returns the typeID of an Float64 type.
        pub fn getTypeID() TypeID {
            return c.mlirFloat64TypeGetTypeID();
        }
    };

    pub const FloatTF32Type = struct {
        type: Type,

        /// Creates a TF32 type in the given context. The type is owned by the
        /// context.
        pub fn init(ctx: Context) FloatTF32Type {
            return .{ .type = Type.fromRaw(c.mlirTF32TypeGet(ctx.getRaw())) };
        }

        /// Returns the typeID of a TF32 type.
        pub fn getTypeID() TypeID {
            return c.mlirFloatTF32TypeGetTypeID();
        }
    };

    pub const BFloat16Type = struct {
        type: Type,

        /// Creates a bf16 type in the given context. The type is owned by the
        /// context.
        pub fn init(ctx: Context) BFloat16Type {
            return .{ .type = Type.fromRaw(c.mlirBF16TypeGet(ctx.getRaw())) };
        }

        /// Returns the typeID of an BFloat16 type.
        pub fn getTypeID() TypeID {
            return c.mlirBFloat16TypeGetTypeID();
        }
    };
};

//===----------------------------------------------------------------------===//
// None type.
//===----------------------------------------------------------------------===//
const NoneType = struct {
    type: Type,

    /// Creates a None type in the given context. The type is owned by the
    /// context.
    pub fn init(ctx: Context) IndexType {
        return .{ .type = c.mlirNoneTypeGet(ctx.getRaw()) };
    }

    /// Returns the typeID of an None type.
    pub fn getTypeID() TypeID {
        return TypeID.fromRaw(c.mlirNoneTypeGetTypeID());
    }
};

//===----------------------------------------------------------------------===//
// Complex type.
//===----------------------------------------------------------------------===//
const ComplexType = struct {
    type: Type,

    /// Creates a complex type with the given element type in the same context as
    /// the element type. The type is owned by the context.
    pub fn init(element_type: Type) ComplexType {
        return .{ .type = c.mlirComplexTypeGet(element_type.getRaw()) };
    }

    /// Returns the element type of the given complex type.
    pub fn getElementType(@"type": ComplexType) Type {
        return Type.fromRaw(c.mlirComplexTypeGetElementType(@"type".type.getRaw()));
    }

    /// Returns the typeID of an Complex type.
    pub fn getTypeID() TypeID {
        return TypeID.fromRaw(c.mlirComplexTypeGetTypeID());
    }
};

//===----------------------------------------------------------------------===//
// Shaped type.
//===----------------------------------------------------------------------===//
const ShapedType = struct {
    type: Type,

    /// Returns the element type of the shaped type.
    pub fn getElementType(@"type": ShapedType) Type {
        return Type.fromRaw(c.mlirShapedTypeGetElementType(@"type".type.getRaw()));
    }

    /// Checks whether the given shaped type is ranked.
    pub fn hasRank(@"type": ShapedType) bool {
        return c.mlirShapedTypeHasRank(@"type".type.getRaw());
    }

    /// Returns the rank of the given ranked shaped type.
    pub fn getRank(@"type": ShapedType) i64 {
        return c.mlirShapedTypeGetRank(@"type".type.getRaw());
    }

    /// Checks whether the given shaped type has a static shape.
    pub fn hasStaticShape(@"type": ShapedType) bool {
        return c.mlirShapedTypeHasStaticShape(@"type".type.getRaw());
    }

    /// Checks wither the dim-th dimension of the given shaped type is dynamic.
    pub fn isDynamicDim(@"type": ShapedType, dim: isize) bool {
        return c.mlirShapedTypeIsDynamicDim(@"type".type.getRaw(), dim);
    }

    /// Returns the dim-th dimension of the given ranked shaped type.
    pub fn getDimSize(@"type": ShapedType, dim: isize) i64 {
        return c.mlirShapedTypeGetDimSize(@"type".type.getRaw(), dim);
    }

    /// Checks whether the given value is used as a placeholder for dynamic sizes
    /// in shaped types.
    pub fn isDynamicSize(size: i64) bool {
        return c.mlirShapedTypeIsDynamicSize(size);
    }

    /// Returns the value indicating a dynamic size in a shaped type. Prefer
    /// mlirShapedTypeIsDynamicSize to direct comparisons with this value.
    pub fn getDynamicSize() i64 {
        return c.mlirShapedTypeGetDynamicSize();
    }

    /// Checks whether the given value is used as a placeholder for dynamic strides
    /// and offsets in shaped types.
    pub fn isDynamicStrideOrOffset(val: i64) bool {
        return c.mlirShapedTypeIsDynamicSize(val);
    }

    /// Returns the value indicating a dynamic stride or offset in a shaped type.
    /// Prefer mlirShapedTypeGetDynamicStrideOrOffset to direct comparisons with
    /// this value.
    pub fn getDynamicStrideOrOffset() i64 {
        return c.mlirShapedTypeGetDynamicSize();
    }
};

//===----------------------------------------------------------------------===//
// Vector type.
//===----------------------------------------------------------------------===//
const VectorType = struct {
    type: Type,

    /// Creates a vector type of the shape identified by its rank and dimensions,
    /// with the given element type in the same context as the element type. The
    /// type is owned by the context.
    pub fn init(shape: []const i64, element_type: Type) VectorType {
        return .{ .type = c.mlirVectorTypeGet(shape.len, shape.ptr, element_type.getRaw()) };
    }

    /// Same as "mlirVectorTypeGet" but returns a nullptr wrapping MlirType on
    /// illegal arguments, emitting appropriate diagnostics.
    pub fn initChecked(loc: Location, shape: []const i64, element_type: Type) VectorType {
        return .{ .type = c.mlirVectorTypeGetChecked(
            loc.getRaw(),
            shape.len,
            shape.ptr,
            element_type.getRaw(),
        ) };
    }

    /// Returns the typeID of an Vector type.
    pub fn getTypeID() TypeID {
        return TypeID.fromRaw(c.mlirVectorTypeGetTypeID());
    }
};

//===----------------------------------------------------------------------===//
// Ranked / Unranked Tensor type.
//===----------------------------------------------------------------------===//
const TensorType = struct {
    type: Type,

    /// Creates a tensor type of a fixed rank with the given shape, element type,
    /// and optional encoding in the same context as the element type. The type is
    /// owned by the context. Tensor types without any specific encoding field
    /// should assign mlirAttributeGetNull() to this parameter.
    pub fn rankedInit(shape: []const i64, element_type: Type, encoding: Attribute) TensorType {
        return .{ .type = c.mlirRankedTensorTypeGet(
            shape.len,
            shape.ptr,
            element_type.getRaw(),
            encoding.getRaw(),
        ) };
    }

    /// Same as "mlirRankedTensorTypeGet" but returns a nullptr wrapping MlirType on
    /// illegal arguments, emitting appropriate diagnostics.
    pub fn rankedInitChecked(loc: Location, shape: []const i64, element_type: Type, encoding: Attribute) TensorType {
        return .{ .type = c.mlirRankedTensorTypeGetChecked(
            loc.getRaw(),
            shape.len,
            shape.ptr,
            element_type.getRaw(),
            encoding.getRaw(),
        ) };
    }

    /// Gets the 'encoding' attribute from the ranked tensor type, returning a null
    /// attribute if none.
    pub fn rankedGetEncoding(@"type": TensorType) Attribute {
        return Attribute.fromRaw(c.mlirRankedTensorTypeGetEncoding(@"type".type.getRaw()));
    }

    /// Creates an unranked tensor type with the given element type in the same
    /// context as the element type. The type is owned by the context.
    pub fn unrankedInit(element_type: Type) TensorType {
        return .{ .type = c.mlirUnrankedTensorTypeGet(element_type.getRaw()) };
    }

    /// Same as "mlirUnrankedTensorTypeGet" but returns a nullptr wrapping MlirType
    /// on illegal arguments, emitting appropriate diagnostics.
    pub fn unrankedInitChecked(loc: Location, element_type: Type) TensorType {
        return .{ .type = c.mlirUnrankedTensorTypeGetChecked(
            loc.getRaw(),
            element_type.getRaw(),
        ) };
    }

    /// Returns the typeID of an RankedTensor type.
    pub fn getRankedTypeID() TypeID {
        return TypeID.fromRaw(c.mlirRankedTensorTypeGetTypeID());
    }

    /// Returns the typeID of an UnrankedTensor type.
    pub fn unrankedTensorTypeGetTypeID() TypeID {
        return TypeID.fromRaw(c.mlirUnrankedTensorTypeGetTypeID());
    }
};

//===----------------------------------------------------------------------===//
// Ranked / Unranked MemRef type.
//===----------------------------------------------------------------------===//
const MemRefType = struct {
    type: Type,

    /// Creates a MemRef type with the given rank and shape, a potentially empty
    /// list of affine layout maps, the given memory space and element type, in the
    /// same context as element type. The type is owned by the context.
    pub fn init(element_type: Type, shape: []const isize, layout: Attribute, memory_space: Attribute) MemRefType {
        return .{ .type = c.mlirMemRefTypeGet(element_type.getRaw(), shape.len, shape.ptr, layout.getRaw(), memory_space.getRaw()) };
    }

    /// Same as "mlirMemRefTypeGet" but returns a nullptr-wrapping MlirType o
    /// illegal arguments, emitting appropriate diagnostics.
    pub fn initChecked(loc: Location, element_type: Type, shape: []const isize, layout: Attribute, memory_space: Attribute) MemRefType {
        return .{ .type = c.mlirMemRefTypeGetChecked(loc.getRaw(), element_type.getRaw(), shape.len, shape.ptr, layout.getRaw(), memory_space.getRaw()) };
    }

    /// Creates a MemRef type with the given rank, shape, memory space and element
    /// type in the same context as the element type. The type has no affine maps,
    /// i.e. represents a default row-major contiguous memref. The type is owned by
    /// the context.
    pub fn contiguousGet(element_type: Type, shape: []const isize, memory_space: Attribute) MemRefType {
        return .{ .type = c.mlirMemRefTypeContiguousGet(element_type.getRaw(), shape.len, shape.ptr, memory_space.getRaw()) };
    }

    /// Same as "mlirMemRefTypeContiguousGet" but returns a nullptr wrapping
    /// MlirType on illegal arguments, emitting appropriate diagnostics.
    pub fn contiguousGetChecked(loc: Location, element_type: Type, shape: []const isize, layout: Attribute, memory_space: Attribute) MemRefType {
        return .{ .type = c.mlirMemRefTypeContiguousGetChecked(loc.getRaw(), element_type.getRaw(), shape.len, shape.ptr, layout.getRaw(), memory_space.getRaw()) };
    }

    /// Creates an Unranked MemRef type with the given element type and in the given
    /// memory space. The type is owned by the context of element type.
    pub fn initUnranked(element_type: Type, memory_space: Attribute) MemRefType {
        return .{ .type = c.mlirUnrankedMemRefTypeGet(element_type.getRaw(), memory_space.getRaw()) };
    }

    /// Same as "mlirUnrankedMemRefTypeGet" but returns a nullptr wrapping
    /// MlirType on illegal arguments, emitting appropriate diagnostics.
    pub fn initCheckedUnranked(element_type: Type, memory_space: Attribute) MemRefType {
        return .{ .type = c.mlirUnrankedMemRefTypeGetChecked(element_type.getRaw(), memory_space.getRaw()) };
    }

    /// Returns the layout of the given MemRef type.
    pub fn getLayout(@"type": MemRefType) Attribute {
        return Attribute.fromRaw(c.mlirMemRefTypeGetLayout(@"type".type.getRaw()));
    }

    /// Returns the affine map of the given MemRef type.
    // TODO: support affine map type
    pub fn getAffineMap(@"type": MemRefType) ?*const anyopaque {
        return c.mlirMemRefTypeGetAffineMap(@"type".type.getRaw()).ptr;
    }

    /// Returns the memory space of the given MemRef type.
    pub fn getMemorySpace(@"type": MemRefType) Attribute {
        return Attribute.fromRaw(c.mlirMemRefTypeGetMemorySpace(@"type".type.getRaw()));
    }

    /// Returns the memory spcae of the given Unranked MemRef type.
    pub fn getMemorySpaceUnranked(@"type": MemRefType) Attribute {
        return Attribute.fromRaw(c.mlirUnrankedMemrefGetMemorySpace(@"type".type.getRaw()));
    }

    /// Returns the typeID of an MemRef type.
    pub fn getTypeID() TypeID {
        return TypeID.fromRaw(c.mlirMemRefTypeGetTypeID());
    }

    /// Returns the typeID of an UnrankedMemRef type.
    pub fn getUnrankedTypeID() TypeID {
        return TypeID.fromRaw(c.mlirUnrankedMemRefTypeGetTypeID());
    }
};

//===----------------------------------------------------------------------===//
// Tuple type.
//===----------------------------------------------------------------------===//
const TupleType = struct {
    type: Type,

    /// Creates a tuple type that consists of the given list of elemental types. The
    /// type is owned by the context.
    pub fn init(ctx: Context, elements: []const Type) TupleType {
        return .{ .type = Type.fromRaw(c.mlirTupleTypeGet(ctx.getRaw(), elements.len, elements.ptr)) };
    }

    /// Returns the number of types contained in a tuple.
    pub fn getNumTypes(@"type": TupleType) isize {
        return c.mlirTupleTypeGetNumTypes(@"type".type.getRaw());
    }

    /// Returns the pos-th type in the tuple type.
    pub fn get(@"type": Type, pos: isize) Type {
        return Type.fromRaw(c.mlirTupleTypeGetType(@"type".getRaw(), pos));
    }

    /// Returns the typeID of an Tuple type.
    pub fn getTypeID() TypeID {
        return TypeID.fromRaw(c.mlirTupleTypeGetTypeID());
    }
};

//===----------------------------------------------------------------------===//
// Function type.
//===----------------------------------------------------------------------===//
const FuncType = struct {
    type: Type,

    /// Creates a function type, mapping a list of input types to result types.
    pub fn init(ctx: Context, inputs: []const Type, results: []const Type) FuncType {
        return .{ .type = c.mlirFunctionTypeGet(
            ctx.fromRaw(),
            inputs.len,
            inputs.ptr,
            results.len,
            results.ptr,
        ) };
    }

    /// Returns the number of input types.
    pub fn getNumInputs(@"type": FuncType) isize {
        return c.mlirFunctionTypeGetNumInputs(@"type".type.getRaw());
    }

    /// Returns the number of result types.
    pub fn getNumResults(@"type": FuncType) isize {
        return c.mlirFunctionTypeGetNumResults(@"type".type.getRaw());
    }

    /// Returns the pos-th input type.
    pub fn getInput(@"type": FuncType, pos: isize) Type {
        return Type.fromRaw(c.mlirFunctionTypeGetInput(@"type".type.getRaw(), pos));
    }

    /// Returns the pos-th result type.
    pub fn getResult(@"type": FuncType, pos: isize) Type {
        return Type.fromRaw(c.mlirFunctionTypeGetResult(@"type".type.getRaw(), pos));
    }

    /// Returns the typeID of an Function type.
    pub fn getTypeID() TypeID {
        return TypeID.fromRaw(c.mlirFunctionTypeGetTypeID());
    }
};

//===----------------------------------------------------------------------===//
// Opaque type.
//===----------------------------------------------------------------------===//
const OpaqueType = struct {
    type: Type,

    /// Creates an opaque type in the given context associated with the dialect
    /// identified by its namespace. The type contains opaque byte data of the
    /// specified length (data need not be null-terminated).
    pub fn init(ctx: Context, dialect_namespace: []const u8, type_data: []const u8) OpaqueType {
        return .{ .type = c.mlirOpaqueTypeGet(
            ctx.getRaw(),
            StringRef.init(dialect_namespace),
            StringRef.init(type_data),
        ) };
    }

    /// Returns the namespace of the dialect with which the given opaque type
    /// is associated. The namespace string is owned by the context.
    pub fn getDialectNamespace(@"type": OpaqueType) []const u8 {
        return StringRef.fromRaw(
            c.mlirOpaqueTypeGetDialectNamespace(@"type".type.getRaw()),
        ).slice();
    }

    /// Returns the raw data as a string reference. The data remains live as long as
    /// the context in which the type lives.
    pub fn getData(@"type": OpaqueType) []const u8 {
        return StringRef.fromRaw(c.mlirOpaqueTypeGetData(@"type".type.getRaw())).slice();
    }

    /// Returns the typeID of an Opaque type.
    pub fn getTypeID() TypeID {
        return TypeID.fromRaw(c.mlirOpaqueTypeGetTypeID());
    }
};

test {
    std.testing.refAllDeclsRecursive(@This());
}
