const std = @import("std");
const c = @import("c.zig");
const ir = @import("ir.zig");
const Attribute = ir.Attribute;
const Context = ir.Context;
const Location = ir.Location;
const NamedAttribute = ir.NamedAttribute;
const Type = ir.Type;
const support = @import("support.zig");
const StringRef = support.StringRef;
const TypeID = support.TypeID;

/// Returns an empty attribute.
pub const EmptyAttr = struct {
    attr: Attribute,

    pub fn init() EmptyAttr {
        return .{ .attr = Attribute.fromRaw(c.mlirAttributeGetNull()) };
    }
};

//===----------------------------------------------------------------------===//
// Location attribute.
//===----------------------------------------------------------------------===//
pub const LocationAttr = struct {
    attr: Attribute,
};

//===----------------------------------------------------------------------===//
// Affine map attribute.
//===----------------------------------------------------------------------===//
pub const AffineMapAttr = struct {
    attr: Attribute,

    /// Creates an affine map attribute wrapping the given map. The attribute
    /// belongs to the same context as the affine map.
    pub fn init(map: c.MlirAffineMap) AffineMapAttr {
        return .{ .attr = Attribute.fromRaw(c.mlirAffineMapAttrGet(map)) };
    }

    /// Returns the affine map wrapped in the given affine map attribute.
    pub fn value(attr: AffineMapAttr) c.MlirAffineMap {
        return c.mlirAffineMapAttrGetValue(attr.attr.getRaw());
    }

    /// Returns the typeID of an AffineMap attribute.
    pub fn typeID() TypeID {
        return TypeID.fromRaw(c.mlirAffineMapAttrGetTypeID());
    }
};

//===----------------------------------------------------------------------===//
// Array attribute.
//===----------------------------------------------------------------------===//
pub const ArrayAttr = struct {
    attr: Attribute,

    /// Creates an array element containing the given list of elements in the given
    /// context.
    pub fn init(ctx: Context, elements: []const Attribute) ArrayAttr {
        return .{ .attr = Attribute.fromRaw(c.mlirArrayAttrGet(
            ctx.getRaw(),
            @intCast(elements.len),
            @ptrCast(elements.ptr),
        )) };
    }

    /// Returns the number of elements stored in the given array attribute.
    pub fn getNumElements(attr: ArrayAttr) isize {
        return c.mlirArrayAttrGetNumElements(attr.attr.getRaw());
    }

    /// Returns pos-th element stored in the given array attribute.
    pub fn getElement(attr: ArrayAttr, pos: isize) Attribute {
        return Attribute.fromRaw(c.mlirArrayAttrGetElement(attr.attr.getRaw(), pos));
    }

    /// Returns the typeID of an Array attribute.
    pub fn getTypeID() TypeID {
        return TypeID.fromRaw(c.mlirArrayAttrGetTypeID());
    }
};

//===----------------------------------------------------------------------===//
// Dictionary attribute.
//===----------------------------------------------------------------------===//
pub const DictionaryAttr = struct {
    attr: Attribute,

    /// Creates a dictionary attribute containing the given list of elements in the
    /// provided context.
    pub fn init(ctx: Context, elements: []const Attribute) DictionaryAttr {
        return .{ .attr = Attribute.fromRaw(c.mlirDictionaryAttrGet(
            ctx.getRaw(),
            @intCast(elements.len),
            @ptrCast(elements.ptr),
        )) };
    }

    /// Returns the number of attributes contained in a dictionary attribute.
    pub fn getNumElements(attr: DictionaryAttr) isize {
        return c.mlirDictionaryAttrGetNumElements(attr.attr.getRaw());
    }

    /// Returns pos-th element of the given dictionary attribute.
    pub fn getElement(attr: DictionaryAttr, pos: isize) NamedAttribute {
        return NamedAttribute.fromRaw(c.mlirDictionaryAttrGetElement(attr.attr.getRaw(), pos));
    }

    /// Returns the dictionary attribute element with the given name or NULL if the
    /// given name does not exist in the dictionary.
    pub fn getElementByName(attr: DictionaryAttr, name: []const u8) Attribute {
        return Attribute.fromRaw(c.mlirDictionaryAttrGetElementByName(
            attr.attr.getRaw(),
            StringRef.init(name).getRaw(),
        ));
    }

    /// Returns the typeID of a Dictionary attribute.
    pub fn getTypeID() TypeID {
        return TypeID.fromRaw(c.mlirDictionaryAttrGetTypeID());
    }
};

//===----------------------------------------------------------------------===//
// Floating point attribute.
//===----------------------------------------------------------------------===//
pub const FloatingPointAttr = struct {
    attr: Attribute,

    // TODO: add support for APFloat and APInt to LLVM IR C API, then expose the
    // relevant functions here.

    /// Creates a floating point attribute in the given context with the given
    /// double value and double-precision FP semantics.
    pub fn get(ctx: Context, @"type": Type, val: f64) FloatingPointAttr {
        return .{ .attr = Attribute.fromRaw(c.mlirFloatAttrDoubleGet(ctx.getRaw(), @"type".getRaw(), val)) };
    }

    /// Same as "mlirFloatAttrDoubleGet", but if the type is not valid for a
    /// construction of a FloatAttr, returns a null MlirAttribute.
    pub fn getChecked(loc: Location, @"type": Type, val: f64) FloatingPointAttr {
        return .{ .attr = Attribute.fromRaw(c.mlirFloatAttrDoubleGetChecked(loc.getRaw(), @"type".getRaw(), val)) };
    }

    /// Returns the value stored in the given floating point attribute, interpreting
    /// the value as double.
    pub fn value(attr: FloatingPointAttr) f64 {
        return c.mlirFloatAttrGetValueDouble(attr.attr.getRaw());
    }

    /// Returns the typeID of a Float attribute.
    pub fn getTypeID() TypeID {
        return TypeID.fromRaw(c.mlirFloatAttrGetTypeID());
    }
};

//===----------------------------------------------------------------------===//
// Integer attribute.
//===----------------------------------------------------------------------===//
pub const IntegerAttr = struct {
    attr: Attribute,

    // TODO: add support for APFloat and APInt to LLVM IR C API, then expose the
    // relevant functions here.

    /// Creates an integer attribute of the given type with the given integer
    /// value.
    pub fn mlirIntegerAttrGet(@"type": Type, value: i64) IntegerAttr {
        return .{ .attr = Attribute.fromRaw(c.mlirIntegerAttrGet(@"type".getRaw(), value)) };
    }

    /// Returns the value stored in the given integer attribute, assuming the value
    /// is of signless type and fits into a signed 64-bit integer.
    pub fn valueSignless(attr: IntegerAttr) i64 {
        return c.mlirIntegerAttrGetValueInt(attr.attr.getRaw());
    }

    /// Returns the value stored in the given integer attribute, assuming the value
    /// is of signed type and fits into a signed 64-bit integer.
    pub fn valueSigned(attr: IntegerAttr) i64 {
        return c.mlirIntegerAttrGetValueSInt(attr.attr.getRaw());
    }

    /// Returns the value stored in the given integer attribute, assuming the value
    /// is of unsigned type and fits into an unsigned 64-bit integer.
    pub fn valueUnsigned(attr: IntegerAttr) u64 {
        return c.mlirIntegerAttrGetValueUInt(attr.attr.getRaw());
    }

    /// Returns the typeID of an Integer attribute.
    pub fn getTypeID() TypeID {
        return TypeID.fromRaw(c.mlirIntegerAttrGetTypeID());
    }
};

//===----------------------------------------------------------------------===//
// Bool attribute.
//===----------------------------------------------------------------------===//
pub const BoolAttr = struct {
    attr: Attribute,

    /// Creates a bool attribute in the given context with the given value.
    pub fn init(ctx: Context, val: bool) BoolAttr {
        return .{ .attr = Attribute.fromRaw(c.mlirBoolAttrGet(ctx.getRaw(), @intFromBool(val))) };
    }

    /// Returns the value stored in the given bool attribute.
    pub fn value(attr: BoolAttr) bool {
        return c.mlirBoolAttrGetValue(attr.attr.getRaw());
    }
};

//===----------------------------------------------------------------------===//
// Integer set attribute.
//===----------------------------------------------------------------------===//
pub const IntegerSetAttr = struct {
    attr: Attribute,

    /// Returns the typeID of an IntegerSet attribute.
    pub fn getTypeID() TypeID {
        return TypeID.fromRaw(c.mlirIntegerSetAttrGetTypeID());
    }
};

//===----------------------------------------------------------------------===//
// Opaque attribute.
//===----------------------------------------------------------------------===//
pub const OpaqueAttr = struct {
    attr: Attribute,

    /// Creates an opaque attribute in the given context associated with the dialect
    /// identified by its namespace. The attribute contains opaque byte data of the
    /// specified length (data need not be null-terminated).
    pub fn init(ctx: Context, dialect_namespace: []const u8, data: []const u8, @"type": Type) OpaqueAttr {
        return .{ .attr = Attribute.fromRaw(c.mlirOpaqueAttrGet(
            ctx.getRaw(),
            StringRef.init(dialect_namespace).getRaw(),
            @intCast(data.len),
            @ptrCast(data),
            @"type".getRaw(),
        )) };
    }

    /// Returns the namespace of the dialect with which the given opaque attribute
    /// is associated. The namespace string is owned by the context.
    pub fn getDialectNamespace(attr: OpaqueAttr) []const u8 {
        return StringRef.fromRaw(
            c.mlirOpaqueAttrGetDialectNamespace(attr.attr.getRaw()),
        ).slice();
    }

    /// Returns the raw data as a string reference. The data remains live as long as
    /// the context in which the attribute lives.
    pub fn getData(attr: OpaqueAttr) []const u8 {
        return StringRef.fromRaw(
            c.mlirOpaqueAttrGetData(attr.attr.getRaw()),
        ).slice();
    }

    /// Returns the typeID of an Opaque attribute.
    pub fn getTypeID() TypeID {
        return TypeID.fromRaw(c.mlirOpaqueAttrGetTypeID());
    }
};

//===----------------------------------------------------------------------===//
// String attribute.
//===----------------------------------------------------------------------===//
pub const StringAttr = struct {
    attr: Attribute,

    /// Creates a string attribute in the given context containing the given string.
    pub fn init(ctx: Context, str: []const u8) StringAttr {
        return .{ .attr = Attribute.fromRaw(c.mlirStringAttrGet(ctx.getRaw(), StringRef.init(str).getRaw())) };
    }

    /// Creates a string attribute in the given context containing the given string.
    /// Additionally, the attribute has the given type.
    pub fn initTyped(@"type": Type, str: []const u8) StringAttr {
        return .{ .attr = Attribute.fromRaw(c.mlirStringAttrTypedGet(@"type".getRaw(), StringRef.init(str).getRaw())) };
    }

    /// Returns the attribute values as a string reference. The data remains live as
    /// long as the context in which the attribute lives.
    pub fn value(attr: StringAttr) []const u8 {
        return StringRef.fromRaw(c.mlirStringAttrGetValue(attr.attr.getRaw())).slice();
    }

    /// Returns the typeID of a String attribute.
    pub fn mlirStringAttrGetTypeID() TypeID {
        return TypeID.fromRaw(c.mlirStringAttrGetTypeID());
    }
};

//===----------------------------------------------------------------------===//
// SymbolRef attribute.
//===----------------------------------------------------------------------===//
pub const SymbolRefAttr = struct {
    attr: Attribute,

    /// Creates a symbol reference attribute in the given context referencing a
    /// symbol identified by the given string inside a list of nested references.
    /// Each of the references in the list must not be nested.
    pub fn init(ctx: Context, symbol: []const u8, references: []const Attribute) SymbolRefAttr {
        return .{ .attr = Attribute.fromRaw(c.mlirSymbolRefAttrGet(
            ctx.getRaw(),
            StringRef.init(symbol).getRaw(),
            @intCast(references.len),
            @ptrCast(references),
        )) };
    }

    /// Returns the string reference to the root referenced symbol. The data remains
    /// live as long as the context in which the attribute lives.
    pub fn getRootReference(attr: SymbolRefAttr) []const u8 {
        return StringRef.fromRaw(c.mlirSymbolRefAttrGetRootReference(attr.attr.getRaw())).slice();
    }

    /// Returns the string reference to the leaf referenced symbol. The data remains
    /// live as long as the context in which the attribute lives.
    pub fn getLeafReference(attr: SymbolRefAttr) []const u8 {
        return StringRef.fromRaw(c.mlirSymbolRefAttrGetLeafReference(attr.attr.getRaw())).slice();
    }

    /// Returns the number of references nested in the given symbol reference
    /// attribute.
    pub fn getNumNestedReferences(attr: SymbolRefAttr) isize {
        return c.mlirSymbolRefAttrGetNumNestedReferences(attr.attr.getRaw());
    }

    /// Returns pos-th reference nested in the given symbol reference attribute.
    pub fn getNestedReference(attr: SymbolRefAttr, pos: isize) Attribute {
        return Attribute.fromRaw(c.mlirSymbolRefAttrGetNestedReference(attr.attr.getRaw(), pos));
    }

    /// Returns the typeID of an SymbolRef attribute.
    pub fn getTypeID() TypeID {
        return TypeID.fromRaw(c.mlirSymbolRefAttrGetTypeID());
    }
};

//===----------------------------------------------------------------------===//
// Flat SymbolRef attribute.
//===----------------------------------------------------------------------===//
pub const FlatSymbolRefAttr = struct {
    attr: Attribute,

    /// Creates a flat symbol reference attribute in the given context referencing a
    /// symbol identified by the given string.
    pub fn init(ctx: Context, symbol: []const u8) FlatSymbolRefAttr {
        return .{ .attr = Attribute.fromRaw(c.mlirFlatSymbolRefAttrGet(
            ctx.getRaw(),
            StringRef.init(symbol).getRaw(),
        )) };
    }

    /// Returns the referenced symbol as a string reference. The data remains live
    /// as long as the context in which the attribute lives.
    pub fn value(attr: FlatSymbolRefAttr) []const u8 {
        return StringRef.fromRaw(c.mlirFlatSymbolRefAttrGetValue(attr.attr.getRaw())).slice();
    }
};

//===----------------------------------------------------------------------===//
// Type attribute.
//===----------------------------------------------------------------------===//
pub const TypeAttr = struct {
    attr: Attribute,

    /// Creates a type attribute wrapping the given type in the same context as the
    /// type.
    pub fn init(@"type": Type) TypeAttr {
        return .{ .attr = Attribute.fromRaw(c.mlirTypeAttrGet(@"type".getRaw())) };
    }

    /// Returns the type stored in the given type attribute.
    pub fn value(attr: TypeAttr) Type {
        return Type.fromRaw(c.mlirTypeAttrGetValue(attr.attr.getRaw()));
    }

    /// Returns the typeID of a Type attribute.
    pub fn getTypeID() TypeID {
        return TypeID.fromRaw(c.mlirTypeAttrGetTypeID());
    }
};

//===----------------------------------------------------------------------===//
// Unit attribute.
//===----------------------------------------------------------------------===//
pub const UnitAttr = struct {
    attr: Attribute,

    /// Creates a unit attribute in the given context.
    pub fn init(ctx: Context) UnitAttr {
        return .{ .attr = Attribute.fromRaw(c.mlirUnitAttrGet(ctx.getRaw())) };
    }

    /// Returns the typeID of a Unit attribute.
    pub fn getTypeID() TypeID {
        return TypeID.fromRaw(c.mlirUnitAttrGetTypeID());
    }
};

//===----------------------------------------------------------------------===//
// Elements attributes.
//===----------------------------------------------------------------------===//
pub const ElementsAttr = struct {
    attr: Attribute,

    /// Returns the element at the given rank-dimensional index.
    pub fn value(attr: ElementsAttr, rank: isize, idxs: [*]u64) Attribute {
        return Attribute.fromRaw(c.mlirElementsAttrGetValue(attr.attr.getRaw(), rank, idxs));
    }

    /// Checks whether the given rank-dimensional index is valid in the given
    /// elements attribute.
    pub fn isValidIndex(attr: ElementsAttr, rank: isize, idxs: [*]u64) bool {
        return c.mlirElementsAttrIsValidIndex(attr.attr.getRaw(), rank, idxs);
    }

    /// Gets the total number of elements in the given elements attribute. In order
    /// to iterate over the attribute, obtain its type, which must be a statically
    /// shaped type and use its sizes to build a multi-dimensional index.
    pub fn getNumElements(attr: ElementsAttr) i64 {
        return c.mlirElementsAttrGetNumElements(attr.attr.getRaw());
    }
};

//===----------------------------------------------------------------------===//
// Dense array attribute.
//===----------------------------------------------------------------------===//
pub const DenseArrayAttr = struct {
    fn createDenseArrayAttr(comptime @"type": type) type {
        return struct {
            const TypeName = @typeName(@"type");
            const FixedTypeName = &[_]u8{std.ascii.toUpper(TypeName[0])} ++ TypeName[1..];
            const InitFn = @field(c, "mlirDense" ++ FixedTypeName ++ "ArrayGet");
            const GetFn = @field(c, "mlirDense" ++ FixedTypeName ++ "ArrayGetElement");

            attr: Attribute,

            /// Create a dense array attribute with the given elements.
            pub fn init(ctx: Context, values: []const @"type") @This() {
                const raw_attr = InitFn(ctx.getRaw(), @intCast(values.len), values.ptr);
                return .{ .attr = Attribute.fromRaw(raw_attr) };
            }

            /// Get the size of a dense array.
            pub fn getNumElements(attr: @This()) isize {
                return c.mlirDenseArrayGetNumElements(attr.attr.getRaw());
            }

            /// Get an element of a dense array.
            pub fn get(attr: @This(), pos: isize) @"type" {
                return GetFn(attr.attr.getRaw(), pos);
            }
        };
    }

    // TODO: possible ABI issues between use of bool slice which expects int slice
    // pub const DenseBoolArrayAttr = createDenseArrayAttr(bool);
    pub const DenseI8ArrayAttr = createDenseArrayAttr(i8);
    pub const DenseI16ArrayAttr = createDenseArrayAttr(i16);
    pub const DenseI32ArrayAttr = createDenseArrayAttr(i32);
    pub const DenseI64ArrayAttr = createDenseArrayAttr(i64);
    pub const DenseF32ArrayAttr = createDenseArrayAttr(f32);
    pub const DenseF64ArrayAttr = createDenseArrayAttr(f64);

    pub fn getTypeID() TypeID {
        return TypeID.fromRaw(c.mlirDenseArrayAttrGetTypeID());
    }
};

//===----------------------------------------------------------------------===//
// Dense elements attribute.
//===----------------------------------------------------------------------===//
pub const DenseElementsAttr = struct {
    attr: Attribute,

    // TODO: decide on the interface and add support for complex elements.
    // TODO: add support for APFloat and APInt to LLVM IR C API, then expose the
    // relevant functions here.

    /// Creates a dense elements attribute with the given Shaped type and elements
    /// in the same context as the type.
    pub fn init(shaped_type: Type, elements: []const Attribute) DenseElementsAttr {
        return .{ .attr = Attribute.fromRaw(c.mlirDenseElementsAttrGet(
            shaped_type.getRaw(),
            @intCast(elements.len),
            @ptrCast(elements.ptr),
        )) };
    }

    /// Creates a dense elements attribute with the given Shaped type and elements
    /// populated from a packed, row-major opaque buffer of contents.
    ///
    /// The format of the raw buffer is a densely packed array of values that
    /// can be bitcast to the storage format of the element type specified.
    /// Types that are not byte aligned will be:
    ///   - For bitwidth > 1: Rounded up to the next byte.
    ///   - For bitwidth = 1: Packed into 8bit bytes with bits corresponding to
    ///     the linear order of the shape type from MSB to LSB, padded to on the
    ///     right.
    ///
    /// A raw buffer of a single element (or for 1-bit, a byte of value 0 or 255)
    /// will be interpreted as a splat. User code should be prepared for additional,
    /// conformant patterns to be identified as splats in the future.
    // pub fn mlirDenseElementsAttrRawBufferGet(MlirType shapedType, size_t rawBufferSize, const void *rawBuffer) MlirAttribute;

    fn CreateDenseElements(comptime @"type": type, comptime name: []const u8) type {
        return struct {
            const InitFn = @field(c, "mlirDenseElementsAttr" ++ name ++ "Get");
            const InitSpatFn = @field(c, "mlirDenseElementsAttr" ++ name ++ "SplatGet");
            const GetFn = @field(c, "mlirDenseElementsAttrGet" ++ name ++ "Value");
            const GetSplatFn = @field(c, "mlirDenseElementsAttrGet" ++ name ++ "SplatValue");

            attr: Attribute,

            /// Creates a dense elements attribute with the given shaped type from elements
            /// of a specific type. Expects the element type of the shaped type to match the
            /// data element type.
            pub fn init(shaped_type: Type, elements: []const @"type") @This() {
                return .{ .attr = Attribute.fromRaw(InitFn(shaped_type.getRaw(), @intCast(elements.len), @ptrCast(elements.ptr))) };
            }

            /// Creates a dense elements attribute with the given Shaped type containing a
            /// single replicated element (splat).
            pub fn initSplat(shaped_type: Type, element: @"type") @This() {
                return .{ .attr = Attribute.fromRaw(InitSpatFn(shaped_type.getRaw(), element)) };
            }

            pub fn toDenseElementsAttr(attr: @This()) DenseElementsAttr {
                return .{ .attr = attr.attr };
            }

            /// Returns the raw data of the given dense elements attribute.
            pub fn getRaw(attr: @This()) [*]const @"type" {
                return @ptrCast(@alignCast(c.mlirDenseElementsAttrGetRawData(attr.attr.getRaw())));
            }

            /// Returns the single replicated value (splat) of a specific type contained by
            /// the given dense elements attribute.
            pub fn getSplatValue(attr: @This()) @"type" {
                return GetSplatFn(attr.attr.getRaw());
            }

            /// Returns the pos-th value (flat contiguous indexing) of a specific type
            /// contained by the given dense elements attribute.
            pub fn get(attr: @This(), pos: isize) @"type" {
                return GetFn(attr.attr.getRaw(), pos);
            }
        };
    }

    // TODO: possible ABI issues between use of bool slice which expects int slice
    // pub const DenseBoolElements = CreateDenseElements(bool, "Bool");
    pub const DenseU8Elements = CreateDenseElements(u8, "UInt8");
    pub const DenseI8Elements = CreateDenseElements(i8, "Int8");
    pub const DenseU32Elements = CreateDenseElements(u32, "UInt32");
    pub const DenseI32Elements = CreateDenseElements(i32, "Int32");
    pub const DenseU64Elements = CreateDenseElements(u64, "UInt64");
    pub const DenseI64Elements = CreateDenseElements(i64, "Int64");
    pub const DenseF32Elements = CreateDenseElements(f32, "Float");
    pub const DenseF64Elements = CreateDenseElements(f64, "Double");
    // TODO: support bflat16 and float16 DenseElementsAttr

    /// Creates a dense elements attribute with the given shaped type from string
    /// elements.
    pub fn initStrs(shaped_type: Type, strs: []StringRef) DenseElementsAttr {
        return .{ .attr = Attribute.fromRaw(c.mlirDenseElementsAttrStringGet(
            shaped_type.getRaw(),
            @intCast(strs.len),
            @ptrCast(strs.ptr),
        )) };
    }

    /// Creates a dense elements attribute that has the same data as the given dense
    /// elements attribute and a different shaped type. The new type must have the
    /// same total number of elements.
    pub fn initReshaped(attr: DenseElementsAttr, shaped_type: Type) DenseElementsAttr {
        return .{ .attr = Attribute.fromRaw(c.mlirDenseElementsAttrReshapeGet(
            attr.attr.getRaw(),
            shaped_type.getRaw(),
        )) };
    }

    /// Checks whether the given dense elements attribute contains a single
    /// replicated value (splat).
    pub fn isSplat(attr: DenseElementsAttr) bool {
        return c.mlirDenseElementsAttrIsSplat(attr.attr.getRaw());
    }

    /// Returns the single replicated value (splat) of a specific type contained by
    /// the given dense elements attribute.
    pub fn getStringSplatValue(attr: DenseElementsAttr) []const u8 {
        return StringRef.fromRaw(c.mlirDenseElementsAttrGetStringSplatValue(attr.attr.getRaw())).slice();
    }

    /// Returns the pos-th value (flat contiguous indexing) of a specific type
    /// contained by the given dense elements attribute.
    pub fn getStringValue(attr: DenseElementsAttr, pos: isize) []const u8 {
        return StringRef.fromRaw(c.mlirDenseElementsAttrGetStringValue(attr.attr.getRaw(), pos)).slice();
    }

    /// Returns the typeID of an DenseIntOrFPElements attribute.
    pub fn getTypeID() TypeID {
        return TypeID.fromRaw(c.mlirDenseIntOrFPElementsAttrGetTypeID());
    }
};

//===----------------------------------------------------------------------===//
// Resource blob attributes.
//===----------------------------------------------------------------------===//
pub const ResourceBlobAttr = struct {
    attr: Attribute,

    /// Unlike the typed accessors below, constructs the attribute with a raw
    /// data buffer and no type/alignment checking. Use a more strongly typed
    /// accessor if possible. If dataIsMutable is false, then an immutable
    /// AsmResourceBlob will be created and that passed data contents will be
    /// treated as const.
    /// If the deleter is non NULL, then it will be called when the data buffer
    /// can no longer be accessed (passing userData to it).
    pub fn init(shaped_type: Type, name: []const u8, data: anytype, data_is_mutable: bool, deleter: DeleterCallback, user_data: *anyopaque) ResourceBlobAttr {
        const DataType = @TypeOf(data);
        if (DataType != .Pointer or DataType.Pointer.size != .Slice) {
            @compileError("Expected slice of values, found: " ++ @typeName(DataType));
        }
        return .{ .attr = Attribute.fromRaw(c.mlirUnmanagedDenseResourceElementsAttrGet(
            shaped_type.getRaw(),
            StringRef.init(name).getRaw(),
            @ptrCast(data.ptr),
            @intCast(data.len),
            @alignOf(DataType.Pointer.child),
            data_is_mutable,
            deleter,
            user_data,
        )) };
    }

    fn CreateResourceBlobAttr(comptime @"type": type, comptime ty_name: []const u8) type {
        return struct {
            const InitFn = @field(c, "mlirUnmanagedDense" ++ ty_name ++ "ResourceElementsAttrGet");
            const GetFn = @field(c, "mlirDense" ++ ty_name ++ "ResourceElementsAttrGetValue");

            attr: Attribute,

            pub fn init(shaped_type: Type, name: []const u8, elements: []const @"type") @This() {
                return .{ .attr = Attribute.fromRaw(InitFn(
                    shaped_type.getRaw(),
                    StringRef.init(name).getRaw(),
                    @intCast(elements.len),
                    @ptrCast(elements.ptr),
                )) };
            }

            /// Returns the pos-th value (flat contiguous indexing) of a specific type
            /// contained by the given dense resource elements attribute.
            pub fn get(attr: @This(), pos: isize) @"type" {
                return GetFn(attr.attr.getRaw(), pos);
            }
        };
    }

    // TODO: possible ABI issues between use of bool slice which expects int slice
    // pub const BoolResourceBlobAttr = CreateResourceBlobAttr(bool, "Bool");
    pub const U8ResourceBlobAttr = CreateResourceBlobAttr(u8, "UInt8");
    pub const I8ResourceBlobAttr = CreateResourceBlobAttr(i8, "Int8");
    pub const U16ResourceBlobAttr = CreateResourceBlobAttr(u16, "UInt16");
    pub const I16ResourceBlobAttr = CreateResourceBlobAttr(i16, "Int16");
    pub const U32ResourceBlobAttr = CreateResourceBlobAttr(u32, "UInt32");
    pub const I32ResourceBlobAttr = CreateResourceBlobAttr(i32, "Int32");
    pub const U64ResourceBlobAttr = CreateResourceBlobAttr(u64, "UInt64");
    pub const I64ResourceBlobAttr = CreateResourceBlobAttr(i64, "Int64");
    pub const F32ResourceBlobAttr = CreateResourceBlobAttr(f32, "Float");
    pub const F64ResourceBlobAttr = CreateResourceBlobAttr(f64, "Double");

    pub const DeleterCallback = *const fn (user_data: *anyopaque, data: *const anyopaque, size: usize, @"align": usize) void;
};

//===----------------------------------------------------------------------===//
// Sparse elements attribute.
//===----------------------------------------------------------------------===//
pub const SparseElementsAttr = struct {
    attr: Attribute,

    /// Creates a sparse elements attribute of the given shape from a list of
    /// indices and a list of associated values. Both lists are expected to be dense
    /// elements attributes with the same number of elements. The list of indices is
    /// expected to contain 64-bit integers. The attribute is created in the same
    /// context as the type.
    pub fn init(shaped_type: Type, dense_indices: Attribute, dense_values: Attribute) SparseElementsAttr {
        return .{ .attr = Attribute.fromRaw(c.mlirSparseElementsAttribute(
            shaped_type.getRaw(),
            dense_indices.getRaw(),
            dense_values.getRaw(),
        )) };
    }

    /// Returns the dense elements attribute containing 64-bit integer indices of
    /// non-null elements in the given sparse elements attribute.
    pub fn getIndices(attr: SparseElementsAttr) Attribute {
        return Attribute.fromRaw(c.mlirSparseElementsAttrGetIndices(attr.attr.getRaw()));
    }

    /// Returns the dense elements attribute containing the non-null elements in the
    /// given sparse elements attribute.
    pub fn getValues(attr: SparseElementsAttr) Attribute {
        return Attribute.fromRaw(c.mlirSparseElementsAttrGetValues(attr.attr.getRaw()));
    }

    /// Returns the typeID of a SparseElements attribute.
    pub fn getTypeID() TypeID {
        return TypeID.fromRaw(c.mlirSparseElementsAttrGetTypeID());
    }
};

//===----------------------------------------------------------------------===//
// Strided layout attribute.
//===----------------------------------------------------------------------===//
pub const StridedLayoutAttr = struct {
    attr: Attribute,

    // Creates a strided layout attribute from given strides and offset.
    pub fn init(ctx: Context, offset: i64, strides: []const i64) StridedLayoutAttr {
        return .{ .attr = Attribute.fromRaw(c.mlirStridedLayoutAttrGet(ctx.getRaw(), offset, @intCast(strides.len), strides.ptr)) };
    }

    // Returns the offset in the given strided layout layout attribute.
    pub fn getOffset(attr: StridedLayoutAttr) i64 {
        return c.mlirStridedLayoutAttrGetOffset(attr.attr.getRaw());
    }

    // Returns the number of strides in the given strided layout attribute.
    pub fn getNumStrides(attr: StridedLayoutAttr) isize {
        return c.mlirStridedLayoutAttrGetNumStrides(attr.attr.getRaw());
    }

    // Returns the pos-th stride stored in the given strided layout attribute.
    pub fn getStride(attr: StridedLayoutAttr, pos: isize) i64 {
        return c.mlirStridedLayoutAttrGetStride(attr.attr.getRaw(), pos);
    }

    /// Returns the typeID of a StridedLayout attribute.
    pub fn getTypeID() TypeID {
        return TypeID.fromRaw(c.mlirStridedLayoutAttrGetTypeID());
    }
};

test {
    std.testing.refAllDeclsRecursive(@This());
}
