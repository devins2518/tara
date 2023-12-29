const std = @import("std");
const assert = std.debug.assert;
const c = @import("c.zig");
const support = @import("support.zig");
const StringRef = support.StringRef;
const StringCallback = StringRef.StringCallback;
const LlvmThreadPool = support.LlvmThreadPool;
const TypeID = support.TypeID;
const LogicalResult = support.LogicalResult;

//===----------------------------------------------------------------------===//
/// Opaque type declarations.
///
/// Types are exposed to C bindings as structs containing opaque pointers. They
/// are not supposed to be inspected from C. This allows the underlying
/// representation to change without affecting the API users. The use of structs
/// instead of typedefs enables some type safety as structs are not implicitly
/// convertible to each other.
///
/// Instances of these types may or may not own the underlying object (most
/// often only point to an IR fragment without owning it). The ownership
/// semantics is defined by how an instance of the type was obtained.
//===----------------------------------------------------------------------===//

/// Named MLIR attribute.
///
/// A named attribute is essentially a (name, attribute) pair where the name is
/// a string.
pub const NamedAttribute = struct {
    _: c.MlirNamedAttribute,

    pub fn fromRaw(raw: c.MlirNamedAttribute) NamedAttribute {
        return .{ ._ = raw };
    }

    pub fn getRaw(named_attr: NamedAttribute) c.MlirNamedAttribute {
        return named_attr._;
    }

    pub fn name(named_attr: NamedAttribute) Identifier {
        return Identifier.fromRaw(named_attr.getRaw().name);
    }

    pub fn attribute(named_attr: NamedAttribute) Attribute {
        return Attribute.fromRaw(named_attr.getRaw().attribute);
    }

    /// Associates an attribute with the name. Takes ownership of neither.
    pub fn initFromNameAndAttr(ident: Identifier, attr: Attribute) NamedAttribute {
        return NamedAttribute.fromRaw(c.mlirNamedAttributeGet(ident.getRaw(), attr.getRaw()));
    }
};

//===----------------------------------------------------------------------===//
// Context API.
//===----------------------------------------------------------------------===//
pub const Context = struct {
    _: c.MlirContext,

    pub fn fromRaw(raw: c.MlirContext) Context {
        return .{ ._ = raw };
    }

    pub fn getRaw(context: Context) c.MlirContext {
        return context._;
    }

    /// Creates an MLIR context and transfers its ownership to the caller.
    /// This sets the default multithreading option (enabled).
    pub fn init() Context {
        return Context.fromRaw(c.mlirContextCreate());
    }

    /// Creates an MLIR context with an explicit setting of the multithreading
    /// setting and transfers its ownership to the caller.
    pub fn initWithThreading(threading_enabled: bool) Context {
        return Context.fromRaw(c.mlirContextCreateWithThreading(threading_enabled));
    }

    /// Creates an MLIR context, setting the multithreading setting explicitly and
    /// pre-loading the dialects from the provided DialectRegistry.
    pub fn initWithRegistry(register: DialectRegistry, threading_enabled: bool) Context {
        return Context.fromRaw(c.mlirContextCreateWithRegistry(register.getRaw(), threading_enabled));
    }

    /// Checks if two contexts are equal.
    pub fn eql(ctx1: Context, ctx2: Context) bool {
        return c.mlirContextEqual(ctx1.getRaw(), ctx2.getRaw());
    }

    /// Checks whether a context is null.
    pub fn contextIsNull(context: Context) bool {
        return context.getRaw().ptr == null;
    }

    /// Takes an MLIR context owned by the caller and destroys it.
    pub fn deinit(context: Context) void {
        c.mlirContextDestroy(context.getRaw());
    }

    /// Sets whether unregistered dialects are allowed in this context.
    pub fn setAllowUnregisteredDialects(context: Context, allow: bool) void {
        c.mlirContextSetAllowUnregisteredDialects(context.getRaw(), allow);
    }

    /// Returns whether the context allows unregistered dialects.
    pub fn getAllowUnregisteredDialects(context: Context) bool {
        return c.mlirContextGetAllowUnregisteredDialects(context.getRaw());
    }

    /// Returns the number of dialects registered with the given context. A
    /// registered dialect will be loaded if needed by the parser.
    pub fn getNumRegisteredDialects(context: Context) isize {
        return c.mlirContextGetNumRegisteredDialects(context.getRaw());
    }

    /// Append the contents of the given dialect registry to the registry associated
    /// with the context.
    pub fn appendDialectRegistry(ctx: Context, registry: DialectRegistry) void {
        c.mlirContextAppendDialectRegistry(ctx.getRaw(), registry.getRaw());
    }

    /// Returns the number of dialects loaded by the context.
    pub fn getNumLoadedDialects(context: Context) isize {
        return c.mlirContextGetNumLoadedDialects(context.getRaw());
    }

    /// Gets the dialect instance owned by the given context using the dialect
    /// namespace to identify it, loads (i.e., constructs the instance of) the
    /// dialect if necessary. If the dialect is not registered with the context,
    /// returns null. Use mlirContextLoad<Name>Dialect to load an unregistered
    /// dialect.
    pub fn getOrLoadDialect(context: Context, name: StringRef) Dialect {
        return Dialect.fromRaw(c.mlirContextGetOrLoadDialect(context.getRaw(), name.getRaw()));
    }

    /// Set threading mode (must be set to false to mlir-print-ir-after-all).
    pub fn enableMultithreading(context: Context, enable: bool) void {
        return c.mlirContextEnableMultithreading(context.getRaw(), enable);
    }

    /// Eagerly loads all available dialects registered with a context, making
    /// them available for use for IR construction.
    pub fn loadAllAvailableDialects(context: Context) void {
        c.mlirContextLoadAllAvailableDialects(context.getRaw());
    }

    /// Returns whether the given fully-qualified operation (i.e.
    /// 'dialect.operation') is registered with the context. This will return true
    /// if the dialect is loaded and the operation is registered within the
    /// dialect.
    pub fn isRegisteredOperation(context: Context, name: StringRef) bool {
        return c.mlirContextIsRegisteredOperation(context.getRaw(), name.getRaw());
    }

    /// Sets the thread pool of the context explicitly, enabling multithreading in
    /// the process. This API should be used to avoid re-creating thread pools in
    /// long-running applications that perform multiple compilations, see
    /// the C++ documentation for MLIRContext for details.
    pub fn setThreadPool(context: Context, thread_pool: LlvmThreadPool) void {
        c.mlirContextSetThreadPool(context.getRaw(), thread_pool.getRaw());
    }
};

//===----------------------------------------------------------------------===//
// Dialect API.
//===----------------------------------------------------------------------===//
pub const Dialect = struct {
    _: c.MlirDialect,

    pub fn fromRaw(raw: c.MlirDialect) Dialect {
        return .{ ._ = raw };
    }

    pub fn getRaw(dialect: Dialect) c.MlirDialect {
        return dialect._;
    }

    /// Returns the context that owns the dialect.
    pub fn getContext(dialect: Dialect) Context {
        return Context.fromRaw(c.mlirDialectGetContext(dialect.getRaw()));
    }

    /// Checks if the dialect is null.
    pub fn isNull(dialect: Dialect) bool {
        return c.mlirDialectIsNull(dialect.getRaw());
    }

    /// Checks if two dialects that belong to the same context are equal. Dialects
    /// from different contexts will not compare equal.
    pub fn eql(dialect: Dialect, other: Dialect) bool {
        return c.mlirDialectEqual(dialect.getRaw(), other.getRaw());
    }

    /// Returns the namespace of the given dialect.
    pub fn getNamespace(dialect: Dialect) StringRef {
        return StringRef.fromRaw(c.mlirDialectGetNamespace(dialect.getRaw()));
    }
};

//===----------------------------------------------------------------------===//
// DialectHandle API.
// Registration entry-points for each dialect are declared using the common
// MLIR_DECLARE_DIALECT_REGISTRATION_CAPI macro, which takes the dialect
// API name (i.e. "Func", "Tensor", "Linalg") and namespace (i.e. "func",
// "tensor", "linalg"). The following declarations are produced:
//
//   /// Gets the above hook methods in struct form for a dialect by namespace.
//   /// This is intended to facilitate dynamic lookup and registration of
//   /// dialects via a plugin facility based on shared library symbol lookup.
//   const MlirDialectHandle *mlirGetDialectHandle__{NAMESPACE}__();
//
// This is done via a common macro to facilitate future expansion to
// registration schemes.
//===----------------------------------------------------------------------===//
pub const DialectHandle = struct {
    _: c.MlirDialectHandle,

    pub fn fromRaw(raw: c.MlirDialectHandle) DialectHandle {
        return .{ ._ = raw };
    }

    pub fn getRaw(dialect_handle: DialectHandle) c.MlirDialectHandle {
        return dialect_handle._;
    }

    //            /// Returns the namespace associated with the provided dialect handle.
    pub fn getNamespace(handle: DialectHandle) StringRef {
        return StringRef.fromRaw(c.mlirDialectHandleGetNamespace(handle.getRaw()));
    }

    /// Inserts the dialect associated with the provided dialect handle into the
    /// provided dialect registry
    pub fn insertDialect(handle: DialectHandle, registry: DialectRegistry) void {
        c.mlirDialectHandleInsertDialect(handle.getRaw(), registry.getRaw());
    }

    /// Registers the dialect associated with the provided dialect handle.
    pub fn registerDialect(handle: DialectHandle, context: Context) void {
        return c.mlirDialectHandleRegisterDialect(handle.getRaw(), context.getRaw());
    }

    /// Loads the dialect associated with the provided dialect handle.
    pub fn loadDialect(handle: DialectHandle, context: Context) Dialect {
        return Dialect.fromRaw(c.mlirDialectHandleLoadDialect(handle.getRaw(), context.getRaw()));
    }
};

//===----------------------------------------------------------------------===//
// DialectRegistry API.
//===----------------------------------------------------------------------===//
pub const DialectRegistry = struct {
    _: c.MlirDialectRegistry,

    pub fn fromRaw(raw: c.MlirDialectRegistry) DialectRegistry {
        return .{ ._ = raw };
    }

    pub fn getRaw(dialect_registry: DialectRegistry) c.MlirDialectRegistry {
        return dialect_registry._;
    }

    /// Creates a dialect registry and transfers its ownership to the caller.
    pub fn init() DialectRegistry {
        return DialectRegistry.fromRaw(c.mlirDialectRegistryCreate());
    }

    /// Checks if the dialect registry is null.
    pub fn isNull(registry: DialectRegistry) bool {
        return c.mlirDialectRegistryIsNull(registry.getRaw());
    }

    /// Takes a dialect registry owned by the caller and destroys it.
    pub fn deinit(registry: DialectRegistry) void {
        c.mlirDialectRegistryDestroy(registry.getRaw());
    }
};

//===----------------------------------------------------------------------===//
// Location API.
//===----------------------------------------------------------------------===//
pub const Location = struct {
    _: c.MlirLocation,

    pub fn fromRaw(raw: c.MlirLocation) Location {
        return .{ ._ = raw };
    }

    pub fn getRaw(location: Location) c.MlirLocation {
        return location._;
    }

    /// Returns the underlying location attribute of this location.
    pub fn getAttribute(location: Location) Attribute {
        return Attribute.fromRaw(c.mlirLocationGetAttribute(location.getRaw()));
    }

    /// Creates a location from a location attribute.
    pub fn fromAttribute(attribute: Attribute) Location {
        return Location.fromRaw(c.mlirLocationFromAttribute(attribute.getRaw()));
    }

    /// Creates an File/Line/Column location owned by the given context.
    pub fn fileLineColGet(context: Context, filename: StringRef, line: u32, col: u32) Location {
        return Location.fromRaw(c.mlirLocationFileLineColGet(context.getRaw(), filename.getRaw(), line, col));
    }

    /// Creates a call site location with a callee and a caller.
    pub fn callSiteGet(callee: Location, caller: Location) Location {
        return Location.fromRaw(c.mlirLocationCallSiteGet(callee.getRaw(), caller.getRaw()));
    }

    /// Creates a fused location with an array of locations and metadata.
    pub fn fusedGet(ctx: Context, num_locations: isize, locations: []const Location, metadata: Attribute) Location {
        return Location.fromRaw(c.mlirLocationFusedGet(ctx.getRaw(), num_locations, @ptrCast(locations.ptr), metadata.getRaw()));
    }

    /// Creates a name location owned by the given context. Providing null location
    /// for childLoc is allowed and if childLoc is null location, then the behavior
    /// is the same as having unknown child location.
    pub fn nameGet(context: Context, name: StringRef, childLoc: Location) Location {
        return Location.fromRaw(c.mlirLocationNameGet(context.getRaw(), name.getRaw(), childLoc.getRaw()));
    }

    /// Creates a location with unknown position owned by the given context.
    pub fn unknownGet(context: Context) Location {
        return Location.fromRaw(c.mlirLocationUnknownGet(context.getRaw()));
    }

    /// Gets the context that a location was created with.
    pub fn getContext(location: Location) Context {
        return Context.fromRaw(c.mlirLocationGetContext(location.getRaw()));
    }

    /// Checks if the location is null.
    pub fn isNull(location: Location) bool {
        return c.mlirLocationIsNull(location.getRaw());
    }

    /// Checks if two locations are equal.
    pub fn eql(location: Location, other: Location) bool {
        return c.mlirLocationEqual(location.getRaw(), other.getRaw());
    }

    /// Prints a location by sending chunks of the string representation and
    /// forwarding `userData to `callback`. Note that the callback may be called
    /// several times with consecutive chunks of the string.
    pub fn print(location: Location, callback: StringCallback, user_data: *anyopaque) void {
        c.mlirLocationPrint(location.getRaw(), callback, user_data);
    }
};

//===----------------------------------------------------------------------===//
// Module API.
//===----------------------------------------------------------------------===//
pub const Module = struct {
    _: c.MlirModule,

    pub fn fromRaw(raw: c.MlirModule) Module {
        return .{ ._ = raw };
    }

    pub fn getRaw(module: Module) c.MlirModule {
        return module._;
    }

    /// Creates a new, empty module and transfers ownership to the caller.
    pub fn initEmpty(location: Location) Module {
        return Module.fromRaw(c.mlirModuleCreateEmpty(location.getRaw()));
    }

    /// Parses a module from the string and transfers ownership to the caller.
    pub fn initParse(context: Context, module: StringRef) Module {
        return Module.fromRaw(c.mlirModuleCreateParse(context.getRaw(), module.getRaw()));
    }

    /// Gets the context that a module was created with.
    pub fn getContext(module: Module) Context {
        return Context.fromRaw(c.mlirModuleGetContext(module.getRaw()));
    }

    /// Gets the body of the module, i.e. the only block it contains.
    pub fn getBody(module: Module) Block {
        return Block.fromRaw(c.mlirModuleGetBody(module.getRaw()));
    }

    /// Checks whether a module is null.
    pub fn isNull(module: Module) bool {
        return c.mlirModuleIsNull(module.getRaw());
    }

    /// Takes a module owned by the caller and deletes it.
    pub fn deinit(module: Module) void {
        c.mlirModuleDestroy(module.getRaw());
    }

    /// Views the module as a generic operation.
    pub fn getOperation(module: Module) Operation {
        return Operation.fromRaw(c.mlirModuleGetOperation(module.getRaw()));
    }

    /// Views the generic operation as a module.
    /// The returned module is null when the input operation was not a ModuleOp.
    pub fn fromOperation(op: Operation) Module {
        return Module.fromRaw(c.mlirModuleFromOperation(op.getRaw()));
    }
};

//===----------------------------------------------------------------------===//
// Operation state.
//===----------------------------------------------------------------------===//
/// An auxiliary class for constructing operations.
///
/// This class contains all the information necessary to construct the
/// operation. It owns the MlirRegions it has pointers to and does not own
/// anything else. By default, the state can be constructed from a name and
/// location, the latter being also used to access the context, and has no other
/// components. These components can be added progressively until the operation
/// is constructed. Users are not expected to rely on the internals of this
/// class and should use mlirOperationState* functions instead.
pub const OperationState = struct {
    _: c.MlirOperationState,

    pub fn fromRaw(raw: c.MlirOperationState) OperationState {
        return .{ ._ = raw };
    }

    pub fn getRaw(operation_state: OperationState) c.MlirOperationState {
        return operation_state._;
    }

    /// Constructs an operation state from a name and a location.
    pub fn get(name: StringRef, loc: Location) OperationState {
        return OperationState.fromRaw(c.mlirOperationStateGet(name.getRaw(), loc.getRaw()));
    }

    /// Adds a list of components to the operation state.
    pub fn addResults(state: *OperationState, results: []const Type) void {
        var raw_state = state.getRaw();
        c.mlirOperationStateAddResults(&raw_state, @intCast(results.len), @ptrCast(results.ptr));
    }
    pub fn addOperands(state: *OperationState, operands: []const Value) void {
        var raw_state = state.getRaw();
        c.mlirOperationStateAddOperands(&raw_state, @intCast(operands.len), @ptrCast(operands.ptr));
    }
    pub fn addOwnedRegions(state: *OperationState, regions: []const Region) void {
        var raw_state = state.getRaw();
        c.mlirOperationStateAddOwnedRegions(&raw_state, @intCast(regions.len), @ptrCast(regions.ptr));
    }
    pub fn addSuccessors(state: *OperationState, successors: []const Block) void {
        var raw_state = state.getRaw();
        c.mlirOperationStateAddSuccessors(&raw_state, @intCast(successors.len), @ptrCast(successors.ptr));
    }
    pub fn addAttributes(state: *OperationState, attributes: []const NamedAttribute) void {
        var raw_state = state.getRaw();
        c.mlirOperationStateAddAttributes(&raw_state, @intCast(attributes.len), @ptrCast(attributes.ptr));
    }

    /// Enables result type inference for the operation under construction. If
    /// enabled, then the caller must not have called
    /// mlirOperationStateAddResults(). Note that if enabled, the
    /// mlirOperationCreate() call is failable: it will return a null operation
    /// on inference failure and will emit diagnostics.
    pub fn enableResultTypeInference(state: *OperationState) void {
        var raw_state = state.getRaw();
        c.mlirOperationStateEnableResultTypeInference(&raw_state);
    }
};

//===----------------------------------------------------------------------===//
// AsmState API.
// While many of these are simple settings that could be represented in a
// struct, they are wrapped in a heap allocated object and accessed via
// functions to maximize the possibility of compatibility over time.
//===----------------------------------------------------------------------===//
pub const AsmState = struct {
    _: c.MlirAsmState,

    pub fn fromRaw(raw: c.MlirAsmState) AsmState {
        return .{ ._ = raw };
    }

    pub fn getRaw(asm_state: AsmState) c.MlirAsmState {
        return asm_state._;
    }

    /// Creates new AsmState, as with AsmState the IR should not be mutated
    /// in-between using this state.
    /// Must be freed with a call to mlirAsmStateDestroy().
    // TODO: This should be expanded to handle location & resouce map.
    pub fn initForOperation(op: Operation, flags: OpPrintingFlags) AsmState {
        return AsmState.fromRaw(c.mlirAsmStateCreateForOperation(op.getRaw(), flags.getRaw()));
    }

    /// Creates new AsmState from value.
    /// Must be freed with a call to mlirAsmStateDestroy().
    // TODO: This should be expanded to handle location & resouce map.
    pub fn initForValue(value: Value, flags: OpPrintingFlags) AsmState {
        return AsmState.fromRaw(c.mlirAsmStateCreateForValue(value.getRaw(), flags.getRaw()));
    }

    /// Destroys printing flags created with mlirAsmStateCreate.
    pub fn deinit(state: AsmState) void {
        c.mlirAsmStateDestroy(state.getRaw());
    }
};

//===----------------------------------------------------------------------===//
// Op Printing flags API.
// While many of these are simple settings that could be represented in a
// struct, they are wrapped in a heap allocated object and accessed via
// functions to maximize the possibility of compatibility over time.
//===----------------------------------------------------------------------===//
pub const OpPrintingFlags = struct {
    _: c.MlirOpPrintingFlags,

    pub fn fromRaw(raw: c.MlirOpPrintingFlags) OpPrintingFlags {
        return .{ ._ = raw };
    }

    pub fn getRaw(flags: OpPrintingFlags) c.MlirOpPrintingFlags {
        return flags._;
    }

    /// Creates new printing flags with defaults, intended for customization.
    /// Must be freed with a call to mlirOpPrintingFlagsDestroy().
    pub fn init() OpPrintingFlags {
        return OpPrintingFlags.fromRaw(c.mlirOpPrintingFlagsCreate());
    }

    /// Destroys printing flags created with mlirOpPrintingFlagsCreate.
    pub fn deinit(flags: OpPrintingFlags) void {
        c.mlirOpPrintingFlagsDestroy(flags.getRaw());
    }

    /// Enables the elision of large elements attributes by printing a lexically
    /// valid but otherwise meaningless form instead of the element data. The
    /// `largeElementLimit` is used to configure what is considered to be a "large"
    /// ElementsAttr by providing an upper limit to the number of elements.
    pub fn elideLargeElementsAttrs(flags: OpPrintingFlags, largeElementLimit: isize) void {
        c.mlirOpPrintingFlagsElideLargeElementsAttrs(flags.getRaw(), largeElementLimit);
    }

    /// Enable or disable printing of debug information (based on `enable`). If
    /// 'prettyForm' is set to true, debug information is printed in a more readable
    /// 'pretty' form. Note: The IR generated with 'prettyForm' is not parsable.
    pub fn enableDebugInfo(flags: OpPrintingFlags, enable: bool, pretty_form: bool) void {
        c.mlirOpPrintingFlagsEnableDebugInfo(flags.getRaw(), enable, pretty_form);
    }

    /// Always print operations in the generic form.
    pub fn printGenericOpForm(flags: OpPrintingFlags) void {
        c.mlirOpPrintingFlagsPrintGenericOpForm(flags.getRaw());
    }

    /// Use local scope when printing the operation. This allows for using the
    /// printer in a more localized and thread-safe setting, but may not
    /// necessarily be identical to what the IR will look like when dumping
    /// the full module.
    pub fn useLocalScope(flags: OpPrintingFlags) void {
        c.mlirOpPrintingFlagsUseLocalScope(flags.getRaw());
    }

    /// Do not verify the operation when using custom operation printers.
    pub fn assumeVerified(flags: OpPrintingFlags) void {
        c.mlirOpPrintingFlagsAssumeVerified(flags.getRaw());
    }
};

//===----------------------------------------------------------------------===//
// Bytecode printing flags API.
//===----------------------------------------------------------------------===//
pub const BytecodeWriterConfig = struct {
    _: c.MlirBytecodeWriterConfig,

    pub fn fromRaw(raw: c.MlirBytecodeWriterConfig) BytecodeWriterConfig {
        return .{ ._ = raw };
    }

    pub fn getRaw(config: BytecodeWriterConfig) c.MlirBytecodeWriterConfig {
        return config._;
    }

    /// Creates new printing flags with defaults, intended for customization.
    /// Must be freed with a call to mlirBytecodeWriterConfigDestroy().
    pub fn init() BytecodeWriterConfig {
        return BytecodeWriterConfig.fromRaw(c.mlirBytecodeWriterConfigCreate());
    }

    /// Destroys printing flags created with mlirBytecodeWriterConfigCreate.
    pub fn deinit(config: BytecodeWriterConfig) void {
        c.mlirBytecodeWriterConfigDestroy(config.getRaw());
    }

    /// Sets the version to emit in the writer config.
    pub fn desiredEmitVersion(flags: BytecodeWriterConfig, version: i64) void {
        c.mlirBytecodeWriterConfigDesiredEmitVersion(flags.getRaw(), version);
    }
};

//===----------------------------------------------------------------------===//
// Operation API.
//===----------------------------------------------------------------------===//
pub const Operation = struct {
    _: c.MlirOperation,

    pub fn fromRaw(raw: c.MlirOperation) Operation {
        return .{ ._ = raw };
    }

    pub fn getRaw(operation: Operation) c.MlirOperation {
        return operation._;
    }

    /// Creates an operation and transfers ownership to the caller.
    /// Note that caller owned child objects are transferred in this call and must
    /// not be further used. Particularly, this applies to any regions added to
    /// the state (the implementation may invalidate any such pointers).
    ///
    /// This call can fail under the following conditions, in which case, it will
    /// return a null operation and emit diagnostics:
    ///   - Result type inference is enabled and cannot be performed.
    pub fn init(state: *OperationState) Operation {
        var raw_state = state.getRaw();
        return Operation.fromRaw(c.mlirOperationCreate(&raw_state));
    }

    /// Parses an operation, giving ownership to the caller. If parsing fails a null
    /// operation will be returned, and an error diagnostic emitted.
    ///
    /// `sourceStr` may be either the text assembly format, or binary bytecode
    /// format. `sourceName` is used as the file name of the source; any IR without
    /// locations will get a `FileLineColLoc` location with `sourceName` as the file
    /// name.
    pub fn initParse(context: Context, source_str: StringRef, source_name: StringRef) Operation {
        return Operation.fromRaw(c.mlirOperationCreateParse(context.getRaw(), source_str.getRaw(), source_name.getRaw()));
    }

    /// Creates a deep copy of an operation. The operation is not inserted and
    /// ownership is transferred to the caller.
    pub fn clone(op: Operation) Operation {
        return Operation.fromRaw(c.mlirOperationClone(op.getRaw()));
    }

    /// Takes an operation owned by the caller and destroys it.
    pub fn deinit(op: Operation) void {
        c.mlirOperationDestroy(op.getRaw());
    }

    /// Removes the given operation from its parent block. The operation is not
    /// destroyed. The ownership of the operation is transferred to the caller.
    pub fn removeFromParent(op: Operation) void {
        c.mlirOperationRemoveFromParent(op.getRaw());
    }

    /// Checks whether the underlying operation is null.
    pub fn isNull(op: Operation) bool {
        return c.mlirOperationIsNull(op.getRaw());
    }

    /// Checks whether two operation handles point to the same operation. This does
    /// not perform deep comparison.
    pub fn eql(op: Operation, other: Operation) bool {
        return c.mlirOperationEqual(op.getRaw(), other.getRaw());
    }

    /// Gets the context this operation is associated with
    pub fn getContext(op: Operation) Context {
        return Context.fromRaw(c.mlirOperationGetContext(op.getRaw()));
    }

    /// Gets the location of the operation.
    pub fn getLocation(op: Operation) Location {
        return Location.fromRaw(c.mlirOperationGetLocation(op.getRaw()));
    }

    /// Gets the type id of the operation.
    /// Returns null if the operation does not have a registered operation
    /// description.
    pub fn getTypeID(op: Operation) ?TypeID {
        const type_id = c.mlirOperationGetTypeID(op.getRaw());
        return if (type_id.ptr) |ptr|
            TypeID.fromRaw(.{ .ptr = ptr })
        else
            null;
    }

    /// Gets the name of the operation as an identifier.
    pub fn getName(op: Operation) Identifier {
        return Identifier.fromRaw(c.mlirOperationGetName(op.getRaw()));
    }

    /// Gets the block that owns this operation, returning null if the operation is
    /// not owned.
    pub fn getBlock(op: Operation) ?Block {
        const block = c.mlirOperationGetBlock(op.getRaw());
        return if (block.ptr) |ptr|
            Block.fromRaw(.{ .ptr = ptr })
        else
            null;
    }

    /// Gets the operation that owns this operation, returning null if the operation
    /// is not owned.
    pub fn getParentOperation(op: Operation) ?Operation {
        const parent = c.mlirOperationGetParentOperation(op.getRaw());
        return if (parent.ptr) |ptr|
            Operation.fromRaw(.{ .ptr = ptr })
        else
            null;
    }

    /// Returns the number of regions attached to the given operation.
    pub fn getNumRegions(op: Operation) isize {
        return c.mlirOperationGetNumRegions(op.getRaw());
    }

    /// Returns `pos`-th region attached to the operation.
    pub fn getRegion(op: Operation, pos: isize) Region {
        return Region.fromRaw(c.mlirOperationGetRegion(op.getRaw(), pos));
    }

    /// Returns an operation immediately following the given operation it its
    /// enclosing block.
    pub fn getNextInBlock(op: Operation) Operation {
        return Operation.fromRaw(c.mlirOperationGetNextInBlock(op.getRaw()));
    }

    /// Returns the number of operands of the operation.
    pub fn getNumOperands(op: Operation) isize {
        return c.mlirOperationGetNumOperands(op.getRaw());
    }

    /// Returns `pos`-th operand of the operation.
    pub fn getOperand(op: Operation, pos: isize) Value {
        return Value.fromRaw(c.mlirOperationGetOperand(op.getRaw(), pos));
    }

    /// Sets the `pos`-th operand of the operation.
    pub fn setOperand(op: Operation, pos: isize, new_value: Value) void {
        c.mlirOperationSetOperand(op.getRaw(), pos, new_value.getRaw());
    }

    /// Replaces the operands of the operation.
    pub fn setOperands(op: Operation, operands: []const Value) void {
        c.mlirOperationSetOperands(op.getRaw(), @intCast(operands.len), @ptrCast(operands));
    }

    /// Returns the number of results of the operation.
    pub fn getNumResults(op: Operation) isize {
        return c.mlirOperationGetNumResults(op.getRaw());
    }

    /// Returns `pos`-th result of the operation.
    pub fn getResult(op: Operation, pos: isize) Value {
        return Value.fromRaw(c.mlirOperationGetResult(op.getRaw(), pos));
    }

    /// Returns the number of successor blocks of the operation.
    pub fn getNumSuccessors(op: Operation) isize {
        return c.mlirOperationGetNumSuccessors(op.getRaw());
    }

    /// Returns `pos`-th successor of the operation.
    pub fn getSuccessor(op: Operation, pos: isize) Block {
        return Block.fromRaw(c.mlirOperationGetSuccessor(op.getRaw(), pos));
    }

    /// Set `pos`-th successor of the operation.
    pub fn setSuccessor(op: Operation, pos: isize, block: Block) void {
        _ = op; // autofix
        _ = pos; // autofix
        _ = block; // autofix
    }

    /// Returns true if this operation defines an inherent attribute with this name.
    /// Note: the attribute can be optional, so
    /// `mlirOperationGetInherentAttributeByName` can still return a null attribute.
    pub fn hasInherentAttributeByName(op: Operation, name: StringRef) bool {
        return c.mlirOperationHasInherentAttributeByName(op.getRaw(), name.getRaw());
    }

    /// Returns an inherent attribute attached to the operation given its name.
    pub fn getInherentAttributeByName(op: Operation, name: StringRef) Attribute {
        return Attribute.fromRaw(c.mlirOperationGetInherentAttributeByName(op.getRaw(), name.getRaw()));
    }

    /// Sets an inherent attribute by name, replacing the existing if it exists.
    /// This has no effect if "name" does not match an inherent attribute.
    pub fn setInherentAttributeByName(op: Operation, name: StringRef, attr: Attribute) void {
        c.mlirOperationSetInherentAttributeByName(op.getRaw(), name.getRaw(), attr.getRaw());
    }

    /// Returns the number of discardable attributes attached to the operation.
    pub fn getNumDiscardableAttributes(op: Operation) isize {
        return c.mlirOperationGetNumDiscardableAttributes(op.getRaw());
    }

    /// Return `pos`-th discardable attribute of the operation.
    pub fn getDiscardableAttribute(op: Operation, pos: isize) NamedAttribute {
        return NamedAttribute.fromRaw(c.mlirOperationGetDiscardableAttribute(op.getRaw(), pos));
    }

    /// Returns a discardable attribute attached to the operation given its name.
    pub fn getDiscardableAttributeByName(op: Operation, name: StringRef) Attribute {
        return Attribute.fromRaw(c.mlirOperationGetDiscardableAttributeByName(op.getRaw(), name.getRaw()));
    }

    /// Sets a discardable attribute by name, replacing the existing if it exists or
    /// adding a new one otherwise. The new `attr` Attribute is not allowed to be
    /// null, use `mlirOperationRemoveDiscardableAttributeByName` to remove an
    /// Attribute instead.
    pub fn setDiscardableAttributeByName(op: Operation, name: StringRef, attr: Attribute) void {
        c.mlirOperationSetDiscardableAttributeByName(op.getRaw(), name.getRaw(), attr.getRaw());
    }

    /// Removes a discardable attribute by name. Returns false if the attribute was
    /// not found and true if removed.
    pub fn removeDiscardableAttributeByName(op: Operation, name: StringRef) bool {
        return c.mlirOperationRemoveDiscardableAttributeByName(op.getRaw(), name.getRaw());
    }

    /// Returns the number of attributes attached to the operation.
    /// Deprecated, please use `mlirOperationGetNumInherentAttributes` or
    /// `mlirOperationGetNumDiscardableAttributes`.
    pub fn getNumAttributes(op: Operation) isize {
        return c.mlirOperationGetNumAttributes(op.getRaw());
    }

    /// Return `pos`-th attribute of the operation.
    /// Deprecated, please use `mlirOperationGetInherentAttribute` or
    /// `mlirOperationGetDiscardableAttribute`.
    pub fn getAttribute(op: Operation, pos: isize) NamedAttribute {
        return NamedAttribute.fromRaw(c.mlirOperationGetAttribute(op.getRaw(), pos));
    }

    /// Returns an attribute attached to the operation given its name.
    /// Deprecated, please use `mlirOperationGetInherentAttributeByName` or
    /// `mlirOperationGetDiscardableAttributeByName`.
    pub fn getAttributeByName(op: Operation, name: StringRef) Attribute {
        return Attribute.fromRaw(c.mlirOperationGetAttributeByName(op.getRaw(), name.getRaw()));
    }

    /// Sets an attribute by name, replacing the existing if it exists or
    /// adding a new one otherwise.
    /// Deprecated, please use `mlirOperationSetInherentAttributeByName` or
    /// `mlirOperationSetDiscardableAttributeByName`.
    pub fn setAttributeByName(op: Operation, name: StringRef, attr: Attribute) void {
        c.mlirOperationSetAttributeByName(op.getRaw(), name.getRaw(), attr.getRaw());
    }

    /// Removes an attribute by name. Returns false if the attribute was not found
    /// and true if removed.
    /// Deprecated, please use `mlirOperationRemoveInherentAttributeByName` or
    /// `mlirOperationRemoveDiscardableAttributeByName`.
    pub fn removeAttributeByName(op: Operation, name: StringRef) bool {
        return c.mlirOperationRemoveAttributeByName(op.getRaw(), name.getRaw());
    }

    /// Prints an operation by sending chunks of the string representation and
    /// forwarding `userData to `callback`. Note that the callback may be called
    /// several times with consecutive chunks of the string.
    pub fn print(op: Operation, callback: StringCallback, user_data: *anyopaque) void {
        c.mlirOperationPrint(op.getRaw(), callback, user_data);
    }

    /// Same as mlirOperationPrint but accepts flags controlling the printing
    /// behavior.
    pub fn printWithFlags(op: Operation, flags: OpPrintingFlags, callback: StringCallback, user_data: *anyopaque) void {
        c.mlirOperationPrintWithFlags(op.getRaw(), flags.getRaw(), callback, user_data);
    }

    /// Same as mlirOperationPrint but accepts AsmState controlling the printing
    /// behavior as well as caching computed names.
    pub fn printWithState(op: Operation, state: AsmState, callback: StringCallback, user_data: *anyopaque) void {
        c.mlirOperationPrintWithState(op.getRaw(), state.getRaw(), callback, user_data);
    }

    /// Same as mlirOperationPrint but writing the bytecode format.
    pub fn writeBytecode(op: Operation, callback: StringCallback, user_data: *anyopaque) void {
        c.mlirOperationWriteBytecode(op.getRaw(), callback, user_data);
    }

    /// Same as mlirOperationWriteBytecode but with writer config and returns
    /// failure only if desired bytecode could not be honored.
    pub fn writeBytecodeWithConfig(op: Operation, config: BytecodeWriterConfig, callback: StringCallback, user_data: *anyopaque) LogicalResult {
        return LogicalResult.fromRaw(c.mlirOperationWriteBytecodeWithConfig(op.getRaw(), config.getRaw(), callback, user_data));
    }

    /// Prints an operation to stderr.
    pub fn dump(op: Operation) void {
        return c.mlirOperationDump(op.getRaw());
    }

    /// Verify the operation and return true if it passes, false if it fails.
    pub fn verify(op: Operation) bool {
        return c.mlirOperationVerify(op.getRaw());
    }

    /// Moves the given operation immediately after the other operation in its
    /// parent block. The given operation may be owned by the caller or by its
    /// current block. The other operation must belong to a block. In any case, the
    /// ownership is transferred to the block of the other operation.
    pub fn moveAfter(op: Operation, other: Operation) void {
        c.mlirOperationMoveAfter(op.getRaw(), other.getRaw());
    }

    /// Moves the given operation immediately before the other operation in its
    /// parent block. The given operation may be owner by the caller or by its
    /// current block. The other operation must belong to a block. In any case, the
    /// ownership is transferred to the block of the other operation.
    pub fn moveBefore(op: Operation, other: Operation) void {
        c.mlirOperationMoveBefore(op.getRaw(), other.getRaw());
    }

    /// Traversal order for operation walk.
    pub const MlirWalkOrder = enum(c.MlirWalkOrder) {
        pre_order = c.MlirWalkPreOrder,
        post_order = c.MlirWalkPostOrder,
    };

    // /// Operation walker type. The handler is passed an (opaque) reference to an
    // /// operation and a pointer to a `userData`.
    pub const MlirOperationWalkCallback = c.MlirOperationWalkCallback;

    /// Walks operation `op` in `walkOrder` and calls `callback` on that operation.
    /// `*userData` is passed to the callback as well and can be used to tunnel some
    /// context or other data into the callback.
    pub fn walk(op: Operation, callback: MlirOperationWalkCallback, user_data: *anyopaque, walk_order: MlirWalkOrder) void {
        c.mlirOperationWalk(op.getRaw(), callback, user_data, @intFromEnum(walk_order));
    }

    /// Returns first region attached to the operation.
    pub fn getFirstRegion(op: Operation) Region {
        return Region.fromRaw(c.mlirOperationGetFirstRegion(op.getRaw()));
    }
};

//===----------------------------------------------------------------------===//
// Region API.
//===----------------------------------------------------------------------===//
pub const Region = struct {
    _: c.MlirRegion,

    pub fn fromRaw(raw: c.MlirRegion) Region {
        return .{ ._ = raw };
    }

    pub fn getRaw(region: Region) c.MlirRegion {
        return region._;
    }

    /// Creates a new empty region and transfers ownership to the caller.
    pub fn init() Region {
        return Region.fromRaw(c.mlirRegionCreate());
    }

    /// Takes a region owned by the caller and destroys it.
    pub fn deinit(region: Region) void {
        c.mlirRegionDestroy(region.getRaw());
    }

    /// Checks whether a region is null.
    pub fn isNull(region: Region) bool {
        return c.mlirRegionIsNull(region.getRaw());
    }

    /// Checks whether two region handles point to the same region. This does not
    /// perform deep comparison.
    pub fn eql(region: Region, other: Region) bool {
        return c.mlirRegionEqual(region.getRaw(), other.getRaw());
    }

    /// Gets the first block in the region.
    pub fn getFirstBlock(region: Region) Block {
        return Block.fromRaw(c.mlirRegionGetFirstBlock(region.getRaw()));
    }

    /// Takes a block owned by the caller and appends it to the given region.
    pub fn appendOwnedBlock(region: Region, block: Block) void {
        c.mlirRegionAppendOwnedBlock(region.getRaw(), block.getRaw());
    }

    /// Takes a block owned by the caller and inserts it at `pos` to the given
    /// region. This is an expensive operation that linearly scans the region,
    /// prefer insertAfter/Before instead.
    pub fn insertOwnedBlock(region: Region, pos: isize, block: Block) void {
        c.mlirRegionInsertOwnedBlock(region.getRaw(), pos, block.getRaw());
    }

    /// Takes a block owned by the caller and inserts it after the (non-owned)
    /// reference block in the given region. The reference block must belong to the
    /// region. If the reference block is null, prepends the block to the region.
    pub fn insertOwnedBlockAfter(region: Region, reference: Block, block: Block) void {
        c.mlirRegionInsertOwnedBlockAfter(region.getRaw(), reference.getRaw(), block.getRaw());
    }

    /// Takes a block owned by the caller and inserts it before the (non-owned)
    /// reference block in the given region. The reference block must belong to the
    /// region. If the reference block is null, appends the block to the region.
    pub fn insertOwnedBlockBefore(region: Region, reference: Block, block: Block) void {
        c.mlirRegionInsertOwnedBlockBefore(region.getRaw(), reference.getRaw(), block.getRaw());
    }

    /// Returns the region immediately following the given region in its parent
    /// operation.
    pub fn getNextInOperation(region: Region) Region {
        return Region.fromRaw(c.mlirRegionGetNextInOperation(region.getRaw()));
    }

    /// Moves the entire content of the source region to the target region.
    pub fn takeBody(target: Region, source: Region) void {
        c.mlirRegionTakeBody(target.getRaw(), source.getRaw());
    }
};

//===----------------------------------------------------------------------===//
// Block API.
//===----------------------------------------------------------------------===//
pub const Block = struct {
    _: c.MlirBlock,

    pub fn fromRaw(raw: c.MlirBlock) Block {
        return .{ ._ = raw };
    }

    pub fn getRaw(block: Block) c.MlirBlock {
        return block._;
    }

    /// Creates a new empty block with the given argument types and transfers
    /// ownership to the caller.
    pub fn init(args: []const Type, locs: []const Location) Block {
        assert(args.len == locs.len);
        return Block.fromRaw(c.mlirBlockCreate(@intCast(args.len), @ptrCast(args.ptr), @ptrCast(locs.ptr)));
    }

    /// Takes a block owned by the caller and destroys it.
    pub fn deinit(block: Block) void {
        c.mlirBlockDestroy(block.getRaw());
    }

    /// Detach a block from the owning region and assume ownership.
    pub fn detach(block: Block) void {
        c.mlirBlockDetach(block.getRaw());
    }

    /// Checks whether a block is null.
    pub fn isNull(block: Block) bool {
        return c.mlirBlockIsNull(block.getRaw());
    }

    /// Checks whether two blocks handles point to the same block. This does not
    /// perform deep comparison.
    pub fn eql(block: Block, other: Block) bool {
        return c.mlirBlockEqual(block.getRaw(), other.getRaw());
    }

    /// Returns the closest surrounding operation that contains this block.
    pub fn getParentOperation(block: Block) Operation {
        return Operation.fromRaw(c.mlirBlockGetParentOperation(block.getRaw()));
    }

    /// Returns the region that contains this block.
    pub fn getParentRegion(block: Block) Region {
        return Region.fromRaw(c.mlirBlockGetParentRegion(block.getRaw()));
    }

    /// Returns the block immediately following the given block in its parent
    /// region.
    pub fn getNextInRegion(block: Block) Block {
        return Block.fromRaw(c.mlirBlockGetNextInRegion(block.getRaw()));
    }

    /// Returns the first operation in the block.
    pub fn getFirstOperation(block: Block) Operation {
        return Operation.fromRaw(c.mlirBlockGetFirstOperation(block.getRaw()));
    }

    /// Returns the terminator operation in the block or null if no terminator.
    pub fn getTerminator(block: Block) Operation {
        return Operation.fromRaw(c.mlirBlockGetTerminator(block.getRaw()));
    }

    /// Takes an operation owned by the caller and appends it to the block.
    pub fn appendOwnedOperation(block: Block, operation: Operation) void {
        c.mlirBlockAppendOwnedOperation(block.getRaw(), operation.getRaw());
    }

    /// Takes an operation owned by the caller and inserts it as `pos` to the block.
    /// This is an expensive operation that scans the block linearly, prefer
    /// insertBefore/After instead.
    pub fn insertOwnedOperation(block: Block, pos: isize, operation: Operation) void {
        c.mlirBlockInsertOwnedOperation(block.getRaw(), pos, operation.getRaw());
    }

    /// Takes an operation owned by the caller and inserts it after the (non-owned)
    /// reference operation in the given block. If the reference is null, prepends
    /// the operation. Otherwise, the reference must belong to the block.
    pub fn insertOwnedOperationAfter(block: Block, reference: Operation, operation: Operation) void {
        c.mlirBlockInsertOwnedOperationAfter(block.getRaw(), reference.getRaw(), operation.getRaw());
    }

    /// Takes an operation owned by the caller and inserts it before the (non-owned)
    /// reference operation in the given block. If the reference is null, appends
    /// the operation. Otherwise, the reference must belong to the block.
    pub fn insertOwnedOperationBefore(block: Block, reference: Operation, operation: Operation) void {
        c.mlirBlockInsertOwnedOperationBefore(block.getRaw(), reference.getRaw(), operation.getRaw());
    }

    /// Returns the number of arguments of the block.
    pub fn getNumArguments(block: Block) isize {
        return c.mlirBlockGetNumArguments(block.getRaw());
    }

    /// Appends an argument of the specified type to the block. Returns the newly
    /// added argument.
    pub fn addArgument(block: Block, @"type": Type, loc: Location) Value {
        return Value.fromRaw(c.mlirBlockAddArgument(block.getRaw(), @"type".getRaw(), loc.getRaw()));
    }

    /// Inserts an argument of the specified type at a specified index to the block.
    /// Returns the newly added argument.
    pub fn insertArgument(block: Block, pos: isize, @"type": Type, loc: Location) Value {
        return Value.fromRaw(c.mlirBlockInsertArgument(block.getRaw(), pos, @"type".getRaw(), loc.getRaw()));
    }

    /// Returns `pos`-th argument of the block.
    pub fn getArgument(block: Block, pos: isize) Value {
        return Value.fromRaw(c.mlirBlockGetArgument(block.getRaw(), pos));
    }

    /// Prints a block by sending chunks of the string representation and
    /// forwarding `userData to `callback`. Note that the callback may be called
    /// several times with consecutive chunks of the string.
    pub fn print(block: Block, callback: StringCallback, user_data: *anyopaque) void {
        c.mlirBlockPrint(block.getRaw(), callback, user_data);
    }
};

//===----------------------------------------------------------------------===//
// Value API.
//===----------------------------------------------------------------------===//
pub const Value = struct {
    _: c.MlirValue,

    pub fn fromRaw(raw: c.MlirValue) Value {
        return .{ ._ = raw };
    }

    pub fn getRaw(value: Value) c.MlirValue {
        return value._;
    }

    /// Returns whether the value is null.
    pub fn isNull(value: Value) bool {
        return c.mlirValueIsNull(value.getRaw());
    }

    /// Returns 1 if two values are equal, 0 otherwise.
    pub fn eql(value: Value, other: Value) bool {
        return c.mlirValueEqual(value.getRaw(), other.getRaw());
    }

    /// Returns 1 if the value is a block argument, 0 otherwise.
    pub fn isABlockArgument(value: Value) bool {
        return c.mlirValueIsABlockArgument(value.getRaw());
    }

    /// Returns 1 if the value is an operation result, 0 otherwise.
    pub fn isAOpResult(value: Value) bool {
        return c.mlirValueIsAOpResult(value.getRaw());
    }

    /// Returns the block in which this value is defined as an argument. Asserts if
    /// the value is not a block argument.
    pub fn blockArgumentGetOwner(value: Value) Block {
        return Block.fromRaw(c.mlirBlockArgumentGetOwner(value.getRaw()));
    }

    /// Returns the position of the value in the argument list of its block.
    pub fn blockArgumentGetArgNumber(value: Value) isize {
        return c.mlirBlockArgumentGetArgNumber(value.getRaw());
    }

    /// Sets the type of the block argument to the given type.
    pub fn blockArgumentSetType(value: Value, @"type": Type) void {
        c.mlirBlockArgumentSetType(value.getRaw(), @"type".getRaw());
    }

    /// Returns an operation that produced this value as its result. Asserts if the
    /// value is not an op result.
    pub fn mlirOpResultGetOwner(value: Value) Operation {
        return Operation.fromRaw(c.mlirOpResultGetOwner(value.getRaw()));
    }

    /// Returns the position of the value in the list of results of the operation
    /// that produced it.
    pub fn opResultGetResultNumber(value: Value) isize {
        return c.mlirOpResultGetResultNumber(value.getRaw());
    }

    /// Returns the type of the value.
    pub fn getType(value: Value) Type {
        return Type.fromRaw(c.mlirValueGetType(value.getRaw()));
    }

    /// Set the type of the value.
    pub fn setType(value: Value, @"type": Type) void {
        c.mlirValueSetType(value.getRaw(), @"type".getRaw());
    }

    /// Prints the value to the standard error stream.
    pub fn dump(value: Value) void {
        c.mlirValueDump(value.getRaw());
    }

    /// Prints a value by sending chunks of the string representation and
    /// forwarding `userData to `callback`. Note that the callback may be called
    /// several times with consecutive chunks of the string.
    pub fn print(value: Value, callback: StringCallback, user_data: *anyopaque) void {
        c.mlirValuePrint(value.getRaw(), callback, user_data);
    }

    /// Prints a value as an operand (i.e., the ValueID).
    pub fn printAsOperand(value: Value, state: AsmState, callback: StringCallback, user_data: *anyopaque) void {
        c.mlirValuePrintAsOperand(value.getRaw(), state.getRaw(), callback, user_data);
    }

    /// Returns an op operand representing the first use of the value, or a null op
    /// operand if there are no uses.
    pub fn getFirstUse(value: Value) OpOperand {
        return OpOperand.fromRaw(c.mlirValueGetFirstUse(value.getRaw()));
    }

    /// Replace all uses of 'of' value with the 'with' value, updating anything in
    /// the IR that uses 'of' to use the other value instead.  When this returns
    /// there are zero uses of 'of'.
    pub fn replaceAllUsesOfWith(of: Value, with: Value) void {
        c.mlirValueReplaceAllUsesOfWith(of.getRaw(), with.getRaw());
    }
};

//===----------------------------------------------------------------------===//
// OpOperand API.
//===----------------------------------------------------------------------===//
pub const OpOperand = struct {
    _: c.MlirOpOperand,

    pub fn fromRaw(raw: c.MlirOpOperand) OpOperand {
        return .{ ._ = raw };
    }

    pub fn getRaw(op_operand: OpOperand) c.MlirOpOperand {
        return op_operand._;
    }

    /// Returns whether the op operand is null.
    pub fn isNull(op_operand: OpOperand) bool {
        return c.mlirOpOperandIsNull(op_operand.getRaw());
    }

    /// Returns the owner operation of an op operand.
    pub fn getOwner(op_operand: OpOperand) Operation {
        return Operation.fromRaw(c.mlirOpOperandGetOwner(op_operand.getRaw()));
    }

    /// Returns the operand number of an op operand.
    pub fn getOperandNumber(op_operand: OpOperand) usize {
        return c.mlirOpOperandGetOperandNumber(op_operand.getRaw());
    }

    /// Returns an op operand representing the next use of the value, or a null op
    /// operand if there is no next use.
    pub fn getNextUse(op_operand: OpOperand) OpOperand {
        return OpOperand.fromRaw(c.mlirOpOperandGetNextUse(op_operand.getRaw()));
    }
};

//===----------------------------------------------------------------------===//
// Type API.
//===----------------------------------------------------------------------===//
pub const Type = struct {
    _: c.MlirType,

    pub fn fromRaw(raw: c.MlirType) Type {
        return .{ ._ = raw };
    }

    pub fn getRaw(@"type": Type) c.MlirType {
        return @"type"._;
    }

    /// Parses a type. The type is owned by the context.
    pub fn parseGet(context: Context, @"type": StringRef) Type {
        return Type.fromRaw(c.mlirTypeParseGet(context.getRaw(), @"type".getRaw()));
    }

    /// Gets the context that a type was created with.
    pub fn getContext(@"type": Type) Context {
        return Context.fromRaw(c.mlirTypeGetContext(@"type".getRaw()));
    }

    /// Gets the type ID of the type.
    pub fn getTypeID(@"type": Type) TypeID {
        return TypeID.fromRaw(c.mlirTypeGetTypeID(@"type".getRaw()));
    }

    /// Gets the dialect a type belongs to.
    pub fn getDialect(@"type": Type) Dialect {
        return Dialect.fromRaw(c.mlirTypeGetDialect(@"type".getRaw()));
    }

    /// Checks whether a type is null.
    pub fn isNull(@"type": Type) bool {
        return c.mlirTypeIsNull(@"type".getRaw());
    }

    /// Checks if two types are equal.
    pub fn eql(@"type": Type, other: Type) bool {
        return c.mlirTypeEqual(@"type".getRaw(), other.getRaw());
    }

    /// Prints a location by sending chunks of the string representation and
    /// forwarding `userData to `callback`. Note that the callback may be called
    /// several times with consecutive chunks of the string.
    pub fn print(@"type": Type, callback: StringCallback, user_data: *anyopaque) void {
        c.mlirTypePrint(@"type".getRaw(), callback, user_data);
    }

    /// Prints the type to the standard error stream.
    pub fn dump(@"type": Type) void {
        c.mlirTypeDump(@"type".getRaw());
    }
};

//===----------------------------------------------------------------------===//
// Attribute API.
//===----------------------------------------------------------------------===//
pub const Attribute = struct {
    _: c.MlirAttribute,

    pub fn fromRaw(raw: c.MlirAttribute) Attribute {
        return .{ ._ = raw };
    }

    pub fn getRaw(attribute: Attribute) c.MlirAttribute {
        return attribute._;
    }

    /// Parses an attribute. The attribute is owned by the context.
    pub fn parseGet(context: Context, attr: StringRef) Attribute {
        return Attribute.fromRaw(c.mlirAttributeParseGet(context.getRaw(), attr.getRaw()));
    }

    /// Gets the context that an attribute was created with.
    pub fn getContext(attribute: Attribute) Context {
        return Context.fromRaw(c.mlirAttributeGetContext(attribute.getRaw()));
    }

    /// Gets the type of this attribute.
    pub fn getType(attribute: Attribute) Type {
        return Type.fromRaw(c.mlirAttributeGetType(attribute.getRaw()));
    }

    /// Gets the type id of the attribute.
    pub fn getTypeID(attribute: Attribute) TypeID {
        return TypeID.fromRaw(c.mlirAttributeGetTypeID(attribute.getRaw()));
    }

    /// Gets the dialect of the attribute.
    pub fn getDialect(attribute: Attribute) Dialect {
        return Dialect.fromRaw(c.mlirAttributeGetDialect(attribute.getRaw()));
    }

    /// Checks whether an attribute is null.
    pub fn isNull(attr: Attribute) bool {
        return c.mlirAttributeIsNull(attr.getRaw());
    }

    /// Checks if two attributes are equal.
    pub fn eql(attribute: Attribute, other: Attribute) bool {
        return c.mlirAttributeEqual(attribute.getRaw(), other.getRaw());
    }

    /// Prints an attribute by sending chunks of the string representation and
    /// forwarding `userData to `callback`. Note that the callback may be called
    /// several times with consecutive chunks of the string.
    pub fn print(attr: Attribute, callback: StringCallback, userData: *anyopaque) void {
        c.mlirAttributePrint(attr.getRaw(), callback, userData);
    }

    /// Prints the attribute to the standard error stream.
    pub fn dump(attr: Attribute) void {
        c.mlirAttributeDump(attr.getRaw());
    }
};

//===----------------------------------------------------------------------===//
// Identifier API.
//===----------------------------------------------------------------------===//
pub const Identifier = struct {
    _: c.MlirIdentifier,

    pub fn fromRaw(raw: c.MlirIdentifier) Identifier {
        return .{ ._ = raw };
    }

    pub fn getRaw(identifier: Identifier) c.MlirIdentifier {
        return identifier._;
    }

    /// Gets an identifier with the given string value.
    pub fn get(context: Context, string: StringRef) Identifier {
        return Identifier.fromRaw(c.mlirIdentifierGet(context.getRaw(), string.getRaw()));
    }

    /// Returns the context associated with this identifier
    pub fn getContext(ident: Identifier) Context {
        return Context.fromRaw(c.mlirIdentifierGetContext(ident.getRaw()));
    }

    /// Checks whether two identifiers are the same.
    pub fn eql(ident: Identifier, other: Identifier) bool {
        return c.mlirIdentifierEqual(ident.getRaw(), other.getRaw());
    }

    /// Gets the string value of the identifier.
    pub fn str(ident: Identifier) StringRef {
        return StringRef.fromRaw(c.mlirIdentifierStr(ident.getRaw()));
    }
};

//===----------------------------------------------------------------------===//
// Symbol and SymbolTable API.
//===----------------------------------------------------------------------===//
pub const SymbolTable = struct {
    _: c.MlirSymbolTable,

    pub fn fromRaw(raw: c.MlirSymbolTable) SymbolTable {
        return .{ ._ = raw };
    }

    pub fn getRaw(symbol_table: SymbolTable) c.MlirSymbolTable {
        return symbol_table._;
    }

    /// Returns the name of the attribute used to store symbol names compatible with
    /// symbol tables.
    pub fn getSymbolAttributeName() StringRef {
        return StringRef.fromRaw(c.mlirSymbolTableGetSymbolAttributeName());
    }

    /// Returns the name of the attribute used to store symbol visibility.
    pub fn getVisibilityAttributeName() StringRef {
        return StringRef.fromRaw(c.mlirSymbolTableGetVisibilityAttributeName());
    }

    /// Creates a symbol table for the given operation. If the operation does not
    /// have the SymbolTable trait, returns a null symbol table.
    pub fn init(operation: Operation) SymbolTable {
        return SymbolTable.fromRaw(c.mlirSymbolTableCreate(operation.getRaw()));
    }

    /// Returns true if the symbol table is null.
    pub fn isNull(symbol_table: SymbolTable) bool {
        return c.mlirSymbolTableIsNull(symbol_table.getRaw());
    }

    /// Destroys the symbol table created with mlirSymbolTableCreate. This does not
    /// affect the operations in the table.
    pub fn deinit(symbol_table: SymbolTable) void {
        c.mlirSymbolTableDestroy(symbol_table.getRaw());
    }

    /// Looks up a symbol with the given name in the given symbol table and returns
    /// the operation that corresponds to the symbol. If the symbol cannot be found,
    /// returns a null operation.
    pub fn lookup(symbol_table: SymbolTable, name: StringRef) Operation {
        return Operation.fromRaw(c.mlirSymbolTableLookup(symbol_table.getRaw(), name.getRaw()));
    }

    /// Inserts the given operation into the given symbol table. The operation must
    /// have the symbol trait. If the symbol table already has a symbol with the
    /// same name, renames the symbol being inserted to ensure name uniqueness. Note
    /// that this does not move the operation itself into the block of the symbol
    /// table operation, this should be done separately. Returns the name of the
    /// symbol after insertion.
    pub fn insert(symbol_table: SymbolTable, operation: Operation) Attribute {
        return Attribute.fromRaw(c.mlirSymbolTableInsert(symbol_table.getRaw(), operation.getRaw()));
    }

    /// Removes the given operation from the symbol table and erases it.
    pub fn erase(symbol_table: SymbolTable, operation: Operation) void {
        c.mlirSymbolTableErase(symbol_table.getRaw(), operation.getRaw());
    }

    /// Attempt to replace all uses that are nested within the given operation
    /// of the given symbol 'oldSymbol' with the provided 'newSymbol'. This does
    /// not traverse into nested symbol tables. Will fail atomically if there are
    /// any unknown operations that may be potential symbol tables.
    pub fn replaceAllSymbolUses(old_symbol: StringRef, new_symbol: StringRef, from: Operation) LogicalResult {
        return LogicalResult.fromRaw(c.mlirSymbolTableReplaceAllSymbolUses(old_symbol.getRaw(), new_symbol.getRaw(), from.getRaw()));
    }

    pub const WalkSymbolTableCallback = *const fn (c.MlirOperation, bool, *anyopaque) callconv(.C) void;

    /// Walks all symbol table operations nested within, and including, `op`. For
    /// each symbol table operation, the provided callback is invoked with the op
    /// and a boolean signifying if the symbols within that symbol table can be
    /// treated as if all uses within the IR are visible to the caller.
    /// `allSymUsesVisible` identifies whether all of the symbol uses of symbols
    /// within `op` are visible.
    pub fn walkSymbolTables(from: Operation, all_sym_uses_visible: bool, callback: WalkSymbolTableCallback, user_data: *anyopaque) void {
        // TODO: *anyopaque doesn't seem to cast to ?*anyopaque for some reason
        c.mlirSymbolTableWalkSymbolTables(from.getRaw(), all_sym_uses_visible, @ptrCast(callback), user_data);
    }
};

test {
    std.testing.refAllDeclsRecursive(@This());
}

test {
    const registry = DialectRegistry.init();
    defer registry.deinit();

    const context = Context.initWithRegistry(registry, false);
    defer context.deinit();

    const location = Location.unknownGet(context);

    const module = Module.initEmpty(location);
    defer module.deinit();

    const index_type = Type.parseGet(context, StringRef.init("index"));
    _ = index_type; // autofix
}
