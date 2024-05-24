#![allow(unused)]

#[doc = "A [`clock_div`](ClockDividerOperation) operation. Produces a clock divided by a power of two."]
#[doc = "\n\n"]
#[doc = "The output clock is phase-aligned to the input clock.\n\n``` text\n%div_clock = seq.clock_div %clock by 1\n```\n"]
pub struct ClockDividerOperation<'c> {
    operation: ::melior::ir::operation::Operation<'c>,
}
impl<'c> ClockDividerOperation<'c> {
    #[doc = r" Returns a name."]
    pub fn name() -> &'static str {
        "seq.clock_div"
    }
    #[doc = r" Returns a generic operation."]
    pub fn as_operation(&self) -> &::melior::ir::operation::Operation<'c> {
        &self.operation
    }
    #[doc = r" Creates a builder."]
    pub fn builder(
        context: &'c ::melior::Context,
        location: ::melior::ir::Location<'c>,
    ) -> ClockDividerOperationBuilder<
        'c,
        ::melior::dialect::ods::__private::Unset,
        ::melior::dialect::ods::__private::Unset,
        ::melior::dialect::ods::__private::Unset,
    > {
        ClockDividerOperationBuilder::new(context, location)
    }
    pub fn output(
        &self,
    ) -> Result<::melior::ir::operation::OperationResult<'c, '_>, ::melior::Error> {
        self.operation.result(0usize)
    }
    pub fn input(&self) -> Result<::melior::ir::Value<'c, '_>, ::melior::Error> {
        self.operation.operand(0usize)
    }
    #[allow(clippy::needless_question_mark)]
    pub fn pow_2(&self) -> Result<::melior::ir::attribute::IntegerAttribute<'c>, ::melior::Error> {
        Ok(self.operation.attribute("pow2")?.try_into()?)
    }
    pub fn set_pow_2(&mut self, value: ::melior::ir::attribute::IntegerAttribute<'c>) {
        self.operation.set_attribute("pow2", value.into());
    }
}
#[doc = "A builder for a [`clock_div`](ClockDividerOperation) operation."]
pub struct ClockDividerOperationBuilder<'c, T0, O0, A0> {
    builder: ::melior::ir::operation::OperationBuilder<'c>,
    context: &'c ::melior::Context,
    _state: ::std::marker::PhantomData<(T0, O0, A0)>,
}
impl<'c>
    ClockDividerOperationBuilder<
        'c,
        ::melior::dialect::ods::__private::Unset,
        ::melior::dialect::ods::__private::Unset,
        ::melior::dialect::ods::__private::Unset,
    >
{
    pub fn new(context: &'c ::melior::Context, location: ::melior::ir::Location<'c>) -> Self {
        Self {
            context,
            builder: ::melior::ir::operation::OperationBuilder::new("seq.clock_div", location),
            _state: Default::default(),
        }
    }
}
impl<'c, O0, A0>
    ClockDividerOperationBuilder<'c, ::melior::dialect::ods::__private::Unset, O0, A0>
{
    pub fn output(
        self,
        output: ::melior::ir::Type<'c>,
    ) -> ClockDividerOperationBuilder<'c, ::melior::dialect::ods::__private::Set, O0, A0> {
        ClockDividerOperationBuilder {
            context: self.context,
            builder: self.builder.add_results(&[output]),
            _state: Default::default(),
        }
    }
}
impl<'c, T0, A0>
    ClockDividerOperationBuilder<'c, T0, ::melior::dialect::ods::__private::Unset, A0>
{
    pub fn input(
        self,
        input: ::melior::ir::Value<'c, '_>,
    ) -> ClockDividerOperationBuilder<'c, T0, ::melior::dialect::ods::__private::Set, A0> {
        ClockDividerOperationBuilder {
            context: self.context,
            builder: self.builder.add_operands(&[input]),
            _state: Default::default(),
        }
    }
}
impl<'c, T0, O0>
    ClockDividerOperationBuilder<'c, T0, O0, ::melior::dialect::ods::__private::Unset>
{
    pub fn pow_2(
        self,
        pow_2: ::melior::ir::attribute::IntegerAttribute<'c>,
    ) -> ClockDividerOperationBuilder<'c, T0, O0, ::melior::dialect::ods::__private::Set> {
        ClockDividerOperationBuilder {
            context: self.context,
            builder: self.builder.add_attributes(&[(
                ::melior::ir::Identifier::new(self.context, "pow2"),
                pow_2.into(),
            )]),
            _state: Default::default(),
        }
    }
}
impl<'c>
    ClockDividerOperationBuilder<
        'c,
        ::melior::dialect::ods::__private::Set,
        ::melior::dialect::ods::__private::Set,
        ::melior::dialect::ods::__private::Set,
    >
{
    pub fn build(self) -> ClockDividerOperation<'c> {
        self.builder
            .build()
            .expect("valid operation")
            .try_into()
            .expect("should be a valid ClockDividerOperation")
    }
}
#[allow(clippy::too_many_arguments)]
#[doc = "Creates a [`clock_div`](ClockDividerOperation) operation."]
pub fn clock_div<'c>(
    context: &'c ::melior::Context,
    output: ::melior::ir::Type<'c>,
    input: ::melior::ir::Value<'c, '_>,
    pow_2: ::melior::ir::attribute::IntegerAttribute<'c>,
    location: ::melior::ir::Location<'c>,
) -> ClockDividerOperation<'c> {
    ClockDividerOperation::builder(context, location)
        .output(output)
        .input(input)
        .pow_2(pow_2)
        .build()
}
impl<'c> TryFrom<::melior::ir::operation::Operation<'c>> for ClockDividerOperation<'c> {
    type Error = ::melior::Error;
    fn try_from(operation: ::melior::ir::operation::Operation<'c>) -> Result<Self, Self::Error> {
        Ok(Self { operation })
    }
}
impl<'c> From<ClockDividerOperation<'c>> for ::melior::ir::operation::Operation<'c> {
    fn from(operation: ClockDividerOperation<'c>) -> Self {
        operation.operation
    }
}
#[doc = "A [`clock_gate`](ClockGateOperation) operation. Safely gates a clock with an enable signal."]
#[doc = "\n\n"]
#[doc = "The `seq.clock_gate` enables and disables a clock safely, without glitches,\nbased on a boolean enable value. If the enable operand is 1, the output\nclock produced by the clock gate is identical to the input clock. If the\nenable operand is 0, the output clock is a constant zero.\n\nThe `enable` operand is sampled at the rising edge of the input clock; any\nchanges on the enable before or after that edge are ignored and do not\naffect the output clock.\n\nThe `test_enable` operand is optional and if present is OR'd together with\nthe `enable` operand to determine whether the output clock is gated or not.\n\nThe op can be referred to using an inner symbol. Upon translation, the\nsymbol will target the instance to the external module it lowers to.\n\n``` text\n%gatedClock = seq.clock_gate %clock, %enable\n%gatedClock = seq.clock_gate %clock, %enable, %test_enable\n```\n"]
pub struct ClockGateOperation<'c> {
    operation: ::melior::ir::operation::Operation<'c>,
}
impl<'c> ClockGateOperation<'c> {
    #[doc = r" Returns a name."]
    pub fn name() -> &'static str {
        "seq.clock_gate"
    }
    #[doc = r" Returns a generic operation."]
    pub fn as_operation(&self) -> &::melior::ir::operation::Operation<'c> {
        &self.operation
    }
    #[doc = r" Creates a builder."]
    pub fn builder(
        context: &'c ::melior::Context,
        location: ::melior::ir::Location<'c>,
    ) -> ClockGateOperationBuilder<
        'c,
        ::melior::dialect::ods::__private::Unset,
        ::melior::dialect::ods::__private::Unset,
        ::melior::dialect::ods::__private::Unset,
    > {
        ClockGateOperationBuilder::new(context, location)
    }
    pub fn output(
        &self,
    ) -> Result<::melior::ir::operation::OperationResult<'c, '_>, ::melior::Error> {
        self.operation.result(0usize)
    }
    pub fn input(&self) -> Result<::melior::ir::Value<'c, '_>, ::melior::Error> {
        self.operation.operand(0usize)
    }
    pub fn enable(&self) -> Result<::melior::ir::Value<'c, '_>, ::melior::Error> {
        self.operation.operand(1usize)
    }
    pub fn test_enable(&self) -> Result<::melior::ir::Value<'c, '_>, ::melior::Error> {
        if self.operation.operand_count() < 3usize {
            Err(::melior::Error::OperandNotFound("test_enable"))
        } else {
            self.operation.operand(2usize)
        }
    }
    #[allow(clippy::needless_question_mark)]
    pub fn inner_sym(&self) -> Result<::melior::ir::attribute::Attribute<'c>, ::melior::Error> {
        Ok(self.operation.attribute("inner_sym")?.try_into()?)
    }
    pub fn set_inner_sym(&mut self, value: ::melior::ir::attribute::Attribute<'c>) {
        self.operation.set_attribute("inner_sym", value.into());
    }
    pub fn remove_inner_sym(&mut self) -> Result<(), ::melior::Error> {
        self.operation.remove_attribute("inner_sym")
    }
}
#[doc = "A builder for a [`clock_gate`](ClockGateOperation) operation."]
pub struct ClockGateOperationBuilder<'c, T0, O0, O1> {
    builder: ::melior::ir::operation::OperationBuilder<'c>,
    context: &'c ::melior::Context,
    _state: ::std::marker::PhantomData<(T0, O0, O1)>,
}
impl<'c>
    ClockGateOperationBuilder<
        'c,
        ::melior::dialect::ods::__private::Unset,
        ::melior::dialect::ods::__private::Unset,
        ::melior::dialect::ods::__private::Unset,
    >
{
    pub fn new(context: &'c ::melior::Context, location: ::melior::ir::Location<'c>) -> Self {
        Self {
            context,
            builder: ::melior::ir::operation::OperationBuilder::new("seq.clock_gate", location),
            _state: Default::default(),
        }
    }
}
impl<'c, O0, O1> ClockGateOperationBuilder<'c, ::melior::dialect::ods::__private::Unset, O0, O1> {
    pub fn output(
        self,
        output: ::melior::ir::Type<'c>,
    ) -> ClockGateOperationBuilder<'c, ::melior::dialect::ods::__private::Set, O0, O1> {
        ClockGateOperationBuilder {
            context: self.context,
            builder: self.builder.add_results(&[output]),
            _state: Default::default(),
        }
    }
}
impl<'c, T0, O1> ClockGateOperationBuilder<'c, T0, ::melior::dialect::ods::__private::Unset, O1> {
    pub fn input(
        self,
        input: ::melior::ir::Value<'c, '_>,
    ) -> ClockGateOperationBuilder<'c, T0, ::melior::dialect::ods::__private::Set, O1> {
        ClockGateOperationBuilder {
            context: self.context,
            builder: self.builder.add_operands(&[input]),
            _state: Default::default(),
        }
    }
}
impl<'c, T0>
    ClockGateOperationBuilder<
        'c,
        T0,
        ::melior::dialect::ods::__private::Set,
        ::melior::dialect::ods::__private::Unset,
    >
{
    pub fn enable(
        self,
        enable: ::melior::ir::Value<'c, '_>,
    ) -> ClockGateOperationBuilder<
        'c,
        T0,
        ::melior::dialect::ods::__private::Set,
        ::melior::dialect::ods::__private::Set,
    > {
        ClockGateOperationBuilder {
            context: self.context,
            builder: self.builder.add_operands(&[enable]),
            _state: Default::default(),
        }
    }
}
impl<'c, T0, O0, O1> ClockGateOperationBuilder<'c, T0, O0, O1> {
    pub fn test_enable(
        mut self,
        test_enable: ::melior::ir::Value<'c, '_>,
    ) -> ClockGateOperationBuilder<'c, T0, O0, O1> {
        self.builder = self.builder.add_operands(&[test_enable]);
        self
    }
}
impl<'c, T0, O0, O1> ClockGateOperationBuilder<'c, T0, O0, O1> {
    pub fn inner_sym(
        mut self,
        inner_sym: ::melior::ir::attribute::Attribute<'c>,
    ) -> ClockGateOperationBuilder<'c, T0, O0, O1> {
        self.builder = self.builder.add_attributes(&[(
            ::melior::ir::Identifier::new(self.context, "inner_sym"),
            inner_sym.into(),
        )]);
        self
    }
}
impl<'c>
    ClockGateOperationBuilder<
        'c,
        ::melior::dialect::ods::__private::Set,
        ::melior::dialect::ods::__private::Set,
        ::melior::dialect::ods::__private::Set,
    >
{
    pub fn build(self) -> ClockGateOperation<'c> {
        self.builder
            .build()
            .expect("valid operation")
            .try_into()
            .expect("should be a valid ClockGateOperation")
    }
}
#[allow(clippy::too_many_arguments)]
#[doc = "Creates a [`clock_gate`](ClockGateOperation) operation."]
pub fn clock_gate<'c>(
    context: &'c ::melior::Context,
    output: ::melior::ir::Type<'c>,
    input: ::melior::ir::Value<'c, '_>,
    enable: ::melior::ir::Value<'c, '_>,
    location: ::melior::ir::Location<'c>,
) -> ClockGateOperation<'c> {
    ClockGateOperation::builder(context, location)
        .output(output)
        .input(input)
        .enable(enable)
        .build()
}
impl<'c> TryFrom<::melior::ir::operation::Operation<'c>> for ClockGateOperation<'c> {
    type Error = ::melior::Error;
    fn try_from(operation: ::melior::ir::operation::Operation<'c>) -> Result<Self, Self::Error> {
        Ok(Self { operation })
    }
}
impl<'c> From<ClockGateOperation<'c>> for ::melior::ir::operation::Operation<'c> {
    fn from(operation: ClockGateOperation<'c>) -> Self {
        operation.operation
    }
}
#[doc = "A [`clock_inv`](ClockInverterOperation) operation. Inverts the clock signal."]
#[doc = "\n\n"]
#[doc = "Note that the compiler can optimize inverters away, preventing their\nuse as part of explicit clock buffers.\n\n``` text\n%inv_clock = seq.clock_inv %clock\n```\n"]
pub struct ClockInverterOperation<'c> {
    operation: ::melior::ir::operation::Operation<'c>,
}
impl<'c> ClockInverterOperation<'c> {
    #[doc = r" Returns a name."]
    pub fn name() -> &'static str {
        "seq.clock_inv"
    }
    #[doc = r" Returns a generic operation."]
    pub fn as_operation(&self) -> &::melior::ir::operation::Operation<'c> {
        &self.operation
    }
    #[doc = r" Creates a builder."]
    pub fn builder(
        context: &'c ::melior::Context,
        location: ::melior::ir::Location<'c>,
    ) -> ClockInverterOperationBuilder<
        'c,
        ::melior::dialect::ods::__private::Unset,
        ::melior::dialect::ods::__private::Unset,
    > {
        ClockInverterOperationBuilder::new(context, location)
    }
    pub fn output(
        &self,
    ) -> Result<::melior::ir::operation::OperationResult<'c, '_>, ::melior::Error> {
        self.operation.result(0usize)
    }
    pub fn input(&self) -> Result<::melior::ir::Value<'c, '_>, ::melior::Error> {
        self.operation.operand(0usize)
    }
}
#[doc = "A builder for a [`clock_inv`](ClockInverterOperation) operation."]
pub struct ClockInverterOperationBuilder<'c, T0, O0> {
    builder: ::melior::ir::operation::OperationBuilder<'c>,
    context: &'c ::melior::Context,
    _state: ::std::marker::PhantomData<(T0, O0)>,
}
impl<'c>
    ClockInverterOperationBuilder<
        'c,
        ::melior::dialect::ods::__private::Unset,
        ::melior::dialect::ods::__private::Unset,
    >
{
    pub fn new(context: &'c ::melior::Context, location: ::melior::ir::Location<'c>) -> Self {
        Self {
            context,
            builder: ::melior::ir::operation::OperationBuilder::new("seq.clock_inv", location),
            _state: Default::default(),
        }
    }
}
impl<'c, O0> ClockInverterOperationBuilder<'c, ::melior::dialect::ods::__private::Unset, O0> {
    pub fn output(
        self,
        output: ::melior::ir::Type<'c>,
    ) -> ClockInverterOperationBuilder<'c, ::melior::dialect::ods::__private::Set, O0> {
        ClockInverterOperationBuilder {
            context: self.context,
            builder: self.builder.add_results(&[output]),
            _state: Default::default(),
        }
    }
}
impl<'c, T0> ClockInverterOperationBuilder<'c, T0, ::melior::dialect::ods::__private::Unset> {
    pub fn input(
        self,
        input: ::melior::ir::Value<'c, '_>,
    ) -> ClockInverterOperationBuilder<'c, T0, ::melior::dialect::ods::__private::Set> {
        ClockInverterOperationBuilder {
            context: self.context,
            builder: self.builder.add_operands(&[input]),
            _state: Default::default(),
        }
    }
}
impl<'c>
    ClockInverterOperationBuilder<
        'c,
        ::melior::dialect::ods::__private::Set,
        ::melior::dialect::ods::__private::Set,
    >
{
    pub fn build(self) -> ClockInverterOperation<'c> {
        self.builder
            .build()
            .expect("valid operation")
            .try_into()
            .expect("should be a valid ClockInverterOperation")
    }
}
#[allow(clippy::too_many_arguments)]
#[doc = "Creates a [`clock_inv`](ClockInverterOperation) operation."]
pub fn clock_inv<'c>(
    context: &'c ::melior::Context,
    output: ::melior::ir::Type<'c>,
    input: ::melior::ir::Value<'c, '_>,
    location: ::melior::ir::Location<'c>,
) -> ClockInverterOperation<'c> {
    ClockInverterOperation::builder(context, location)
        .output(output)
        .input(input)
        .build()
}
impl<'c> TryFrom<::melior::ir::operation::Operation<'c>> for ClockInverterOperation<'c> {
    type Error = ::melior::Error;
    fn try_from(operation: ::melior::ir::operation::Operation<'c>) -> Result<Self, Self::Error> {
        Ok(Self { operation })
    }
}
impl<'c> From<ClockInverterOperation<'c>> for ::melior::ir::operation::Operation<'c> {
    fn from(operation: ClockInverterOperation<'c>) -> Self {
        operation.operation
    }
}
#[doc = "A [`clock_mux`](ClockMuxOperation) operation. Safely selects a clock based on a condition."]
#[doc = "\n\n"]
#[doc = "The `seq.clock_mux` op selects a clock from two options. If `cond` is\ntrue, the first clock operand is selected to drive downstream logic.\nOtherwise, the second clock is used.\n\n``` text\n%clock = seq.clock_mux %cond, %trueClock, %falseClock\n```\n"]
pub struct ClockMuxOperation<'c> {
    operation: ::melior::ir::operation::Operation<'c>,
}
impl<'c> ClockMuxOperation<'c> {
    #[doc = r" Returns a name."]
    pub fn name() -> &'static str {
        "seq.clock_mux"
    }
    #[doc = r" Returns a generic operation."]
    pub fn as_operation(&self) -> &::melior::ir::operation::Operation<'c> {
        &self.operation
    }
    #[doc = r" Creates a builder."]
    pub fn builder(
        context: &'c ::melior::Context,
        location: ::melior::ir::Location<'c>,
    ) -> ClockMuxOperationBuilder<
        'c,
        ::melior::dialect::ods::__private::Unset,
        ::melior::dialect::ods::__private::Unset,
        ::melior::dialect::ods::__private::Unset,
        ::melior::dialect::ods::__private::Unset,
    > {
        ClockMuxOperationBuilder::new(context, location)
    }
    pub fn result(
        &self,
    ) -> Result<::melior::ir::operation::OperationResult<'c, '_>, ::melior::Error> {
        self.operation.result(0usize)
    }
    pub fn cond(&self) -> Result<::melior::ir::Value<'c, '_>, ::melior::Error> {
        self.operation.operand(0usize)
    }
    pub fn true_clock(&self) -> Result<::melior::ir::Value<'c, '_>, ::melior::Error> {
        self.operation.operand(1usize)
    }
    pub fn false_clock(&self) -> Result<::melior::ir::Value<'c, '_>, ::melior::Error> {
        self.operation.operand(2usize)
    }
}
#[doc = "A builder for a [`clock_mux`](ClockMuxOperation) operation."]
pub struct ClockMuxOperationBuilder<'c, T0, O0, O1, O2> {
    builder: ::melior::ir::operation::OperationBuilder<'c>,
    context: &'c ::melior::Context,
    _state: ::std::marker::PhantomData<(T0, O0, O1, O2)>,
}
impl<'c>
    ClockMuxOperationBuilder<
        'c,
        ::melior::dialect::ods::__private::Unset,
        ::melior::dialect::ods::__private::Unset,
        ::melior::dialect::ods::__private::Unset,
        ::melior::dialect::ods::__private::Unset,
    >
{
    pub fn new(context: &'c ::melior::Context, location: ::melior::ir::Location<'c>) -> Self {
        Self {
            context,
            builder: ::melior::ir::operation::OperationBuilder::new("seq.clock_mux", location),
            _state: Default::default(),
        }
    }
}
impl<'c, O0, O1, O2>
    ClockMuxOperationBuilder<'c, ::melior::dialect::ods::__private::Unset, O0, O1, O2>
{
    pub fn result(
        self,
        result: ::melior::ir::Type<'c>,
    ) -> ClockMuxOperationBuilder<'c, ::melior::dialect::ods::__private::Set, O0, O1, O2> {
        ClockMuxOperationBuilder {
            context: self.context,
            builder: self.builder.add_results(&[result]),
            _state: Default::default(),
        }
    }
}
impl<'c, T0, O1, O2>
    ClockMuxOperationBuilder<'c, T0, ::melior::dialect::ods::__private::Unset, O1, O2>
{
    pub fn cond(
        self,
        cond: ::melior::ir::Value<'c, '_>,
    ) -> ClockMuxOperationBuilder<'c, T0, ::melior::dialect::ods::__private::Set, O1, O2> {
        ClockMuxOperationBuilder {
            context: self.context,
            builder: self.builder.add_operands(&[cond]),
            _state: Default::default(),
        }
    }
}
impl<'c, T0, O2>
    ClockMuxOperationBuilder<
        'c,
        T0,
        ::melior::dialect::ods::__private::Set,
        ::melior::dialect::ods::__private::Unset,
        O2,
    >
{
    pub fn true_clock(
        self,
        true_clock: ::melior::ir::Value<'c, '_>,
    ) -> ClockMuxOperationBuilder<
        'c,
        T0,
        ::melior::dialect::ods::__private::Set,
        ::melior::dialect::ods::__private::Set,
        O2,
    > {
        ClockMuxOperationBuilder {
            context: self.context,
            builder: self.builder.add_operands(&[true_clock]),
            _state: Default::default(),
        }
    }
}
impl<'c, T0>
    ClockMuxOperationBuilder<
        'c,
        T0,
        ::melior::dialect::ods::__private::Set,
        ::melior::dialect::ods::__private::Set,
        ::melior::dialect::ods::__private::Unset,
    >
{
    pub fn false_clock(
        self,
        false_clock: ::melior::ir::Value<'c, '_>,
    ) -> ClockMuxOperationBuilder<
        'c,
        T0,
        ::melior::dialect::ods::__private::Set,
        ::melior::dialect::ods::__private::Set,
        ::melior::dialect::ods::__private::Set,
    > {
        ClockMuxOperationBuilder {
            context: self.context,
            builder: self.builder.add_operands(&[false_clock]),
            _state: Default::default(),
        }
    }
}
impl<'c>
    ClockMuxOperationBuilder<
        'c,
        ::melior::dialect::ods::__private::Set,
        ::melior::dialect::ods::__private::Set,
        ::melior::dialect::ods::__private::Set,
        ::melior::dialect::ods::__private::Set,
    >
{
    pub fn build(self) -> ClockMuxOperation<'c> {
        self.builder
            .build()
            .expect("valid operation")
            .try_into()
            .expect("should be a valid ClockMuxOperation")
    }
}
#[allow(clippy::too_many_arguments)]
#[doc = "Creates a [`clock_mux`](ClockMuxOperation) operation."]
pub fn clock_mux<'c>(
    context: &'c ::melior::Context,
    result: ::melior::ir::Type<'c>,
    cond: ::melior::ir::Value<'c, '_>,
    true_clock: ::melior::ir::Value<'c, '_>,
    false_clock: ::melior::ir::Value<'c, '_>,
    location: ::melior::ir::Location<'c>,
) -> ClockMuxOperation<'c> {
    ClockMuxOperation::builder(context, location)
        .result(result)
        .cond(cond)
        .true_clock(true_clock)
        .false_clock(false_clock)
        .build()
}
impl<'c> TryFrom<::melior::ir::operation::Operation<'c>> for ClockMuxOperation<'c> {
    type Error = ::melior::Error;
    fn try_from(operation: ::melior::ir::operation::Operation<'c>) -> Result<Self, Self::Error> {
        Ok(Self { operation })
    }
}
impl<'c> From<ClockMuxOperation<'c>> for ::melior::ir::operation::Operation<'c> {
    fn from(operation: ClockMuxOperation<'c>) -> Self {
        operation.operation
    }
}
#[doc = "A [`compreg.ce`](CompRegClockEnabledOperation) operation. When enabled, register a value."]
#[doc = "\n\n"]
#[doc = "See the Seq dialect rationale for a longer description\n"]
pub struct CompRegClockEnabledOperation<'c> {
    operation: ::melior::ir::operation::Operation<'c>,
}
impl<'c> CompRegClockEnabledOperation<'c> {
    #[doc = r" Returns a name."]
    pub fn name() -> &'static str {
        "seq.compreg.ce"
    }
    #[doc = r" Returns a generic operation."]
    pub fn as_operation(&self) -> &::melior::ir::operation::Operation<'c> {
        &self.operation
    }
    #[doc = r" Creates a builder."]
    pub fn builder(
        context: &'c ::melior::Context,
        location: ::melior::ir::Location<'c>,
    ) -> CompRegClockEnabledOperationBuilder<
        'c,
        ::melior::dialect::ods::__private::Unset,
        ::melior::dialect::ods::__private::Unset,
        ::melior::dialect::ods::__private::Unset,
        ::melior::dialect::ods::__private::Unset,
    > {
        CompRegClockEnabledOperationBuilder::new(context, location)
    }
    pub fn data(
        &self,
    ) -> Result<::melior::ir::operation::OperationResult<'c, '_>, ::melior::Error> {
        self.operation.result(0usize)
    }
    pub fn input(&self) -> Result<::melior::ir::Value<'c, '_>, ::melior::Error> {
        let attribute = ::melior::ir::attribute::DenseI32ArrayAttribute::<'c>::try_from(
            self.operation.attribute("operand_segment_sizes")?,
        )?;
        let start = (0..0usize)
            .map(|index| attribute.element(index))
            .collect::<Result<Vec<_>, _>>()?
            .into_iter()
            .sum::<i32>() as usize;
        let group_len = attribute.element(0usize)? as usize;
        self.operation.operand(start)
    }
    pub fn clk(&self) -> Result<::melior::ir::Value<'c, '_>, ::melior::Error> {
        let attribute = ::melior::ir::attribute::DenseI32ArrayAttribute::<'c>::try_from(
            self.operation.attribute("operand_segment_sizes")?,
        )?;
        let start = (0..1usize)
            .map(|index| attribute.element(index))
            .collect::<Result<Vec<_>, _>>()?
            .into_iter()
            .sum::<i32>() as usize;
        let group_len = attribute.element(1usize)? as usize;
        self.operation.operand(start)
    }
    pub fn clock_enable(&self) -> Result<::melior::ir::Value<'c, '_>, ::melior::Error> {
        let attribute = ::melior::ir::attribute::DenseI32ArrayAttribute::<'c>::try_from(
            self.operation.attribute("operand_segment_sizes")?,
        )?;
        let start = (0..2usize)
            .map(|index| attribute.element(index))
            .collect::<Result<Vec<_>, _>>()?
            .into_iter()
            .sum::<i32>() as usize;
        let group_len = attribute.element(2usize)? as usize;
        self.operation.operand(start)
    }
    pub fn reset(&self) -> Result<::melior::ir::Value<'c, '_>, ::melior::Error> {
        let attribute = ::melior::ir::attribute::DenseI32ArrayAttribute::<'c>::try_from(
            self.operation.attribute("operand_segment_sizes")?,
        )?;
        let start = (0..3usize)
            .map(|index| attribute.element(index))
            .collect::<Result<Vec<_>, _>>()?
            .into_iter()
            .sum::<i32>() as usize;
        let group_len = attribute.element(3usize)? as usize;
        if group_len == 0 {
            Err(::melior::Error::OperandNotFound("reset"))
        } else {
            self.operation.operand(start)
        }
    }
    pub fn reset_value(&self) -> Result<::melior::ir::Value<'c, '_>, ::melior::Error> {
        let attribute = ::melior::ir::attribute::DenseI32ArrayAttribute::<'c>::try_from(
            self.operation.attribute("operand_segment_sizes")?,
        )?;
        let start = (0..4usize)
            .map(|index| attribute.element(index))
            .collect::<Result<Vec<_>, _>>()?
            .into_iter()
            .sum::<i32>() as usize;
        let group_len = attribute.element(4usize)? as usize;
        if group_len == 0 {
            Err(::melior::Error::OperandNotFound("resetValue"))
        } else {
            self.operation.operand(start)
        }
    }
    pub fn power_on_value(&self) -> Result<::melior::ir::Value<'c, '_>, ::melior::Error> {
        let attribute = ::melior::ir::attribute::DenseI32ArrayAttribute::<'c>::try_from(
            self.operation.attribute("operand_segment_sizes")?,
        )?;
        let start = (0..5usize)
            .map(|index| attribute.element(index))
            .collect::<Result<Vec<_>, _>>()?
            .into_iter()
            .sum::<i32>() as usize;
        let group_len = attribute.element(5usize)? as usize;
        if group_len == 0 {
            Err(::melior::Error::OperandNotFound("powerOnValue"))
        } else {
            self.operation.operand(start)
        }
    }
    #[allow(clippy::needless_question_mark)]
    pub fn _name(&self) -> Result<::melior::ir::attribute::StringAttribute<'c>, ::melior::Error> {
        Ok(self.operation.attribute("name")?.try_into()?)
    }
    pub fn set_name(&mut self, value: ::melior::ir::attribute::StringAttribute<'c>) {
        self.operation.set_attribute("name", value.into());
    }
    pub fn remove_name(&mut self) -> Result<(), ::melior::Error> {
        self.operation.remove_attribute("name")
    }
    #[allow(clippy::needless_question_mark)]
    pub fn inner_sym(&self) -> Result<::melior::ir::attribute::Attribute<'c>, ::melior::Error> {
        Ok(self.operation.attribute("inner_sym")?.try_into()?)
    }
    pub fn set_inner_sym(&mut self, value: ::melior::ir::attribute::Attribute<'c>) {
        self.operation.set_attribute("inner_sym", value.into());
    }
    pub fn remove_inner_sym(&mut self) -> Result<(), ::melior::Error> {
        self.operation.remove_attribute("inner_sym")
    }
}
#[doc = "A builder for a [`compreg.ce`](CompRegClockEnabledOperation) operation."]
pub struct CompRegClockEnabledOperationBuilder<'c, T0, O0, O1, O2> {
    builder: ::melior::ir::operation::OperationBuilder<'c>,
    context: &'c ::melior::Context,
    _state: ::std::marker::PhantomData<(T0, O0, O1, O2)>,
}
impl<'c>
    CompRegClockEnabledOperationBuilder<
        'c,
        ::melior::dialect::ods::__private::Unset,
        ::melior::dialect::ods::__private::Unset,
        ::melior::dialect::ods::__private::Unset,
        ::melior::dialect::ods::__private::Unset,
    >
{
    pub fn new(context: &'c ::melior::Context, location: ::melior::ir::Location<'c>) -> Self {
        Self {
            context,
            builder: ::melior::ir::operation::OperationBuilder::new("seq.compreg.ce", location),
            _state: Default::default(),
        }
    }
}
impl<'c, O0, O1, O2>
    CompRegClockEnabledOperationBuilder<'c, ::melior::dialect::ods::__private::Unset, O0, O1, O2>
{
    pub fn data(
        self,
        data: ::melior::ir::Type<'c>,
    ) -> CompRegClockEnabledOperationBuilder<'c, ::melior::dialect::ods::__private::Set, O0, O1, O2>
    {
        CompRegClockEnabledOperationBuilder {
            context: self.context,
            builder: self.builder.add_results(&[data]),
            _state: Default::default(),
        }
    }
}
impl<'c, T0, O1, O2>
    CompRegClockEnabledOperationBuilder<'c, T0, ::melior::dialect::ods::__private::Unset, O1, O2>
{
    pub fn input(
        self,
        input: ::melior::ir::Value<'c, '_>,
    ) -> CompRegClockEnabledOperationBuilder<'c, T0, ::melior::dialect::ods::__private::Set, O1, O2>
    {
        CompRegClockEnabledOperationBuilder {
            context: self.context,
            builder: self.builder.add_operands(&[input]),
            _state: Default::default(),
        }
    }
}
impl<'c, T0, O2>
    CompRegClockEnabledOperationBuilder<
        'c,
        T0,
        ::melior::dialect::ods::__private::Set,
        ::melior::dialect::ods::__private::Unset,
        O2,
    >
{
    pub fn clk(
        self,
        clk: ::melior::ir::Value<'c, '_>,
    ) -> CompRegClockEnabledOperationBuilder<
        'c,
        T0,
        ::melior::dialect::ods::__private::Set,
        ::melior::dialect::ods::__private::Set,
        O2,
    > {
        CompRegClockEnabledOperationBuilder {
            context: self.context,
            builder: self.builder.add_operands(&[clk]),
            _state: Default::default(),
        }
    }
}
impl<'c, T0>
    CompRegClockEnabledOperationBuilder<
        'c,
        T0,
        ::melior::dialect::ods::__private::Set,
        ::melior::dialect::ods::__private::Set,
        ::melior::dialect::ods::__private::Unset,
    >
{
    pub fn clock_enable(
        self,
        clock_enable: ::melior::ir::Value<'c, '_>,
    ) -> CompRegClockEnabledOperationBuilder<
        'c,
        T0,
        ::melior::dialect::ods::__private::Set,
        ::melior::dialect::ods::__private::Set,
        ::melior::dialect::ods::__private::Set,
    > {
        CompRegClockEnabledOperationBuilder {
            context: self.context,
            builder: self.builder.add_operands(&[clock_enable]),
            _state: Default::default(),
        }
    }
}
impl<'c, T0, O0, O1, O2> CompRegClockEnabledOperationBuilder<'c, T0, O0, O1, O2> {
    pub fn reset(
        mut self,
        reset: ::melior::ir::Value<'c, '_>,
    ) -> CompRegClockEnabledOperationBuilder<'c, T0, O0, O1, O2> {
        self.builder = self.builder.add_operands(&[reset]);
        self
    }
}
impl<'c, T0, O0, O1, O2> CompRegClockEnabledOperationBuilder<'c, T0, O0, O1, O2> {
    pub fn reset_value(
        mut self,
        reset_value: ::melior::ir::Value<'c, '_>,
    ) -> CompRegClockEnabledOperationBuilder<'c, T0, O0, O1, O2> {
        self.builder = self.builder.add_operands(&[reset_value]);
        self
    }
}
impl<'c, T0, O0, O1, O2> CompRegClockEnabledOperationBuilder<'c, T0, O0, O1, O2> {
    pub fn power_on_value(
        mut self,
        power_on_value: ::melior::ir::Value<'c, '_>,
    ) -> CompRegClockEnabledOperationBuilder<'c, T0, O0, O1, O2> {
        self.builder = self.builder.add_operands(&[power_on_value]);
        self
    }
}
impl<'c, T0, O0, O1, O2> CompRegClockEnabledOperationBuilder<'c, T0, O0, O1, O2> {
    pub fn _name(
        mut self,
        _name: ::melior::ir::attribute::StringAttribute<'c>,
    ) -> CompRegClockEnabledOperationBuilder<'c, T0, O0, O1, O2> {
        self.builder = self.builder.add_attributes(&[(
            ::melior::ir::Identifier::new(self.context, "name"),
            _name.into(),
        )]);
        self
    }
}
impl<'c, T0, O0, O1, O2> CompRegClockEnabledOperationBuilder<'c, T0, O0, O1, O2> {
    pub fn inner_sym(
        mut self,
        inner_sym: ::melior::ir::attribute::Attribute<'c>,
    ) -> CompRegClockEnabledOperationBuilder<'c, T0, O0, O1, O2> {
        self.builder = self.builder.add_attributes(&[(
            ::melior::ir::Identifier::new(self.context, "inner_sym"),
            inner_sym.into(),
        )]);
        self
    }
}
impl<'c>
    CompRegClockEnabledOperationBuilder<
        'c,
        ::melior::dialect::ods::__private::Set,
        ::melior::dialect::ods::__private::Set,
        ::melior::dialect::ods::__private::Set,
        ::melior::dialect::ods::__private::Set,
    >
{
    pub fn build(self) -> CompRegClockEnabledOperation<'c> {
        self.builder
            .build()
            .expect("valid operation")
            .try_into()
            .expect("should be a valid CompRegClockEnabledOperation")
    }
}
#[allow(clippy::too_many_arguments)]
#[doc = "Creates a [`compreg.ce`](CompRegClockEnabledOperation) operation."]
pub fn compreg_ce<'c>(
    context: &'c ::melior::Context,
    data: ::melior::ir::Type<'c>,
    input: ::melior::ir::Value<'c, '_>,
    clk: ::melior::ir::Value<'c, '_>,
    clock_enable: ::melior::ir::Value<'c, '_>,
    location: ::melior::ir::Location<'c>,
) -> CompRegClockEnabledOperation<'c> {
    CompRegClockEnabledOperation::builder(context, location)
        .data(data)
        .input(input)
        .clk(clk)
        .clock_enable(clock_enable)
        .build()
}
impl<'c> TryFrom<::melior::ir::operation::Operation<'c>> for CompRegClockEnabledOperation<'c> {
    type Error = ::melior::Error;
    fn try_from(operation: ::melior::ir::operation::Operation<'c>) -> Result<Self, Self::Error> {
        Ok(Self { operation })
    }
}
impl<'c> From<CompRegClockEnabledOperation<'c>> for ::melior::ir::operation::Operation<'c> {
    fn from(operation: CompRegClockEnabledOperation<'c>) -> Self {
        operation.operation
    }
}
#[doc = "A [`compreg`](CompRegOperation) operation. Register a value, storing it for one cycle."]
#[doc = "\n\n"]
#[doc = "See the Seq dialect rationale for a longer description\n"]
pub struct CompRegOperation<'c> {
    operation: ::melior::ir::operation::Operation<'c>,
}
impl<'c> CompRegOperation<'c> {
    #[doc = r" Returns a name."]
    pub fn name() -> &'static str {
        "seq.compreg"
    }
    #[doc = r" Returns a generic operation."]
    pub fn as_operation(&self) -> &::melior::ir::operation::Operation<'c> {
        &self.operation
    }
    #[doc = r" Creates a builder."]
    pub fn builder(
        context: &'c ::melior::Context,
        location: ::melior::ir::Location<'c>,
    ) -> CompRegOperationBuilder<
        'c,
        ::melior::dialect::ods::__private::Unset,
        ::melior::dialect::ods::__private::Unset,
        ::melior::dialect::ods::__private::Unset,
    > {
        CompRegOperationBuilder::new(context, location)
    }
    pub fn data(
        &self,
    ) -> Result<::melior::ir::operation::OperationResult<'c, '_>, ::melior::Error> {
        self.operation.result(0usize)
    }
    pub fn input(&self) -> Result<::melior::ir::Value<'c, '_>, ::melior::Error> {
        let attribute = ::melior::ir::attribute::DenseI32ArrayAttribute::<'c>::try_from(
            self.operation.attribute("operand_segment_sizes")?,
        )?;
        let start = (0..0usize)
            .map(|index| attribute.element(index))
            .collect::<Result<Vec<_>, _>>()?
            .into_iter()
            .sum::<i32>() as usize;
        let group_len = attribute.element(0usize)? as usize;
        self.operation.operand(start)
    }
    pub fn clk(&self) -> Result<::melior::ir::Value<'c, '_>, ::melior::Error> {
        let attribute = ::melior::ir::attribute::DenseI32ArrayAttribute::<'c>::try_from(
            self.operation.attribute("operand_segment_sizes")?,
        )?;
        let start = (0..1usize)
            .map(|index| attribute.element(index))
            .collect::<Result<Vec<_>, _>>()?
            .into_iter()
            .sum::<i32>() as usize;
        let group_len = attribute.element(1usize)? as usize;
        self.operation.operand(start)
    }
    pub fn reset(&self) -> Result<::melior::ir::Value<'c, '_>, ::melior::Error> {
        let attribute = ::melior::ir::attribute::DenseI32ArrayAttribute::<'c>::try_from(
            self.operation.attribute("operand_segment_sizes")?,
        )?;
        let start = (0..2usize)
            .map(|index| attribute.element(index))
            .collect::<Result<Vec<_>, _>>()?
            .into_iter()
            .sum::<i32>() as usize;
        let group_len = attribute.element(2usize)? as usize;
        if group_len == 0 {
            Err(::melior::Error::OperandNotFound("reset"))
        } else {
            self.operation.operand(start)
        }
    }
    pub fn reset_value(&self) -> Result<::melior::ir::Value<'c, '_>, ::melior::Error> {
        let attribute = ::melior::ir::attribute::DenseI32ArrayAttribute::<'c>::try_from(
            self.operation.attribute("operand_segment_sizes")?,
        )?;
        let start = (0..3usize)
            .map(|index| attribute.element(index))
            .collect::<Result<Vec<_>, _>>()?
            .into_iter()
            .sum::<i32>() as usize;
        let group_len = attribute.element(3usize)? as usize;
        if group_len == 0 {
            Err(::melior::Error::OperandNotFound("resetValue"))
        } else {
            self.operation.operand(start)
        }
    }
    pub fn power_on_value(&self) -> Result<::melior::ir::Value<'c, '_>, ::melior::Error> {
        let attribute = ::melior::ir::attribute::DenseI32ArrayAttribute::<'c>::try_from(
            self.operation.attribute("operand_segment_sizes")?,
        )?;
        let start = (0..4usize)
            .map(|index| attribute.element(index))
            .collect::<Result<Vec<_>, _>>()?
            .into_iter()
            .sum::<i32>() as usize;
        let group_len = attribute.element(4usize)? as usize;
        if group_len == 0 {
            Err(::melior::Error::OperandNotFound("powerOnValue"))
        } else {
            self.operation.operand(start)
        }
    }
    #[allow(clippy::needless_question_mark)]
    pub fn _name(&self) -> Result<::melior::ir::attribute::StringAttribute<'c>, ::melior::Error> {
        Ok(self.operation.attribute("name")?.try_into()?)
    }
    pub fn set_name(&mut self, value: ::melior::ir::attribute::StringAttribute<'c>) {
        self.operation.set_attribute("name", value.into());
    }
    pub fn remove_name(&mut self) -> Result<(), ::melior::Error> {
        self.operation.remove_attribute("name")
    }
    #[allow(clippy::needless_question_mark)]
    pub fn inner_sym(&self) -> Result<::melior::ir::attribute::Attribute<'c>, ::melior::Error> {
        Ok(self.operation.attribute("inner_sym")?.try_into()?)
    }
    pub fn set_inner_sym(&mut self, value: ::melior::ir::attribute::Attribute<'c>) {
        self.operation.set_attribute("inner_sym", value.into());
    }
    pub fn remove_inner_sym(&mut self) -> Result<(), ::melior::Error> {
        self.operation.remove_attribute("inner_sym")
    }
}
#[doc = "A builder for a [`compreg`](CompRegOperation) operation."]
pub struct CompRegOperationBuilder<'c, T0, O0, O1> {
    builder: ::melior::ir::operation::OperationBuilder<'c>,
    context: &'c ::melior::Context,
    _state: ::std::marker::PhantomData<(T0, O0, O1)>,
}
impl<'c>
    CompRegOperationBuilder<
        'c,
        ::melior::dialect::ods::__private::Unset,
        ::melior::dialect::ods::__private::Unset,
        ::melior::dialect::ods::__private::Unset,
    >
{
    pub fn new(context: &'c ::melior::Context, location: ::melior::ir::Location<'c>) -> Self {
        Self {
            context,
            builder: ::melior::ir::operation::OperationBuilder::new("seq.compreg", location),
            _state: Default::default(),
        }
    }
}
impl<'c, O0, O1> CompRegOperationBuilder<'c, ::melior::dialect::ods::__private::Unset, O0, O1> {
    pub fn data(
        self,
        data: ::melior::ir::Type<'c>,
    ) -> CompRegOperationBuilder<'c, ::melior::dialect::ods::__private::Set, O0, O1> {
        CompRegOperationBuilder {
            context: self.context,
            builder: self.builder.add_results(&[data]),
            _state: Default::default(),
        }
    }
}
impl<'c, T0, O1> CompRegOperationBuilder<'c, T0, ::melior::dialect::ods::__private::Unset, O1> {
    pub fn input(
        self,
        input: ::melior::ir::Value<'c, '_>,
    ) -> CompRegOperationBuilder<'c, T0, ::melior::dialect::ods::__private::Set, O1> {
        CompRegOperationBuilder {
            context: self.context,
            builder: self.builder.add_operands(&[input]),
            _state: Default::default(),
        }
    }
}
impl<'c, T0>
    CompRegOperationBuilder<
        'c,
        T0,
        ::melior::dialect::ods::__private::Set,
        ::melior::dialect::ods::__private::Unset,
    >
{
    pub fn clk(
        self,
        clk: ::melior::ir::Value<'c, '_>,
    ) -> CompRegOperationBuilder<
        'c,
        T0,
        ::melior::dialect::ods::__private::Set,
        ::melior::dialect::ods::__private::Set,
    > {
        CompRegOperationBuilder {
            context: self.context,
            builder: self.builder.add_operands(&[clk]),
            _state: Default::default(),
        }
    }
}
impl<'c, T0, O0, O1> CompRegOperationBuilder<'c, T0, O0, O1> {
    pub fn reset(
        mut self,
        reset: ::melior::ir::Value<'c, '_>,
    ) -> CompRegOperationBuilder<'c, T0, O0, O1> {
        self.builder = self.builder.add_operands(&[reset]);
        self
    }
}
impl<'c, T0, O0, O1> CompRegOperationBuilder<'c, T0, O0, O1> {
    pub fn reset_value(
        mut self,
        reset_value: ::melior::ir::Value<'c, '_>,
    ) -> CompRegOperationBuilder<'c, T0, O0, O1> {
        self.builder = self.builder.add_operands(&[reset_value]);
        self
    }
}
impl<'c, T0, O0, O1> CompRegOperationBuilder<'c, T0, O0, O1> {
    pub fn power_on_value(
        mut self,
        power_on_value: ::melior::ir::Value<'c, '_>,
    ) -> CompRegOperationBuilder<'c, T0, O0, O1> {
        self.builder = self.builder.add_operands(&[power_on_value]);
        self
    }
}
impl<'c, T0, O0, O1> CompRegOperationBuilder<'c, T0, O0, O1> {
    pub fn _name(
        mut self,
        _name: ::melior::ir::attribute::StringAttribute<'c>,
    ) -> CompRegOperationBuilder<'c, T0, O0, O1> {
        self.builder = self.builder.add_attributes(&[(
            ::melior::ir::Identifier::new(self.context, "name"),
            _name.into(),
        )]);
        self
    }
}
impl<'c, T0, O0, O1> CompRegOperationBuilder<'c, T0, O0, O1> {
    pub fn inner_sym(
        mut self,
        inner_sym: ::melior::ir::attribute::Attribute<'c>,
    ) -> CompRegOperationBuilder<'c, T0, O0, O1> {
        self.builder = self.builder.add_attributes(&[(
            ::melior::ir::Identifier::new(self.context, "inner_sym"),
            inner_sym.into(),
        )]);
        self
    }
}
impl<'c>
    CompRegOperationBuilder<
        'c,
        ::melior::dialect::ods::__private::Set,
        ::melior::dialect::ods::__private::Set,
        ::melior::dialect::ods::__private::Set,
    >
{
    pub fn build(self) -> CompRegOperation<'c> {
        self.builder
            .build()
            .expect("valid operation")
            .try_into()
            .expect("should be a valid CompRegOperation")
    }
}
#[allow(clippy::too_many_arguments)]
#[doc = "Creates a [`compreg`](CompRegOperation) operation."]
pub fn compreg<'c>(
    context: &'c ::melior::Context,
    data: ::melior::ir::Type<'c>,
    input: ::melior::ir::Value<'c, '_>,
    clk: ::melior::ir::Value<'c, '_>,
    location: ::melior::ir::Location<'c>,
) -> CompRegOperation<'c> {
    CompRegOperation::builder(context, location)
        .data(data)
        .input(input)
        .clk(clk)
        .build()
}
impl<'c> TryFrom<::melior::ir::operation::Operation<'c>> for CompRegOperation<'c> {
    type Error = ::melior::Error;
    fn try_from(operation: ::melior::ir::operation::Operation<'c>) -> Result<Self, Self::Error> {
        Ok(Self { operation })
    }
}
impl<'c> From<CompRegOperation<'c>> for ::melior::ir::operation::Operation<'c> {
    fn from(operation: CompRegOperation<'c>) -> Self {
        operation.operation
    }
}
#[doc = "A [`const_clock`](ConstClockOperation) operation. Produce constant clock value."]
#[doc = "\n\n"]
#[doc = "The constant operation produces a constant clock value.\n\n``` text\n  %clock = seq.const_clock low\n```\n"]
pub struct ConstClockOperation<'c> {
    operation: ::melior::ir::operation::Operation<'c>,
}
impl<'c> ConstClockOperation<'c> {
    #[doc = r" Returns a name."]
    pub fn name() -> &'static str {
        "seq.const_clock"
    }
    #[doc = r" Returns a generic operation."]
    pub fn as_operation(&self) -> &::melior::ir::operation::Operation<'c> {
        &self.operation
    }
    #[doc = r" Creates a builder."]
    pub fn builder(
        context: &'c ::melior::Context,
        location: ::melior::ir::Location<'c>,
    ) -> ConstClockOperationBuilder<
        'c,
        ::melior::dialect::ods::__private::Unset,
        ::melior::dialect::ods::__private::Unset,
    > {
        ConstClockOperationBuilder::new(context, location)
    }
    pub fn result(
        &self,
    ) -> Result<::melior::ir::operation::OperationResult<'c, '_>, ::melior::Error> {
        self.operation.result(0usize)
    }
    #[allow(clippy::needless_question_mark)]
    pub fn value(&self) -> Result<::melior::ir::attribute::Attribute<'c>, ::melior::Error> {
        Ok(self.operation.attribute("value")?.try_into()?)
    }
    pub fn set_value(&mut self, value: ::melior::ir::attribute::Attribute<'c>) {
        self.operation.set_attribute("value", value.into());
    }
}
#[doc = "A builder for a [`const_clock`](ConstClockOperation) operation."]
pub struct ConstClockOperationBuilder<'c, T0, A0> {
    builder: ::melior::ir::operation::OperationBuilder<'c>,
    context: &'c ::melior::Context,
    _state: ::std::marker::PhantomData<(T0, A0)>,
}
impl<'c>
    ConstClockOperationBuilder<
        'c,
        ::melior::dialect::ods::__private::Unset,
        ::melior::dialect::ods::__private::Unset,
    >
{
    pub fn new(context: &'c ::melior::Context, location: ::melior::ir::Location<'c>) -> Self {
        Self {
            context,
            builder: ::melior::ir::operation::OperationBuilder::new("seq.const_clock", location),
            _state: Default::default(),
        }
    }
}
impl<'c, A0> ConstClockOperationBuilder<'c, ::melior::dialect::ods::__private::Unset, A0> {
    pub fn result(
        self,
        result: ::melior::ir::Type<'c>,
    ) -> ConstClockOperationBuilder<'c, ::melior::dialect::ods::__private::Set, A0> {
        ConstClockOperationBuilder {
            context: self.context,
            builder: self.builder.add_results(&[result]),
            _state: Default::default(),
        }
    }
}
impl<'c, T0> ConstClockOperationBuilder<'c, T0, ::melior::dialect::ods::__private::Unset> {
    pub fn value(
        self,
        value: ::melior::ir::attribute::Attribute<'c>,
    ) -> ConstClockOperationBuilder<'c, T0, ::melior::dialect::ods::__private::Set> {
        ConstClockOperationBuilder {
            context: self.context,
            builder: self.builder.add_attributes(&[(
                ::melior::ir::Identifier::new(self.context, "value"),
                value.into(),
            )]),
            _state: Default::default(),
        }
    }
}
impl<'c>
    ConstClockOperationBuilder<
        'c,
        ::melior::dialect::ods::__private::Set,
        ::melior::dialect::ods::__private::Set,
    >
{
    pub fn build(self) -> ConstClockOperation<'c> {
        self.builder
            .build()
            .expect("valid operation")
            .try_into()
            .expect("should be a valid ConstClockOperation")
    }
}
#[allow(clippy::too_many_arguments)]
#[doc = "Creates a [`const_clock`](ConstClockOperation) operation."]
pub fn const_clock<'c>(
    context: &'c ::melior::Context,
    result: ::melior::ir::Type<'c>,
    value: ::melior::ir::attribute::Attribute<'c>,
    location: ::melior::ir::Location<'c>,
) -> ConstClockOperation<'c> {
    ConstClockOperation::builder(context, location)
        .result(result)
        .value(value)
        .build()
}
impl<'c> TryFrom<::melior::ir::operation::Operation<'c>> for ConstClockOperation<'c> {
    type Error = ::melior::Error;
    fn try_from(operation: ::melior::ir::operation::Operation<'c>) -> Result<Self, Self::Error> {
        Ok(Self { operation })
    }
}
impl<'c> From<ConstClockOperation<'c>> for ::melior::ir::operation::Operation<'c> {
    fn from(operation: ConstClockOperation<'c>) -> Self {
        operation.operation
    }
}
#[doc = "A [`fifo`](FIFOOperation) operation. A high-level FIFO operation."]
#[doc = "\n\n"]
#[doc = "This operation represents a high-level abstraction of a FIFO. Access to the\nFIFO is structural, and thus may be composed with other core RTL dialect\noperations.\nThe fifo operation is configurable with the following parameters:\n\n1.  Depth (cycles)\n2.  Almost full/empty thresholds (optional). If not provided, these will\n    be asserted when the FIFO is full/empty.\n\nLike `seq.hlmem` there are no guarantees that all possible fifo configuration\nare able to be lowered. Available lowering passes will pattern match on the\nrequested fifo configuration and attempt to provide a legal lowering.\n"]
pub struct FIFOOperation<'c> {
    operation: ::melior::ir::operation::Operation<'c>,
}
impl<'c> FIFOOperation<'c> {
    #[doc = r" Returns a name."]
    pub fn name() -> &'static str {
        "seq.fifo"
    }
    #[doc = r" Returns a generic operation."]
    pub fn as_operation(&self) -> &::melior::ir::operation::Operation<'c> {
        &self.operation
    }
    #[doc = r" Creates a builder."]
    pub fn builder(
        context: &'c ::melior::Context,
        location: ::melior::ir::Location<'c>,
    ) -> FIFOOperationBuilder<
        'c,
        ::melior::dialect::ods::__private::Unset,
        ::melior::dialect::ods::__private::Unset,
        ::melior::dialect::ods::__private::Unset,
        ::melior::dialect::ods::__private::Unset,
        ::melior::dialect::ods::__private::Unset,
        ::melior::dialect::ods::__private::Unset,
        ::melior::dialect::ods::__private::Unset,
        ::melior::dialect::ods::__private::Unset,
        ::melior::dialect::ods::__private::Unset,
    > {
        FIFOOperationBuilder::new(context, location)
    }
    pub fn output(
        &self,
    ) -> Result<::melior::ir::operation::OperationResult<'c, '_>, ::melior::Error> {
        let attribute = ::melior::ir::attribute::DenseI32ArrayAttribute::<'c>::try_from(
            self.operation.attribute("result_segment_sizes")?,
        )?;
        let start = (0..0usize)
            .map(|index| attribute.element(index))
            .collect::<Result<Vec<_>, _>>()?
            .into_iter()
            .sum::<i32>() as usize;
        let group_len = attribute.element(0usize)? as usize;
        self.operation.result(start)
    }
    pub fn full(
        &self,
    ) -> Result<::melior::ir::operation::OperationResult<'c, '_>, ::melior::Error> {
        let attribute = ::melior::ir::attribute::DenseI32ArrayAttribute::<'c>::try_from(
            self.operation.attribute("result_segment_sizes")?,
        )?;
        let start = (0..1usize)
            .map(|index| attribute.element(index))
            .collect::<Result<Vec<_>, _>>()?
            .into_iter()
            .sum::<i32>() as usize;
        let group_len = attribute.element(1usize)? as usize;
        self.operation.result(start)
    }
    pub fn empty(
        &self,
    ) -> Result<::melior::ir::operation::OperationResult<'c, '_>, ::melior::Error> {
        let attribute = ::melior::ir::attribute::DenseI32ArrayAttribute::<'c>::try_from(
            self.operation.attribute("result_segment_sizes")?,
        )?;
        let start = (0..2usize)
            .map(|index| attribute.element(index))
            .collect::<Result<Vec<_>, _>>()?
            .into_iter()
            .sum::<i32>() as usize;
        let group_len = attribute.element(2usize)? as usize;
        self.operation.result(start)
    }
    pub fn almost_full(
        &self,
    ) -> Result<::melior::ir::operation::OperationResult<'c, '_>, ::melior::Error> {
        let attribute = ::melior::ir::attribute::DenseI32ArrayAttribute::<'c>::try_from(
            self.operation.attribute("result_segment_sizes")?,
        )?;
        let start = (0..3usize)
            .map(|index| attribute.element(index))
            .collect::<Result<Vec<_>, _>>()?
            .into_iter()
            .sum::<i32>() as usize;
        let group_len = attribute.element(3usize)? as usize;
        if group_len == 0 {
            Err(::melior::Error::ResultNotFound("almostFull"))
        } else {
            self.operation.result(start)
        }
    }
    pub fn almost_empty(
        &self,
    ) -> Result<::melior::ir::operation::OperationResult<'c, '_>, ::melior::Error> {
        let attribute = ::melior::ir::attribute::DenseI32ArrayAttribute::<'c>::try_from(
            self.operation.attribute("result_segment_sizes")?,
        )?;
        let start = (0..4usize)
            .map(|index| attribute.element(index))
            .collect::<Result<Vec<_>, _>>()?
            .into_iter()
            .sum::<i32>() as usize;
        let group_len = attribute.element(4usize)? as usize;
        if group_len == 0 {
            Err(::melior::Error::ResultNotFound("almostEmpty"))
        } else {
            self.operation.result(start)
        }
    }
    pub fn input(&self) -> Result<::melior::ir::Value<'c, '_>, ::melior::Error> {
        self.operation.operand(0usize)
    }
    pub fn rd_en(&self) -> Result<::melior::ir::Value<'c, '_>, ::melior::Error> {
        self.operation.operand(1usize)
    }
    pub fn wr_en(&self) -> Result<::melior::ir::Value<'c, '_>, ::melior::Error> {
        self.operation.operand(2usize)
    }
    pub fn clk(&self) -> Result<::melior::ir::Value<'c, '_>, ::melior::Error> {
        self.operation.operand(3usize)
    }
    pub fn rst(&self) -> Result<::melior::ir::Value<'c, '_>, ::melior::Error> {
        self.operation.operand(4usize)
    }
    #[allow(clippy::needless_question_mark)]
    pub fn depth(&self) -> Result<::melior::ir::attribute::IntegerAttribute<'c>, ::melior::Error> {
        Ok(self.operation.attribute("depth")?.try_into()?)
    }
    pub fn set_depth(&mut self, value: ::melior::ir::attribute::IntegerAttribute<'c>) {
        self.operation.set_attribute("depth", value.into());
    }
    #[allow(clippy::needless_question_mark)]
    pub fn almost_full_threshold(
        &self,
    ) -> Result<::melior::ir::attribute::IntegerAttribute<'c>, ::melior::Error> {
        Ok(self
            .operation
            .attribute("almostFullThreshold")?
            .try_into()?)
    }
    pub fn set_almost_full_threshold(
        &mut self,
        value: ::melior::ir::attribute::IntegerAttribute<'c>,
    ) {
        self.operation
            .set_attribute("almostFullThreshold", value.into());
    }
    pub fn remove_almost_full_threshold(&mut self) -> Result<(), ::melior::Error> {
        self.operation.remove_attribute("almostFullThreshold")
    }
    #[allow(clippy::needless_question_mark)]
    pub fn almost_empty_threshold(
        &self,
    ) -> Result<::melior::ir::attribute::IntegerAttribute<'c>, ::melior::Error> {
        Ok(self
            .operation
            .attribute("almostEmptyThreshold")?
            .try_into()?)
    }
    pub fn set_almost_empty_threshold(
        &mut self,
        value: ::melior::ir::attribute::IntegerAttribute<'c>,
    ) {
        self.operation
            .set_attribute("almostEmptyThreshold", value.into());
    }
    pub fn remove_almost_empty_threshold(&mut self) -> Result<(), ::melior::Error> {
        self.operation.remove_attribute("almostEmptyThreshold")
    }
}
#[doc = "A builder for a [`fifo`](FIFOOperation) operation."]
pub struct FIFOOperationBuilder<'c, T0, T1, T2, O0, O1, O2, O3, O4, A0> {
    builder: ::melior::ir::operation::OperationBuilder<'c>,
    context: &'c ::melior::Context,
    _state: ::std::marker::PhantomData<(T0, T1, T2, O0, O1, O2, O3, O4, A0)>,
}
impl<'c>
    FIFOOperationBuilder<
        'c,
        ::melior::dialect::ods::__private::Unset,
        ::melior::dialect::ods::__private::Unset,
        ::melior::dialect::ods::__private::Unset,
        ::melior::dialect::ods::__private::Unset,
        ::melior::dialect::ods::__private::Unset,
        ::melior::dialect::ods::__private::Unset,
        ::melior::dialect::ods::__private::Unset,
        ::melior::dialect::ods::__private::Unset,
        ::melior::dialect::ods::__private::Unset,
    >
{
    pub fn new(context: &'c ::melior::Context, location: ::melior::ir::Location<'c>) -> Self {
        Self {
            context,
            builder: ::melior::ir::operation::OperationBuilder::new("seq.fifo", location),
            _state: Default::default(),
        }
    }
}
impl<'c, T1, T2, O0, O1, O2, O3, O4, A0>
    FIFOOperationBuilder<
        'c,
        ::melior::dialect::ods::__private::Unset,
        T1,
        T2,
        O0,
        O1,
        O2,
        O3,
        O4,
        A0,
    >
{
    pub fn output(
        self,
        output: ::melior::ir::Type<'c>,
    ) -> FIFOOperationBuilder<
        'c,
        ::melior::dialect::ods::__private::Set,
        T1,
        T2,
        O0,
        O1,
        O2,
        O3,
        O4,
        A0,
    > {
        FIFOOperationBuilder {
            context: self.context,
            builder: self.builder.add_results(&[output]),
            _state: Default::default(),
        }
    }
}
impl<'c, T2, O0, O1, O2, O3, O4, A0>
    FIFOOperationBuilder<
        'c,
        ::melior::dialect::ods::__private::Set,
        ::melior::dialect::ods::__private::Unset,
        T2,
        O0,
        O1,
        O2,
        O3,
        O4,
        A0,
    >
{
    pub fn full(
        self,
        full: ::melior::ir::Type<'c>,
    ) -> FIFOOperationBuilder<
        'c,
        ::melior::dialect::ods::__private::Set,
        ::melior::dialect::ods::__private::Set,
        T2,
        O0,
        O1,
        O2,
        O3,
        O4,
        A0,
    > {
        FIFOOperationBuilder {
            context: self.context,
            builder: self.builder.add_results(&[full]),
            _state: Default::default(),
        }
    }
}
impl<'c, O0, O1, O2, O3, O4, A0>
    FIFOOperationBuilder<
        'c,
        ::melior::dialect::ods::__private::Set,
        ::melior::dialect::ods::__private::Set,
        ::melior::dialect::ods::__private::Unset,
        O0,
        O1,
        O2,
        O3,
        O4,
        A0,
    >
{
    pub fn empty(
        self,
        empty: ::melior::ir::Type<'c>,
    ) -> FIFOOperationBuilder<
        'c,
        ::melior::dialect::ods::__private::Set,
        ::melior::dialect::ods::__private::Set,
        ::melior::dialect::ods::__private::Set,
        O0,
        O1,
        O2,
        O3,
        O4,
        A0,
    > {
        FIFOOperationBuilder {
            context: self.context,
            builder: self.builder.add_results(&[empty]),
            _state: Default::default(),
        }
    }
}
impl<'c, T0, T1, T2, O0, O1, O2, O3, O4, A0>
    FIFOOperationBuilder<'c, T0, T1, T2, O0, O1, O2, O3, O4, A0>
{
    pub fn almost_full(
        mut self,
        almost_full: ::melior::ir::Type<'c>,
    ) -> FIFOOperationBuilder<'c, T0, T1, T2, O0, O1, O2, O3, O4, A0> {
        self.builder = self.builder.add_results(&[almost_full]);
        self
    }
}
impl<'c, T0, T1, T2, O0, O1, O2, O3, O4, A0>
    FIFOOperationBuilder<'c, T0, T1, T2, O0, O1, O2, O3, O4, A0>
{
    pub fn almost_empty(
        mut self,
        almost_empty: ::melior::ir::Type<'c>,
    ) -> FIFOOperationBuilder<'c, T0, T1, T2, O0, O1, O2, O3, O4, A0> {
        self.builder = self.builder.add_results(&[almost_empty]);
        self
    }
}
impl<'c, T0, T1, T2, O1, O2, O3, O4, A0>
    FIFOOperationBuilder<
        'c,
        T0,
        T1,
        T2,
        ::melior::dialect::ods::__private::Unset,
        O1,
        O2,
        O3,
        O4,
        A0,
    >
{
    pub fn input(
        self,
        input: ::melior::ir::Value<'c, '_>,
    ) -> FIFOOperationBuilder<
        'c,
        T0,
        T1,
        T2,
        ::melior::dialect::ods::__private::Set,
        O1,
        O2,
        O3,
        O4,
        A0,
    > {
        FIFOOperationBuilder {
            context: self.context,
            builder: self.builder.add_operands(&[input]),
            _state: Default::default(),
        }
    }
}
impl<'c, T0, T1, T2, O2, O3, O4, A0>
    FIFOOperationBuilder<
        'c,
        T0,
        T1,
        T2,
        ::melior::dialect::ods::__private::Set,
        ::melior::dialect::ods::__private::Unset,
        O2,
        O3,
        O4,
        A0,
    >
{
    pub fn rd_en(
        self,
        rd_en: ::melior::ir::Value<'c, '_>,
    ) -> FIFOOperationBuilder<
        'c,
        T0,
        T1,
        T2,
        ::melior::dialect::ods::__private::Set,
        ::melior::dialect::ods::__private::Set,
        O2,
        O3,
        O4,
        A0,
    > {
        FIFOOperationBuilder {
            context: self.context,
            builder: self.builder.add_operands(&[rd_en]),
            _state: Default::default(),
        }
    }
}
impl<'c, T0, T1, T2, O3, O4, A0>
    FIFOOperationBuilder<
        'c,
        T0,
        T1,
        T2,
        ::melior::dialect::ods::__private::Set,
        ::melior::dialect::ods::__private::Set,
        ::melior::dialect::ods::__private::Unset,
        O3,
        O4,
        A0,
    >
{
    pub fn wr_en(
        self,
        wr_en: ::melior::ir::Value<'c, '_>,
    ) -> FIFOOperationBuilder<
        'c,
        T0,
        T1,
        T2,
        ::melior::dialect::ods::__private::Set,
        ::melior::dialect::ods::__private::Set,
        ::melior::dialect::ods::__private::Set,
        O3,
        O4,
        A0,
    > {
        FIFOOperationBuilder {
            context: self.context,
            builder: self.builder.add_operands(&[wr_en]),
            _state: Default::default(),
        }
    }
}
impl<'c, T0, T1, T2, O4, A0>
    FIFOOperationBuilder<
        'c,
        T0,
        T1,
        T2,
        ::melior::dialect::ods::__private::Set,
        ::melior::dialect::ods::__private::Set,
        ::melior::dialect::ods::__private::Set,
        ::melior::dialect::ods::__private::Unset,
        O4,
        A0,
    >
{
    pub fn clk(
        self,
        clk: ::melior::ir::Value<'c, '_>,
    ) -> FIFOOperationBuilder<
        'c,
        T0,
        T1,
        T2,
        ::melior::dialect::ods::__private::Set,
        ::melior::dialect::ods::__private::Set,
        ::melior::dialect::ods::__private::Set,
        ::melior::dialect::ods::__private::Set,
        O4,
        A0,
    > {
        FIFOOperationBuilder {
            context: self.context,
            builder: self.builder.add_operands(&[clk]),
            _state: Default::default(),
        }
    }
}
impl<'c, T0, T1, T2, A0>
    FIFOOperationBuilder<
        'c,
        T0,
        T1,
        T2,
        ::melior::dialect::ods::__private::Set,
        ::melior::dialect::ods::__private::Set,
        ::melior::dialect::ods::__private::Set,
        ::melior::dialect::ods::__private::Set,
        ::melior::dialect::ods::__private::Unset,
        A0,
    >
{
    pub fn rst(
        self,
        rst: ::melior::ir::Value<'c, '_>,
    ) -> FIFOOperationBuilder<
        'c,
        T0,
        T1,
        T2,
        ::melior::dialect::ods::__private::Set,
        ::melior::dialect::ods::__private::Set,
        ::melior::dialect::ods::__private::Set,
        ::melior::dialect::ods::__private::Set,
        ::melior::dialect::ods::__private::Set,
        A0,
    > {
        FIFOOperationBuilder {
            context: self.context,
            builder: self.builder.add_operands(&[rst]),
            _state: Default::default(),
        }
    }
}
impl<'c, T0, T1, T2, O0, O1, O2, O3, O4>
    FIFOOperationBuilder<
        'c,
        T0,
        T1,
        T2,
        O0,
        O1,
        O2,
        O3,
        O4,
        ::melior::dialect::ods::__private::Unset,
    >
{
    pub fn depth(
        self,
        depth: ::melior::ir::attribute::IntegerAttribute<'c>,
    ) -> FIFOOperationBuilder<
        'c,
        T0,
        T1,
        T2,
        O0,
        O1,
        O2,
        O3,
        O4,
        ::melior::dialect::ods::__private::Set,
    > {
        FIFOOperationBuilder {
            context: self.context,
            builder: self.builder.add_attributes(&[(
                ::melior::ir::Identifier::new(self.context, "depth"),
                depth.into(),
            )]),
            _state: Default::default(),
        }
    }
}
impl<'c, T0, T1, T2, O0, O1, O2, O3, O4, A0>
    FIFOOperationBuilder<'c, T0, T1, T2, O0, O1, O2, O3, O4, A0>
{
    pub fn almost_full_threshold(
        mut self,
        almost_full_threshold: ::melior::ir::attribute::IntegerAttribute<'c>,
    ) -> FIFOOperationBuilder<'c, T0, T1, T2, O0, O1, O2, O3, O4, A0> {
        self.builder = self.builder.add_attributes(&[(
            ::melior::ir::Identifier::new(self.context, "almostFullThreshold"),
            almost_full_threshold.into(),
        )]);
        self
    }
}
impl<'c, T0, T1, T2, O0, O1, O2, O3, O4, A0>
    FIFOOperationBuilder<'c, T0, T1, T2, O0, O1, O2, O3, O4, A0>
{
    pub fn almost_empty_threshold(
        mut self,
        almost_empty_threshold: ::melior::ir::attribute::IntegerAttribute<'c>,
    ) -> FIFOOperationBuilder<'c, T0, T1, T2, O0, O1, O2, O3, O4, A0> {
        self.builder = self.builder.add_attributes(&[(
            ::melior::ir::Identifier::new(self.context, "almostEmptyThreshold"),
            almost_empty_threshold.into(),
        )]);
        self
    }
}
impl<'c>
    FIFOOperationBuilder<
        'c,
        ::melior::dialect::ods::__private::Set,
        ::melior::dialect::ods::__private::Set,
        ::melior::dialect::ods::__private::Set,
        ::melior::dialect::ods::__private::Set,
        ::melior::dialect::ods::__private::Set,
        ::melior::dialect::ods::__private::Set,
        ::melior::dialect::ods::__private::Set,
        ::melior::dialect::ods::__private::Set,
        ::melior::dialect::ods::__private::Set,
    >
{
    pub fn build(self) -> FIFOOperation<'c> {
        self.builder
            .build()
            .expect("valid operation")
            .try_into()
            .expect("should be a valid FIFOOperation")
    }
}
#[allow(clippy::too_many_arguments)]
#[doc = "Creates a [`fifo`](FIFOOperation) operation."]
pub fn fifo<'c>(
    context: &'c ::melior::Context,
    output: ::melior::ir::Type<'c>,
    full: ::melior::ir::Type<'c>,
    empty: ::melior::ir::Type<'c>,
    input: ::melior::ir::Value<'c, '_>,
    rd_en: ::melior::ir::Value<'c, '_>,
    wr_en: ::melior::ir::Value<'c, '_>,
    clk: ::melior::ir::Value<'c, '_>,
    rst: ::melior::ir::Value<'c, '_>,
    depth: ::melior::ir::attribute::IntegerAttribute<'c>,
    location: ::melior::ir::Location<'c>,
) -> FIFOOperation<'c> {
    FIFOOperation::builder(context, location)
        .output(output)
        .full(full)
        .empty(empty)
        .input(input)
        .rd_en(rd_en)
        .wr_en(wr_en)
        .clk(clk)
        .rst(rst)
        .depth(depth)
        .build()
}
impl<'c> TryFrom<::melior::ir::operation::Operation<'c>> for FIFOOperation<'c> {
    type Error = ::melior::Error;
    fn try_from(operation: ::melior::ir::operation::Operation<'c>) -> Result<Self, Self::Error> {
        Ok(Self { operation })
    }
}
impl<'c> From<FIFOOperation<'c>> for ::melior::ir::operation::Operation<'c> {
    fn from(operation: FIFOOperation<'c>) -> Self {
        operation.operation
    }
}
#[doc = "A [`firmem`](FirMemOperation) operation. A FIRRTL-flavored memory."]
#[doc = "\n\n"]
#[doc = "The `seq.firmem` op represents memories lowered from the FIRRTL dialect. It\nis used to capture some of the peculiarities of what FIRRTL expects from\nmemories, while still representing them at the HW dialect level.\n\nA `seq.firmem` declares the memory and captures the memory-level parameters\nsuch as width and depth or how read/write collisions are resolved. The read,\nwrite, and read-write ports are expressed as separate operations that take\nthe declared memory as an operand.\n"]
pub struct FirMemOperation<'c> {
    operation: ::melior::ir::operation::Operation<'c>,
}
impl<'c> FirMemOperation<'c> {
    #[doc = r" Returns a name."]
    pub fn name() -> &'static str {
        "seq.firmem"
    }
    #[doc = r" Returns a generic operation."]
    pub fn as_operation(&self) -> &::melior::ir::operation::Operation<'c> {
        &self.operation
    }
    #[doc = r" Creates a builder."]
    pub fn builder(
        context: &'c ::melior::Context,
        location: ::melior::ir::Location<'c>,
    ) -> FirMemOperationBuilder<
        'c,
        ::melior::dialect::ods::__private::Unset,
        ::melior::dialect::ods::__private::Unset,
        ::melior::dialect::ods::__private::Unset,
        ::melior::dialect::ods::__private::Unset,
        ::melior::dialect::ods::__private::Unset,
    > {
        FirMemOperationBuilder::new(context, location)
    }
    pub fn memory(
        &self,
    ) -> Result<::melior::ir::operation::OperationResult<'c, '_>, ::melior::Error> {
        self.operation.result(0usize)
    }
    #[allow(clippy::needless_question_mark)]
    pub fn read_latency(
        &self,
    ) -> Result<::melior::ir::attribute::IntegerAttribute<'c>, ::melior::Error> {
        Ok(self.operation.attribute("readLatency")?.try_into()?)
    }
    pub fn set_read_latency(&mut self, value: ::melior::ir::attribute::IntegerAttribute<'c>) {
        self.operation.set_attribute("readLatency", value.into());
    }
    #[allow(clippy::needless_question_mark)]
    pub fn write_latency(
        &self,
    ) -> Result<::melior::ir::attribute::IntegerAttribute<'c>, ::melior::Error> {
        Ok(self.operation.attribute("writeLatency")?.try_into()?)
    }
    pub fn set_write_latency(&mut self, value: ::melior::ir::attribute::IntegerAttribute<'c>) {
        self.operation.set_attribute("writeLatency", value.into());
    }
    #[allow(clippy::needless_question_mark)]
    pub fn ruw(&self) -> Result<::melior::ir::attribute::Attribute<'c>, ::melior::Error> {
        Ok(self.operation.attribute("ruw")?.try_into()?)
    }
    pub fn set_ruw(&mut self, value: ::melior::ir::attribute::Attribute<'c>) {
        self.operation.set_attribute("ruw", value.into());
    }
    #[allow(clippy::needless_question_mark)]
    pub fn wuw(&self) -> Result<::melior::ir::attribute::Attribute<'c>, ::melior::Error> {
        Ok(self.operation.attribute("wuw")?.try_into()?)
    }
    pub fn set_wuw(&mut self, value: ::melior::ir::attribute::Attribute<'c>) {
        self.operation.set_attribute("wuw", value.into());
    }
    #[allow(clippy::needless_question_mark)]
    pub fn _name(&self) -> Result<::melior::ir::attribute::StringAttribute<'c>, ::melior::Error> {
        Ok(self.operation.attribute("name")?.try_into()?)
    }
    pub fn set_name(&mut self, value: ::melior::ir::attribute::StringAttribute<'c>) {
        self.operation.set_attribute("name", value.into());
    }
    pub fn remove_name(&mut self) -> Result<(), ::melior::Error> {
        self.operation.remove_attribute("name")
    }
    #[allow(clippy::needless_question_mark)]
    pub fn inner_sym(&self) -> Result<::melior::ir::attribute::Attribute<'c>, ::melior::Error> {
        Ok(self.operation.attribute("inner_sym")?.try_into()?)
    }
    pub fn set_inner_sym(&mut self, value: ::melior::ir::attribute::Attribute<'c>) {
        self.operation.set_attribute("inner_sym", value.into());
    }
    pub fn remove_inner_sym(&mut self) -> Result<(), ::melior::Error> {
        self.operation.remove_attribute("inner_sym")
    }
    #[allow(clippy::needless_question_mark)]
    pub fn init(&self) -> Result<::melior::ir::attribute::Attribute<'c>, ::melior::Error> {
        Ok(self.operation.attribute("init")?.try_into()?)
    }
    pub fn set_init(&mut self, value: ::melior::ir::attribute::Attribute<'c>) {
        self.operation.set_attribute("init", value.into());
    }
    pub fn remove_init(&mut self) -> Result<(), ::melior::Error> {
        self.operation.remove_attribute("init")
    }
    #[allow(clippy::needless_question_mark)]
    pub fn prefix(&self) -> Result<::melior::ir::attribute::StringAttribute<'c>, ::melior::Error> {
        Ok(self.operation.attribute("prefix")?.try_into()?)
    }
    pub fn set_prefix(&mut self, value: ::melior::ir::attribute::StringAttribute<'c>) {
        self.operation.set_attribute("prefix", value.into());
    }
    pub fn remove_prefix(&mut self) -> Result<(), ::melior::Error> {
        self.operation.remove_attribute("prefix")
    }
    #[allow(clippy::needless_question_mark)]
    pub fn output_file(&self) -> Result<::melior::ir::attribute::Attribute<'c>, ::melior::Error> {
        Ok(self.operation.attribute("output_file")?.try_into()?)
    }
    pub fn set_output_file(&mut self, value: ::melior::ir::attribute::Attribute<'c>) {
        self.operation.set_attribute("output_file", value.into());
    }
    pub fn remove_output_file(&mut self) -> Result<(), ::melior::Error> {
        self.operation.remove_attribute("output_file")
    }
}
#[doc = "A builder for a [`firmem`](FirMemOperation) operation."]
pub struct FirMemOperationBuilder<'c, T0, A0, A1, A2, A3> {
    builder: ::melior::ir::operation::OperationBuilder<'c>,
    context: &'c ::melior::Context,
    _state: ::std::marker::PhantomData<(T0, A0, A1, A2, A3)>,
}
impl<'c>
    FirMemOperationBuilder<
        'c,
        ::melior::dialect::ods::__private::Unset,
        ::melior::dialect::ods::__private::Unset,
        ::melior::dialect::ods::__private::Unset,
        ::melior::dialect::ods::__private::Unset,
        ::melior::dialect::ods::__private::Unset,
    >
{
    pub fn new(context: &'c ::melior::Context, location: ::melior::ir::Location<'c>) -> Self {
        Self {
            context,
            builder: ::melior::ir::operation::OperationBuilder::new("seq.firmem", location),
            _state: Default::default(),
        }
    }
}
impl<'c, A0, A1, A2, A3>
    FirMemOperationBuilder<'c, ::melior::dialect::ods::__private::Unset, A0, A1, A2, A3>
{
    pub fn memory(
        self,
        memory: ::melior::ir::Type<'c>,
    ) -> FirMemOperationBuilder<'c, ::melior::dialect::ods::__private::Set, A0, A1, A2, A3> {
        FirMemOperationBuilder {
            context: self.context,
            builder: self.builder.add_results(&[memory]),
            _state: Default::default(),
        }
    }
}
impl<'c, T0, A1, A2, A3>
    FirMemOperationBuilder<'c, T0, ::melior::dialect::ods::__private::Unset, A1, A2, A3>
{
    pub fn read_latency(
        self,
        read_latency: ::melior::ir::attribute::IntegerAttribute<'c>,
    ) -> FirMemOperationBuilder<'c, T0, ::melior::dialect::ods::__private::Set, A1, A2, A3> {
        FirMemOperationBuilder {
            context: self.context,
            builder: self.builder.add_attributes(&[(
                ::melior::ir::Identifier::new(self.context, "readLatency"),
                read_latency.into(),
            )]),
            _state: Default::default(),
        }
    }
}
impl<'c, T0, A0, A2, A3>
    FirMemOperationBuilder<'c, T0, A0, ::melior::dialect::ods::__private::Unset, A2, A3>
{
    pub fn write_latency(
        self,
        write_latency: ::melior::ir::attribute::IntegerAttribute<'c>,
    ) -> FirMemOperationBuilder<'c, T0, A0, ::melior::dialect::ods::__private::Set, A2, A3> {
        FirMemOperationBuilder {
            context: self.context,
            builder: self.builder.add_attributes(&[(
                ::melior::ir::Identifier::new(self.context, "writeLatency"),
                write_latency.into(),
            )]),
            _state: Default::default(),
        }
    }
}
impl<'c, T0, A0, A1, A3>
    FirMemOperationBuilder<'c, T0, A0, A1, ::melior::dialect::ods::__private::Unset, A3>
{
    pub fn ruw(
        self,
        ruw: ::melior::ir::attribute::Attribute<'c>,
    ) -> FirMemOperationBuilder<'c, T0, A0, A1, ::melior::dialect::ods::__private::Set, A3> {
        FirMemOperationBuilder {
            context: self.context,
            builder: self.builder.add_attributes(&[(
                ::melior::ir::Identifier::new(self.context, "ruw"),
                ruw.into(),
            )]),
            _state: Default::default(),
        }
    }
}
impl<'c, T0, A0, A1, A2>
    FirMemOperationBuilder<'c, T0, A0, A1, A2, ::melior::dialect::ods::__private::Unset>
{
    pub fn wuw(
        self,
        wuw: ::melior::ir::attribute::Attribute<'c>,
    ) -> FirMemOperationBuilder<'c, T0, A0, A1, A2, ::melior::dialect::ods::__private::Set> {
        FirMemOperationBuilder {
            context: self.context,
            builder: self.builder.add_attributes(&[(
                ::melior::ir::Identifier::new(self.context, "wuw"),
                wuw.into(),
            )]),
            _state: Default::default(),
        }
    }
}
impl<'c, T0, A0, A1, A2, A3> FirMemOperationBuilder<'c, T0, A0, A1, A2, A3> {
    pub fn _name(
        mut self,
        _name: ::melior::ir::attribute::StringAttribute<'c>,
    ) -> FirMemOperationBuilder<'c, T0, A0, A1, A2, A3> {
        self.builder = self.builder.add_attributes(&[(
            ::melior::ir::Identifier::new(self.context, "name"),
            _name.into(),
        )]);
        self
    }
}
impl<'c, T0, A0, A1, A2, A3> FirMemOperationBuilder<'c, T0, A0, A1, A2, A3> {
    pub fn inner_sym(
        mut self,
        inner_sym: ::melior::ir::attribute::Attribute<'c>,
    ) -> FirMemOperationBuilder<'c, T0, A0, A1, A2, A3> {
        self.builder = self.builder.add_attributes(&[(
            ::melior::ir::Identifier::new(self.context, "inner_sym"),
            inner_sym.into(),
        )]);
        self
    }
}
impl<'c, T0, A0, A1, A2, A3> FirMemOperationBuilder<'c, T0, A0, A1, A2, A3> {
    pub fn init(
        mut self,
        init: ::melior::ir::attribute::Attribute<'c>,
    ) -> FirMemOperationBuilder<'c, T0, A0, A1, A2, A3> {
        self.builder = self.builder.add_attributes(&[(
            ::melior::ir::Identifier::new(self.context, "init"),
            init.into(),
        )]);
        self
    }
}
impl<'c, T0, A0, A1, A2, A3> FirMemOperationBuilder<'c, T0, A0, A1, A2, A3> {
    pub fn prefix(
        mut self,
        prefix: ::melior::ir::attribute::StringAttribute<'c>,
    ) -> FirMemOperationBuilder<'c, T0, A0, A1, A2, A3> {
        self.builder = self.builder.add_attributes(&[(
            ::melior::ir::Identifier::new(self.context, "prefix"),
            prefix.into(),
        )]);
        self
    }
}
impl<'c, T0, A0, A1, A2, A3> FirMemOperationBuilder<'c, T0, A0, A1, A2, A3> {
    pub fn output_file(
        mut self,
        output_file: ::melior::ir::attribute::Attribute<'c>,
    ) -> FirMemOperationBuilder<'c, T0, A0, A1, A2, A3> {
        self.builder = self.builder.add_attributes(&[(
            ::melior::ir::Identifier::new(self.context, "output_file"),
            output_file.into(),
        )]);
        self
    }
}
impl<'c>
    FirMemOperationBuilder<
        'c,
        ::melior::dialect::ods::__private::Set,
        ::melior::dialect::ods::__private::Set,
        ::melior::dialect::ods::__private::Set,
        ::melior::dialect::ods::__private::Set,
        ::melior::dialect::ods::__private::Set,
    >
{
    pub fn build(self) -> FirMemOperation<'c> {
        self.builder
            .build()
            .expect("valid operation")
            .try_into()
            .expect("should be a valid FirMemOperation")
    }
}
#[allow(clippy::too_many_arguments)]
#[doc = "Creates a [`firmem`](FirMemOperation) operation."]
pub fn firmem<'c>(
    context: &'c ::melior::Context,
    memory: ::melior::ir::Type<'c>,
    read_latency: ::melior::ir::attribute::IntegerAttribute<'c>,
    write_latency: ::melior::ir::attribute::IntegerAttribute<'c>,
    ruw: ::melior::ir::attribute::Attribute<'c>,
    wuw: ::melior::ir::attribute::Attribute<'c>,
    location: ::melior::ir::Location<'c>,
) -> FirMemOperation<'c> {
    FirMemOperation::builder(context, location)
        .memory(memory)
        .read_latency(read_latency)
        .write_latency(write_latency)
        .ruw(ruw)
        .wuw(wuw)
        .build()
}
impl<'c> TryFrom<::melior::ir::operation::Operation<'c>> for FirMemOperation<'c> {
    type Error = ::melior::Error;
    fn try_from(operation: ::melior::ir::operation::Operation<'c>) -> Result<Self, Self::Error> {
        Ok(Self { operation })
    }
}
impl<'c> From<FirMemOperation<'c>> for ::melior::ir::operation::Operation<'c> {
    fn from(operation: FirMemOperation<'c>) -> Self {
        operation.operation
    }
}
#[doc = "A [`firmem.read_port`](FirMemReadOperation) operation. A memory read port."]
#[doc = "\n\n"]
#[doc = "The `seq.firmem.read_port` op represents a read port on a `seq.firmem`\nmemory. It takes the memory as an operand, together with the address to\nbe read, the clock on which the read is synchronized, and an optional\nenable. Omitting the enable operand has the same effect as passing a\nconstant `true` to it.\n"]
pub struct FirMemReadOperation<'c> {
    operation: ::melior::ir::operation::Operation<'c>,
}
impl<'c> FirMemReadOperation<'c> {
    #[doc = r" Returns a name."]
    pub fn name() -> &'static str {
        "seq.firmem.read_port"
    }
    #[doc = r" Returns a generic operation."]
    pub fn as_operation(&self) -> &::melior::ir::operation::Operation<'c> {
        &self.operation
    }
    #[doc = r" Creates a builder."]
    pub fn builder(
        context: &'c ::melior::Context,
        location: ::melior::ir::Location<'c>,
    ) -> FirMemReadOperationBuilder<
        'c,
        ::melior::dialect::ods::__private::Unset,
        ::melior::dialect::ods::__private::Unset,
        ::melior::dialect::ods::__private::Unset,
        ::melior::dialect::ods::__private::Unset,
    > {
        FirMemReadOperationBuilder::new(context, location)
    }
    pub fn data(
        &self,
    ) -> Result<::melior::ir::operation::OperationResult<'c, '_>, ::melior::Error> {
        self.operation.result(0usize)
    }
    pub fn memory(&self) -> Result<::melior::ir::Value<'c, '_>, ::melior::Error> {
        self.operation.operand(0usize)
    }
    pub fn address(&self) -> Result<::melior::ir::Value<'c, '_>, ::melior::Error> {
        self.operation.operand(1usize)
    }
    pub fn clk(&self) -> Result<::melior::ir::Value<'c, '_>, ::melior::Error> {
        self.operation.operand(2usize)
    }
    pub fn enable(&self) -> Result<::melior::ir::Value<'c, '_>, ::melior::Error> {
        if self.operation.operand_count() < 4usize {
            Err(::melior::Error::OperandNotFound("enable"))
        } else {
            self.operation.operand(3usize)
        }
    }
}
#[doc = "A builder for a [`firmem.read_port`](FirMemReadOperation) operation."]
pub struct FirMemReadOperationBuilder<'c, T0, O0, O1, O2> {
    builder: ::melior::ir::operation::OperationBuilder<'c>,
    context: &'c ::melior::Context,
    _state: ::std::marker::PhantomData<(T0, O0, O1, O2)>,
}
impl<'c>
    FirMemReadOperationBuilder<
        'c,
        ::melior::dialect::ods::__private::Unset,
        ::melior::dialect::ods::__private::Unset,
        ::melior::dialect::ods::__private::Unset,
        ::melior::dialect::ods::__private::Unset,
    >
{
    pub fn new(context: &'c ::melior::Context, location: ::melior::ir::Location<'c>) -> Self {
        Self {
            context,
            builder: ::melior::ir::operation::OperationBuilder::new(
                "seq.firmem.read_port",
                location,
            ),
            _state: Default::default(),
        }
    }
}
impl<'c, O0, O1, O2>
    FirMemReadOperationBuilder<'c, ::melior::dialect::ods::__private::Unset, O0, O1, O2>
{
    pub fn data(
        self,
        data: ::melior::ir::Type<'c>,
    ) -> FirMemReadOperationBuilder<'c, ::melior::dialect::ods::__private::Set, O0, O1, O2> {
        FirMemReadOperationBuilder {
            context: self.context,
            builder: self.builder.add_results(&[data]),
            _state: Default::default(),
        }
    }
}
impl<'c, T0, O1, O2>
    FirMemReadOperationBuilder<'c, T0, ::melior::dialect::ods::__private::Unset, O1, O2>
{
    pub fn memory(
        self,
        memory: ::melior::ir::Value<'c, '_>,
    ) -> FirMemReadOperationBuilder<'c, T0, ::melior::dialect::ods::__private::Set, O1, O2> {
        FirMemReadOperationBuilder {
            context: self.context,
            builder: self.builder.add_operands(&[memory]),
            _state: Default::default(),
        }
    }
}
impl<'c, T0, O2>
    FirMemReadOperationBuilder<
        'c,
        T0,
        ::melior::dialect::ods::__private::Set,
        ::melior::dialect::ods::__private::Unset,
        O2,
    >
{
    pub fn address(
        self,
        address: ::melior::ir::Value<'c, '_>,
    ) -> FirMemReadOperationBuilder<
        'c,
        T0,
        ::melior::dialect::ods::__private::Set,
        ::melior::dialect::ods::__private::Set,
        O2,
    > {
        FirMemReadOperationBuilder {
            context: self.context,
            builder: self.builder.add_operands(&[address]),
            _state: Default::default(),
        }
    }
}
impl<'c, T0>
    FirMemReadOperationBuilder<
        'c,
        T0,
        ::melior::dialect::ods::__private::Set,
        ::melior::dialect::ods::__private::Set,
        ::melior::dialect::ods::__private::Unset,
    >
{
    pub fn clk(
        self,
        clk: ::melior::ir::Value<'c, '_>,
    ) -> FirMemReadOperationBuilder<
        'c,
        T0,
        ::melior::dialect::ods::__private::Set,
        ::melior::dialect::ods::__private::Set,
        ::melior::dialect::ods::__private::Set,
    > {
        FirMemReadOperationBuilder {
            context: self.context,
            builder: self.builder.add_operands(&[clk]),
            _state: Default::default(),
        }
    }
}
impl<'c, T0, O0, O1, O2> FirMemReadOperationBuilder<'c, T0, O0, O1, O2> {
    pub fn enable(
        mut self,
        enable: ::melior::ir::Value<'c, '_>,
    ) -> FirMemReadOperationBuilder<'c, T0, O0, O1, O2> {
        self.builder = self.builder.add_operands(&[enable]);
        self
    }
}
impl<'c>
    FirMemReadOperationBuilder<
        'c,
        ::melior::dialect::ods::__private::Set,
        ::melior::dialect::ods::__private::Set,
        ::melior::dialect::ods::__private::Set,
        ::melior::dialect::ods::__private::Set,
    >
{
    pub fn build(self) -> FirMemReadOperation<'c> {
        self.builder
            .build()
            .expect("valid operation")
            .try_into()
            .expect("should be a valid FirMemReadOperation")
    }
}
#[allow(clippy::too_many_arguments)]
#[doc = "Creates a [`firmem.read_port`](FirMemReadOperation) operation."]
pub fn firmem_read_port<'c>(
    context: &'c ::melior::Context,
    data: ::melior::ir::Type<'c>,
    memory: ::melior::ir::Value<'c, '_>,
    address: ::melior::ir::Value<'c, '_>,
    clk: ::melior::ir::Value<'c, '_>,
    location: ::melior::ir::Location<'c>,
) -> FirMemReadOperation<'c> {
    FirMemReadOperation::builder(context, location)
        .data(data)
        .memory(memory)
        .address(address)
        .clk(clk)
        .build()
}
impl<'c> TryFrom<::melior::ir::operation::Operation<'c>> for FirMemReadOperation<'c> {
    type Error = ::melior::Error;
    fn try_from(operation: ::melior::ir::operation::Operation<'c>) -> Result<Self, Self::Error> {
        Ok(Self { operation })
    }
}
impl<'c> From<FirMemReadOperation<'c>> for ::melior::ir::operation::Operation<'c> {
    fn from(operation: FirMemReadOperation<'c>) -> Self {
        operation.operation
    }
}
#[doc = "A [`firmem.read_write_port`](FirMemReadWriteOperation) operation. A memory read-write port."]
#[doc = "\n\n"]
#[doc = "The `seq.firmem.read_write_port` op represents a read-write port on a\n`seq.firmem` memory. It takes the memory as an operand, together with the\naddress and data to be written, a mode operand indicating whether the port\nshould perform a read (`mode=0`) or a write (`mode=1`), the clock on which\nthe read and write is synchronized, an optional enable, and and optional\nwrite mask. Omitting the enable operand has the same effect as passing a\nconstant `true` to it. Omitting the write mask operand has the same effect\nas passing an all-ones value to it. A write mask operand can only be present\nif the `seq.firmem` specifies a mask width; otherwise it must be omitted.\n"]
pub struct FirMemReadWriteOperation<'c> {
    operation: ::melior::ir::operation::Operation<'c>,
}
impl<'c> FirMemReadWriteOperation<'c> {
    #[doc = r" Returns a name."]
    pub fn name() -> &'static str {
        "seq.firmem.read_write_port"
    }
    #[doc = r" Returns a generic operation."]
    pub fn as_operation(&self) -> &::melior::ir::operation::Operation<'c> {
        &self.operation
    }
    #[doc = r" Creates a builder."]
    pub fn builder(
        context: &'c ::melior::Context,
        location: ::melior::ir::Location<'c>,
    ) -> FirMemReadWriteOperationBuilder<
        'c,
        ::melior::dialect::ods::__private::Unset,
        ::melior::dialect::ods::__private::Unset,
        ::melior::dialect::ods::__private::Unset,
        ::melior::dialect::ods::__private::Unset,
        ::melior::dialect::ods::__private::Unset,
        ::melior::dialect::ods::__private::Unset,
    > {
        FirMemReadWriteOperationBuilder::new(context, location)
    }
    pub fn read_data(
        &self,
    ) -> Result<::melior::ir::operation::OperationResult<'c, '_>, ::melior::Error> {
        self.operation.result(0usize)
    }
    pub fn memory(&self) -> Result<::melior::ir::Value<'c, '_>, ::melior::Error> {
        let attribute = ::melior::ir::attribute::DenseI32ArrayAttribute::<'c>::try_from(
            self.operation.attribute("operand_segment_sizes")?,
        )?;
        let start = (0..0usize)
            .map(|index| attribute.element(index))
            .collect::<Result<Vec<_>, _>>()?
            .into_iter()
            .sum::<i32>() as usize;
        let group_len = attribute.element(0usize)? as usize;
        self.operation.operand(start)
    }
    pub fn address(&self) -> Result<::melior::ir::Value<'c, '_>, ::melior::Error> {
        let attribute = ::melior::ir::attribute::DenseI32ArrayAttribute::<'c>::try_from(
            self.operation.attribute("operand_segment_sizes")?,
        )?;
        let start = (0..1usize)
            .map(|index| attribute.element(index))
            .collect::<Result<Vec<_>, _>>()?
            .into_iter()
            .sum::<i32>() as usize;
        let group_len = attribute.element(1usize)? as usize;
        self.operation.operand(start)
    }
    pub fn clk(&self) -> Result<::melior::ir::Value<'c, '_>, ::melior::Error> {
        let attribute = ::melior::ir::attribute::DenseI32ArrayAttribute::<'c>::try_from(
            self.operation.attribute("operand_segment_sizes")?,
        )?;
        let start = (0..2usize)
            .map(|index| attribute.element(index))
            .collect::<Result<Vec<_>, _>>()?
            .into_iter()
            .sum::<i32>() as usize;
        let group_len = attribute.element(2usize)? as usize;
        self.operation.operand(start)
    }
    pub fn enable(&self) -> Result<::melior::ir::Value<'c, '_>, ::melior::Error> {
        let attribute = ::melior::ir::attribute::DenseI32ArrayAttribute::<'c>::try_from(
            self.operation.attribute("operand_segment_sizes")?,
        )?;
        let start = (0..3usize)
            .map(|index| attribute.element(index))
            .collect::<Result<Vec<_>, _>>()?
            .into_iter()
            .sum::<i32>() as usize;
        let group_len = attribute.element(3usize)? as usize;
        if group_len == 0 {
            Err(::melior::Error::OperandNotFound("enable"))
        } else {
            self.operation.operand(start)
        }
    }
    pub fn write_data(&self) -> Result<::melior::ir::Value<'c, '_>, ::melior::Error> {
        let attribute = ::melior::ir::attribute::DenseI32ArrayAttribute::<'c>::try_from(
            self.operation.attribute("operand_segment_sizes")?,
        )?;
        let start = (0..4usize)
            .map(|index| attribute.element(index))
            .collect::<Result<Vec<_>, _>>()?
            .into_iter()
            .sum::<i32>() as usize;
        let group_len = attribute.element(4usize)? as usize;
        self.operation.operand(start)
    }
    pub fn mode(&self) -> Result<::melior::ir::Value<'c, '_>, ::melior::Error> {
        let attribute = ::melior::ir::attribute::DenseI32ArrayAttribute::<'c>::try_from(
            self.operation.attribute("operand_segment_sizes")?,
        )?;
        let start = (0..5usize)
            .map(|index| attribute.element(index))
            .collect::<Result<Vec<_>, _>>()?
            .into_iter()
            .sum::<i32>() as usize;
        let group_len = attribute.element(5usize)? as usize;
        self.operation.operand(start)
    }
    pub fn mask(&self) -> Result<::melior::ir::Value<'c, '_>, ::melior::Error> {
        let attribute = ::melior::ir::attribute::DenseI32ArrayAttribute::<'c>::try_from(
            self.operation.attribute("operand_segment_sizes")?,
        )?;
        let start = (0..6usize)
            .map(|index| attribute.element(index))
            .collect::<Result<Vec<_>, _>>()?
            .into_iter()
            .sum::<i32>() as usize;
        let group_len = attribute.element(6usize)? as usize;
        if group_len == 0 {
            Err(::melior::Error::OperandNotFound("mask"))
        } else {
            self.operation.operand(start)
        }
    }
}
#[doc = "A builder for a [`firmem.read_write_port`](FirMemReadWriteOperation) operation."]
pub struct FirMemReadWriteOperationBuilder<'c, T0, O0, O1, O2, O3, O4> {
    builder: ::melior::ir::operation::OperationBuilder<'c>,
    context: &'c ::melior::Context,
    _state: ::std::marker::PhantomData<(T0, O0, O1, O2, O3, O4)>,
}
impl<'c>
    FirMemReadWriteOperationBuilder<
        'c,
        ::melior::dialect::ods::__private::Unset,
        ::melior::dialect::ods::__private::Unset,
        ::melior::dialect::ods::__private::Unset,
        ::melior::dialect::ods::__private::Unset,
        ::melior::dialect::ods::__private::Unset,
        ::melior::dialect::ods::__private::Unset,
    >
{
    pub fn new(context: &'c ::melior::Context, location: ::melior::ir::Location<'c>) -> Self {
        Self {
            context,
            builder: ::melior::ir::operation::OperationBuilder::new(
                "seq.firmem.read_write_port",
                location,
            ),
            _state: Default::default(),
        }
    }
}
impl<'c, O0, O1, O2, O3, O4>
    FirMemReadWriteOperationBuilder<
        'c,
        ::melior::dialect::ods::__private::Unset,
        O0,
        O1,
        O2,
        O3,
        O4,
    >
{
    pub fn read_data(
        self,
        read_data: ::melior::ir::Type<'c>,
    ) -> FirMemReadWriteOperationBuilder<
        'c,
        ::melior::dialect::ods::__private::Set,
        O0,
        O1,
        O2,
        O3,
        O4,
    > {
        FirMemReadWriteOperationBuilder {
            context: self.context,
            builder: self.builder.add_results(&[read_data]),
            _state: Default::default(),
        }
    }
}
impl<'c, T0, O1, O2, O3, O4>
    FirMemReadWriteOperationBuilder<
        'c,
        T0,
        ::melior::dialect::ods::__private::Unset,
        O1,
        O2,
        O3,
        O4,
    >
{
    pub fn memory(
        self,
        memory: ::melior::ir::Value<'c, '_>,
    ) -> FirMemReadWriteOperationBuilder<
        'c,
        T0,
        ::melior::dialect::ods::__private::Set,
        O1,
        O2,
        O3,
        O4,
    > {
        FirMemReadWriteOperationBuilder {
            context: self.context,
            builder: self.builder.add_operands(&[memory]),
            _state: Default::default(),
        }
    }
}
impl<'c, T0, O2, O3, O4>
    FirMemReadWriteOperationBuilder<
        'c,
        T0,
        ::melior::dialect::ods::__private::Set,
        ::melior::dialect::ods::__private::Unset,
        O2,
        O3,
        O4,
    >
{
    pub fn address(
        self,
        address: ::melior::ir::Value<'c, '_>,
    ) -> FirMemReadWriteOperationBuilder<
        'c,
        T0,
        ::melior::dialect::ods::__private::Set,
        ::melior::dialect::ods::__private::Set,
        O2,
        O3,
        O4,
    > {
        FirMemReadWriteOperationBuilder {
            context: self.context,
            builder: self.builder.add_operands(&[address]),
            _state: Default::default(),
        }
    }
}
impl<'c, T0, O3, O4>
    FirMemReadWriteOperationBuilder<
        'c,
        T0,
        ::melior::dialect::ods::__private::Set,
        ::melior::dialect::ods::__private::Set,
        ::melior::dialect::ods::__private::Unset,
        O3,
        O4,
    >
{
    pub fn clk(
        self,
        clk: ::melior::ir::Value<'c, '_>,
    ) -> FirMemReadWriteOperationBuilder<
        'c,
        T0,
        ::melior::dialect::ods::__private::Set,
        ::melior::dialect::ods::__private::Set,
        ::melior::dialect::ods::__private::Set,
        O3,
        O4,
    > {
        FirMemReadWriteOperationBuilder {
            context: self.context,
            builder: self.builder.add_operands(&[clk]),
            _state: Default::default(),
        }
    }
}
impl<'c, T0, O0, O1, O2, O3, O4> FirMemReadWriteOperationBuilder<'c, T0, O0, O1, O2, O3, O4> {
    pub fn enable(
        mut self,
        enable: ::melior::ir::Value<'c, '_>,
    ) -> FirMemReadWriteOperationBuilder<'c, T0, O0, O1, O2, O3, O4> {
        self.builder = self.builder.add_operands(&[enable]);
        self
    }
}
impl<'c, T0, O4>
    FirMemReadWriteOperationBuilder<
        'c,
        T0,
        ::melior::dialect::ods::__private::Set,
        ::melior::dialect::ods::__private::Set,
        ::melior::dialect::ods::__private::Set,
        ::melior::dialect::ods::__private::Unset,
        O4,
    >
{
    pub fn write_data(
        self,
        write_data: ::melior::ir::Value<'c, '_>,
    ) -> FirMemReadWriteOperationBuilder<
        'c,
        T0,
        ::melior::dialect::ods::__private::Set,
        ::melior::dialect::ods::__private::Set,
        ::melior::dialect::ods::__private::Set,
        ::melior::dialect::ods::__private::Set,
        O4,
    > {
        FirMemReadWriteOperationBuilder {
            context: self.context,
            builder: self.builder.add_operands(&[write_data]),
            _state: Default::default(),
        }
    }
}
impl<'c, T0>
    FirMemReadWriteOperationBuilder<
        'c,
        T0,
        ::melior::dialect::ods::__private::Set,
        ::melior::dialect::ods::__private::Set,
        ::melior::dialect::ods::__private::Set,
        ::melior::dialect::ods::__private::Set,
        ::melior::dialect::ods::__private::Unset,
    >
{
    pub fn mode(
        self,
        mode: ::melior::ir::Value<'c, '_>,
    ) -> FirMemReadWriteOperationBuilder<
        'c,
        T0,
        ::melior::dialect::ods::__private::Set,
        ::melior::dialect::ods::__private::Set,
        ::melior::dialect::ods::__private::Set,
        ::melior::dialect::ods::__private::Set,
        ::melior::dialect::ods::__private::Set,
    > {
        FirMemReadWriteOperationBuilder {
            context: self.context,
            builder: self.builder.add_operands(&[mode]),
            _state: Default::default(),
        }
    }
}
impl<'c, T0, O0, O1, O2, O3, O4> FirMemReadWriteOperationBuilder<'c, T0, O0, O1, O2, O3, O4> {
    pub fn mask(
        mut self,
        mask: ::melior::ir::Value<'c, '_>,
    ) -> FirMemReadWriteOperationBuilder<'c, T0, O0, O1, O2, O3, O4> {
        self.builder = self.builder.add_operands(&[mask]);
        self
    }
}
impl<'c>
    FirMemReadWriteOperationBuilder<
        'c,
        ::melior::dialect::ods::__private::Set,
        ::melior::dialect::ods::__private::Set,
        ::melior::dialect::ods::__private::Set,
        ::melior::dialect::ods::__private::Set,
        ::melior::dialect::ods::__private::Set,
        ::melior::dialect::ods::__private::Set,
    >
{
    pub fn build(self) -> FirMemReadWriteOperation<'c> {
        self.builder
            .build()
            .expect("valid operation")
            .try_into()
            .expect("should be a valid FirMemReadWriteOperation")
    }
}
#[allow(clippy::too_many_arguments)]
#[doc = "Creates a [`firmem.read_write_port`](FirMemReadWriteOperation) operation."]
pub fn firmem_read_write_port<'c>(
    context: &'c ::melior::Context,
    read_data: ::melior::ir::Type<'c>,
    memory: ::melior::ir::Value<'c, '_>,
    address: ::melior::ir::Value<'c, '_>,
    clk: ::melior::ir::Value<'c, '_>,
    write_data: ::melior::ir::Value<'c, '_>,
    mode: ::melior::ir::Value<'c, '_>,
    location: ::melior::ir::Location<'c>,
) -> FirMemReadWriteOperation<'c> {
    FirMemReadWriteOperation::builder(context, location)
        .read_data(read_data)
        .memory(memory)
        .address(address)
        .clk(clk)
        .write_data(write_data)
        .mode(mode)
        .build()
}
impl<'c> TryFrom<::melior::ir::operation::Operation<'c>> for FirMemReadWriteOperation<'c> {
    type Error = ::melior::Error;
    fn try_from(operation: ::melior::ir::operation::Operation<'c>) -> Result<Self, Self::Error> {
        Ok(Self { operation })
    }
}
impl<'c> From<FirMemReadWriteOperation<'c>> for ::melior::ir::operation::Operation<'c> {
    fn from(operation: FirMemReadWriteOperation<'c>) -> Self {
        operation.operation
    }
}
#[doc = "A [`firmem.write_port`](FirMemWriteOperation) operation. A memory write port."]
#[doc = "\n\n"]
#[doc = "The `seq.firmem.write_port` op represents a write port on a `seq.firmem`\nmemory. It takes the memory as an operand, together with the address and\ndata to be written, the clock on which the write is synchronized, an\noptional enable, and and optional write mask. Omitting the enable operand\nhas the same effect as passing a constant `true` to it. Omitting the write\nmask operand has the same effect as passing an all-ones value to it. A write\nmask operand can only be present if the `seq.firmem` specifies a mask width;\notherwise it must be omitted.\n"]
pub struct FirMemWriteOperation<'c> {
    operation: ::melior::ir::operation::Operation<'c>,
}
impl<'c> FirMemWriteOperation<'c> {
    #[doc = r" Returns a name."]
    pub fn name() -> &'static str {
        "seq.firmem.write_port"
    }
    #[doc = r" Returns a generic operation."]
    pub fn as_operation(&self) -> &::melior::ir::operation::Operation<'c> {
        &self.operation
    }
    #[doc = r" Creates a builder."]
    pub fn builder(
        context: &'c ::melior::Context,
        location: ::melior::ir::Location<'c>,
    ) -> FirMemWriteOperationBuilder<
        'c,
        ::melior::dialect::ods::__private::Unset,
        ::melior::dialect::ods::__private::Unset,
        ::melior::dialect::ods::__private::Unset,
        ::melior::dialect::ods::__private::Unset,
    > {
        FirMemWriteOperationBuilder::new(context, location)
    }
    pub fn memory(&self) -> Result<::melior::ir::Value<'c, '_>, ::melior::Error> {
        let attribute = ::melior::ir::attribute::DenseI32ArrayAttribute::<'c>::try_from(
            self.operation.attribute("operand_segment_sizes")?,
        )?;
        let start = (0..0usize)
            .map(|index| attribute.element(index))
            .collect::<Result<Vec<_>, _>>()?
            .into_iter()
            .sum::<i32>() as usize;
        let group_len = attribute.element(0usize)? as usize;
        self.operation.operand(start)
    }
    pub fn address(&self) -> Result<::melior::ir::Value<'c, '_>, ::melior::Error> {
        let attribute = ::melior::ir::attribute::DenseI32ArrayAttribute::<'c>::try_from(
            self.operation.attribute("operand_segment_sizes")?,
        )?;
        let start = (0..1usize)
            .map(|index| attribute.element(index))
            .collect::<Result<Vec<_>, _>>()?
            .into_iter()
            .sum::<i32>() as usize;
        let group_len = attribute.element(1usize)? as usize;
        self.operation.operand(start)
    }
    pub fn clk(&self) -> Result<::melior::ir::Value<'c, '_>, ::melior::Error> {
        let attribute = ::melior::ir::attribute::DenseI32ArrayAttribute::<'c>::try_from(
            self.operation.attribute("operand_segment_sizes")?,
        )?;
        let start = (0..2usize)
            .map(|index| attribute.element(index))
            .collect::<Result<Vec<_>, _>>()?
            .into_iter()
            .sum::<i32>() as usize;
        let group_len = attribute.element(2usize)? as usize;
        self.operation.operand(start)
    }
    pub fn enable(&self) -> Result<::melior::ir::Value<'c, '_>, ::melior::Error> {
        let attribute = ::melior::ir::attribute::DenseI32ArrayAttribute::<'c>::try_from(
            self.operation.attribute("operand_segment_sizes")?,
        )?;
        let start = (0..3usize)
            .map(|index| attribute.element(index))
            .collect::<Result<Vec<_>, _>>()?
            .into_iter()
            .sum::<i32>() as usize;
        let group_len = attribute.element(3usize)? as usize;
        if group_len == 0 {
            Err(::melior::Error::OperandNotFound("enable"))
        } else {
            self.operation.operand(start)
        }
    }
    pub fn data(&self) -> Result<::melior::ir::Value<'c, '_>, ::melior::Error> {
        let attribute = ::melior::ir::attribute::DenseI32ArrayAttribute::<'c>::try_from(
            self.operation.attribute("operand_segment_sizes")?,
        )?;
        let start = (0..4usize)
            .map(|index| attribute.element(index))
            .collect::<Result<Vec<_>, _>>()?
            .into_iter()
            .sum::<i32>() as usize;
        let group_len = attribute.element(4usize)? as usize;
        self.operation.operand(start)
    }
    pub fn mask(&self) -> Result<::melior::ir::Value<'c, '_>, ::melior::Error> {
        let attribute = ::melior::ir::attribute::DenseI32ArrayAttribute::<'c>::try_from(
            self.operation.attribute("operand_segment_sizes")?,
        )?;
        let start = (0..5usize)
            .map(|index| attribute.element(index))
            .collect::<Result<Vec<_>, _>>()?
            .into_iter()
            .sum::<i32>() as usize;
        let group_len = attribute.element(5usize)? as usize;
        if group_len == 0 {
            Err(::melior::Error::OperandNotFound("mask"))
        } else {
            self.operation.operand(start)
        }
    }
}
#[doc = "A builder for a [`firmem.write_port`](FirMemWriteOperation) operation."]
pub struct FirMemWriteOperationBuilder<'c, O0, O1, O2, O3> {
    builder: ::melior::ir::operation::OperationBuilder<'c>,
    context: &'c ::melior::Context,
    _state: ::std::marker::PhantomData<(O0, O1, O2, O3)>,
}
impl<'c>
    FirMemWriteOperationBuilder<
        'c,
        ::melior::dialect::ods::__private::Unset,
        ::melior::dialect::ods::__private::Unset,
        ::melior::dialect::ods::__private::Unset,
        ::melior::dialect::ods::__private::Unset,
    >
{
    pub fn new(context: &'c ::melior::Context, location: ::melior::ir::Location<'c>) -> Self {
        Self {
            context,
            builder: ::melior::ir::operation::OperationBuilder::new(
                "seq.firmem.write_port",
                location,
            ),
            _state: Default::default(),
        }
    }
}
impl<'c, O1, O2, O3>
    FirMemWriteOperationBuilder<'c, ::melior::dialect::ods::__private::Unset, O1, O2, O3>
{
    pub fn memory(
        self,
        memory: ::melior::ir::Value<'c, '_>,
    ) -> FirMemWriteOperationBuilder<'c, ::melior::dialect::ods::__private::Set, O1, O2, O3> {
        FirMemWriteOperationBuilder {
            context: self.context,
            builder: self.builder.add_operands(&[memory]),
            _state: Default::default(),
        }
    }
}
impl<'c, O2, O3>
    FirMemWriteOperationBuilder<
        'c,
        ::melior::dialect::ods::__private::Set,
        ::melior::dialect::ods::__private::Unset,
        O2,
        O3,
    >
{
    pub fn address(
        self,
        address: ::melior::ir::Value<'c, '_>,
    ) -> FirMemWriteOperationBuilder<
        'c,
        ::melior::dialect::ods::__private::Set,
        ::melior::dialect::ods::__private::Set,
        O2,
        O3,
    > {
        FirMemWriteOperationBuilder {
            context: self.context,
            builder: self.builder.add_operands(&[address]),
            _state: Default::default(),
        }
    }
}
impl<'c, O3>
    FirMemWriteOperationBuilder<
        'c,
        ::melior::dialect::ods::__private::Set,
        ::melior::dialect::ods::__private::Set,
        ::melior::dialect::ods::__private::Unset,
        O3,
    >
{
    pub fn clk(
        self,
        clk: ::melior::ir::Value<'c, '_>,
    ) -> FirMemWriteOperationBuilder<
        'c,
        ::melior::dialect::ods::__private::Set,
        ::melior::dialect::ods::__private::Set,
        ::melior::dialect::ods::__private::Set,
        O3,
    > {
        FirMemWriteOperationBuilder {
            context: self.context,
            builder: self.builder.add_operands(&[clk]),
            _state: Default::default(),
        }
    }
}
impl<'c, O0, O1, O2, O3> FirMemWriteOperationBuilder<'c, O0, O1, O2, O3> {
    pub fn enable(
        mut self,
        enable: ::melior::ir::Value<'c, '_>,
    ) -> FirMemWriteOperationBuilder<'c, O0, O1, O2, O3> {
        self.builder = self.builder.add_operands(&[enable]);
        self
    }
}
impl<'c>
    FirMemWriteOperationBuilder<
        'c,
        ::melior::dialect::ods::__private::Set,
        ::melior::dialect::ods::__private::Set,
        ::melior::dialect::ods::__private::Set,
        ::melior::dialect::ods::__private::Unset,
    >
{
    pub fn data(
        self,
        data: ::melior::ir::Value<'c, '_>,
    ) -> FirMemWriteOperationBuilder<
        'c,
        ::melior::dialect::ods::__private::Set,
        ::melior::dialect::ods::__private::Set,
        ::melior::dialect::ods::__private::Set,
        ::melior::dialect::ods::__private::Set,
    > {
        FirMemWriteOperationBuilder {
            context: self.context,
            builder: self.builder.add_operands(&[data]),
            _state: Default::default(),
        }
    }
}
impl<'c, O0, O1, O2, O3> FirMemWriteOperationBuilder<'c, O0, O1, O2, O3> {
    pub fn mask(
        mut self,
        mask: ::melior::ir::Value<'c, '_>,
    ) -> FirMemWriteOperationBuilder<'c, O0, O1, O2, O3> {
        self.builder = self.builder.add_operands(&[mask]);
        self
    }
}
impl<'c>
    FirMemWriteOperationBuilder<
        'c,
        ::melior::dialect::ods::__private::Set,
        ::melior::dialect::ods::__private::Set,
        ::melior::dialect::ods::__private::Set,
        ::melior::dialect::ods::__private::Set,
    >
{
    pub fn build(self) -> FirMemWriteOperation<'c> {
        self.builder
            .build()
            .expect("valid operation")
            .try_into()
            .expect("should be a valid FirMemWriteOperation")
    }
}
#[allow(clippy::too_many_arguments)]
#[doc = "Creates a [`firmem.write_port`](FirMemWriteOperation) operation."]
pub fn firmem_write_port<'c>(
    context: &'c ::melior::Context,
    memory: ::melior::ir::Value<'c, '_>,
    address: ::melior::ir::Value<'c, '_>,
    clk: ::melior::ir::Value<'c, '_>,
    data: ::melior::ir::Value<'c, '_>,
    location: ::melior::ir::Location<'c>,
) -> FirMemWriteOperation<'c> {
    FirMemWriteOperation::builder(context, location)
        .memory(memory)
        .address(address)
        .clk(clk)
        .data(data)
        .build()
}
impl<'c> TryFrom<::melior::ir::operation::Operation<'c>> for FirMemWriteOperation<'c> {
    type Error = ::melior::Error;
    fn try_from(operation: ::melior::ir::operation::Operation<'c>) -> Result<Self, Self::Error> {
        Ok(Self { operation })
    }
}
impl<'c> From<FirMemWriteOperation<'c>> for ::melior::ir::operation::Operation<'c> {
    fn from(operation: FirMemWriteOperation<'c>) -> Self {
        operation.operation
    }
}
#[doc = "A [`firreg`](FirRegOperation) operation. Register with preset and sync or async reset."]
#[doc = "\n\n"]
#[doc = "`firreg` represents registers originating from FIRRTL after the lowering\nof the IR to HW.  The register is used as an intermediary in the process\nof lowering to SystemVerilog to facilitate optimisation at the HW level,\ncompactly representing a register with a single operation instead of\ncomposing it from register definitions, always blocks and if statements.\n\nThe `data` output of the register accesses the value it stores.  On the\nrising edge of the `clk` input, the register takes a new value provided\nby the `next` signal.  Optionally, the register can also be provided with\na synchronous or an asynchronous `reset` signal and `resetValue`, as shown\nin the example below.\n\n``` text\n%name = seq.firreg %next clock %clk [ sym @sym ]\n    [ reset (sync|async) %reset, %value ]\n    [ preset value ] : type\n```\n\nImplicitly, all registers are pre-set to a randomized value.\n\nA register implementing a counter starting at 0 from reset can be defined\nas follows:\n\n``` text\n%zero = hw.constant 0 : i32\n%reg = seq.firreg %next clock %clk reset sync %reset, %zero : i32\n%one = hw.constant 1 : i32\n%next = comb.add %reg, %one : i32\n```\n"]
pub struct FirRegOperation<'c> {
    operation: ::melior::ir::operation::Operation<'c>,
}
impl<'c> FirRegOperation<'c> {
    #[doc = r" Returns a name."]
    pub fn name() -> &'static str {
        "seq.firreg"
    }
    #[doc = r" Returns a generic operation."]
    pub fn as_operation(&self) -> &::melior::ir::operation::Operation<'c> {
        &self.operation
    }
    #[doc = r" Creates a builder."]
    pub fn builder(
        context: &'c ::melior::Context,
        location: ::melior::ir::Location<'c>,
    ) -> FirRegOperationBuilder<
        'c,
        ::melior::dialect::ods::__private::Unset,
        ::melior::dialect::ods::__private::Unset,
        ::melior::dialect::ods::__private::Unset,
        ::melior::dialect::ods::__private::Unset,
    > {
        FirRegOperationBuilder::new(context, location)
    }
    pub fn data(
        &self,
    ) -> Result<::melior::ir::operation::OperationResult<'c, '_>, ::melior::Error> {
        self.operation.result(0usize)
    }
    pub fn next(&self) -> Result<::melior::ir::Value<'c, '_>, ::melior::Error> {
        let total_var_len = self.operation.operand_count() - 2usize + 1;
        let group_len = total_var_len / 2usize;
        let start = 0usize + 0usize * group_len;
        self.operation.operand(start)
    }
    pub fn clk(&self) -> Result<::melior::ir::Value<'c, '_>, ::melior::Error> {
        let total_var_len = self.operation.operand_count() - 2usize + 1;
        let group_len = total_var_len / 2usize;
        let start = 1usize + 0usize * group_len;
        self.operation.operand(start)
    }
    pub fn reset(&self) -> Result<::melior::ir::Value<'c, '_>, ::melior::Error> {
        let total_var_len = self.operation.operand_count() - 2usize + 1;
        let group_len = total_var_len / 2usize;
        let start = 2usize + 0usize * group_len;
        self.operation.operand(start)
    }
    pub fn reset_value(&self) -> Result<::melior::ir::Value<'c, '_>, ::melior::Error> {
        let total_var_len = self.operation.operand_count() - 2usize + 1;
        let group_len = total_var_len / 2usize;
        let start = 2usize + 1usize * group_len;
        self.operation.operand(start)
    }
    #[allow(clippy::needless_question_mark)]
    pub fn _name(&self) -> Result<::melior::ir::attribute::StringAttribute<'c>, ::melior::Error> {
        Ok(self.operation.attribute("name")?.try_into()?)
    }
    pub fn set_name(&mut self, value: ::melior::ir::attribute::StringAttribute<'c>) {
        self.operation.set_attribute("name", value.into());
    }
    #[allow(clippy::needless_question_mark)]
    pub fn inner_sym(&self) -> Result<::melior::ir::attribute::Attribute<'c>, ::melior::Error> {
        Ok(self.operation.attribute("inner_sym")?.try_into()?)
    }
    pub fn set_inner_sym(&mut self, value: ::melior::ir::attribute::Attribute<'c>) {
        self.operation.set_attribute("inner_sym", value.into());
    }
    pub fn remove_inner_sym(&mut self) -> Result<(), ::melior::Error> {
        self.operation.remove_attribute("inner_sym")
    }
    #[allow(clippy::needless_question_mark)]
    pub fn preset(&self) -> Result<::melior::ir::attribute::IntegerAttribute<'c>, ::melior::Error> {
        Ok(self.operation.attribute("preset")?.try_into()?)
    }
    pub fn set_preset(&mut self, value: ::melior::ir::attribute::IntegerAttribute<'c>) {
        self.operation.set_attribute("preset", value.into());
    }
    pub fn remove_preset(&mut self) -> Result<(), ::melior::Error> {
        self.operation.remove_attribute("preset")
    }
    #[allow(clippy::needless_question_mark)]
    pub fn is_async(&self) -> Result<::melior::ir::attribute::Attribute<'c>, ::melior::Error> {
        Ok(self.operation.attribute("isAsync")?.try_into()?)
    }
    pub fn set_is_async(&mut self, value: ::melior::ir::attribute::Attribute<'c>) {
        self.operation.set_attribute("isAsync", value.into());
    }
    pub fn remove_is_async(&mut self) -> Result<(), ::melior::Error> {
        self.operation.remove_attribute("isAsync")
    }
}
#[doc = "A builder for a [`firreg`](FirRegOperation) operation."]
pub struct FirRegOperationBuilder<'c, T0, O0, O1, A0> {
    builder: ::melior::ir::operation::OperationBuilder<'c>,
    context: &'c ::melior::Context,
    _state: ::std::marker::PhantomData<(T0, O0, O1, A0)>,
}
impl<'c>
    FirRegOperationBuilder<
        'c,
        ::melior::dialect::ods::__private::Unset,
        ::melior::dialect::ods::__private::Unset,
        ::melior::dialect::ods::__private::Unset,
        ::melior::dialect::ods::__private::Unset,
    >
{
    pub fn new(context: &'c ::melior::Context, location: ::melior::ir::Location<'c>) -> Self {
        Self {
            context,
            builder: ::melior::ir::operation::OperationBuilder::new("seq.firreg", location),
            _state: Default::default(),
        }
    }
}
impl<'c, O0, O1, A0>
    FirRegOperationBuilder<'c, ::melior::dialect::ods::__private::Unset, O0, O1, A0>
{
    pub fn data(
        self,
        data: ::melior::ir::Type<'c>,
    ) -> FirRegOperationBuilder<'c, ::melior::dialect::ods::__private::Set, O0, O1, A0> {
        FirRegOperationBuilder {
            context: self.context,
            builder: self.builder.add_results(&[data]),
            _state: Default::default(),
        }
    }
}
impl<'c, T0, O1, A0>
    FirRegOperationBuilder<'c, T0, ::melior::dialect::ods::__private::Unset, O1, A0>
{
    pub fn next(
        self,
        next: ::melior::ir::Value<'c, '_>,
    ) -> FirRegOperationBuilder<'c, T0, ::melior::dialect::ods::__private::Set, O1, A0> {
        FirRegOperationBuilder {
            context: self.context,
            builder: self.builder.add_operands(&[next]),
            _state: Default::default(),
        }
    }
}
impl<'c, T0, A0>
    FirRegOperationBuilder<
        'c,
        T0,
        ::melior::dialect::ods::__private::Set,
        ::melior::dialect::ods::__private::Unset,
        A0,
    >
{
    pub fn clk(
        self,
        clk: ::melior::ir::Value<'c, '_>,
    ) -> FirRegOperationBuilder<
        'c,
        T0,
        ::melior::dialect::ods::__private::Set,
        ::melior::dialect::ods::__private::Set,
        A0,
    > {
        FirRegOperationBuilder {
            context: self.context,
            builder: self.builder.add_operands(&[clk]),
            _state: Default::default(),
        }
    }
}
impl<'c, T0, O0, O1, A0> FirRegOperationBuilder<'c, T0, O0, O1, A0> {
    pub fn reset(
        mut self,
        reset: ::melior::ir::Value<'c, '_>,
    ) -> FirRegOperationBuilder<'c, T0, O0, O1, A0> {
        self.builder = self.builder.add_operands(&[reset]);
        self
    }
}
impl<'c, T0, O0, O1, A0> FirRegOperationBuilder<'c, T0, O0, O1, A0> {
    pub fn reset_value(
        mut self,
        reset_value: ::melior::ir::Value<'c, '_>,
    ) -> FirRegOperationBuilder<'c, T0, O0, O1, A0> {
        self.builder = self.builder.add_operands(&[reset_value]);
        self
    }
}
impl<'c, T0, O0, O1>
    FirRegOperationBuilder<'c, T0, O0, O1, ::melior::dialect::ods::__private::Unset>
{
    pub fn _name(
        self,
        _name: ::melior::ir::attribute::StringAttribute<'c>,
    ) -> FirRegOperationBuilder<'c, T0, O0, O1, ::melior::dialect::ods::__private::Set> {
        FirRegOperationBuilder {
            context: self.context,
            builder: self.builder.add_attributes(&[(
                ::melior::ir::Identifier::new(self.context, "name"),
                _name.into(),
            )]),
            _state: Default::default(),
        }
    }
}
impl<'c, T0, O0, O1, A0> FirRegOperationBuilder<'c, T0, O0, O1, A0> {
    pub fn inner_sym(
        mut self,
        inner_sym: ::melior::ir::attribute::Attribute<'c>,
    ) -> FirRegOperationBuilder<'c, T0, O0, O1, A0> {
        self.builder = self.builder.add_attributes(&[(
            ::melior::ir::Identifier::new(self.context, "inner_sym"),
            inner_sym.into(),
        )]);
        self
    }
}
impl<'c, T0, O0, O1, A0> FirRegOperationBuilder<'c, T0, O0, O1, A0> {
    pub fn preset(
        mut self,
        preset: ::melior::ir::attribute::IntegerAttribute<'c>,
    ) -> FirRegOperationBuilder<'c, T0, O0, O1, A0> {
        self.builder = self.builder.add_attributes(&[(
            ::melior::ir::Identifier::new(self.context, "preset"),
            preset.into(),
        )]);
        self
    }
}
impl<'c, T0, O0, O1, A0> FirRegOperationBuilder<'c, T0, O0, O1, A0> {
    pub fn is_async(
        mut self,
        is_async: ::melior::ir::attribute::Attribute<'c>,
    ) -> FirRegOperationBuilder<'c, T0, O0, O1, A0> {
        self.builder = self.builder.add_attributes(&[(
            ::melior::ir::Identifier::new(self.context, "isAsync"),
            is_async.into(),
        )]);
        self
    }
}
impl<'c>
    FirRegOperationBuilder<
        'c,
        ::melior::dialect::ods::__private::Set,
        ::melior::dialect::ods::__private::Set,
        ::melior::dialect::ods::__private::Set,
        ::melior::dialect::ods::__private::Set,
    >
{
    pub fn build(self) -> FirRegOperation<'c> {
        self.builder
            .build()
            .expect("valid operation")
            .try_into()
            .expect("should be a valid FirRegOperation")
    }
}
#[allow(clippy::too_many_arguments)]
#[doc = "Creates a [`firreg`](FirRegOperation) operation."]
pub fn firreg<'c>(
    context: &'c ::melior::Context,
    data: ::melior::ir::Type<'c>,
    next: ::melior::ir::Value<'c, '_>,
    clk: ::melior::ir::Value<'c, '_>,
    _name: ::melior::ir::attribute::StringAttribute<'c>,
    location: ::melior::ir::Location<'c>,
) -> FirRegOperation<'c> {
    FirRegOperation::builder(context, location)
        .data(data)
        .next(next)
        .clk(clk)
        ._name(_name)
        .build()
}
impl<'c> TryFrom<::melior::ir::operation::Operation<'c>> for FirRegOperation<'c> {
    type Error = ::melior::Error;
    fn try_from(operation: ::melior::ir::operation::Operation<'c>) -> Result<Self, Self::Error> {
        Ok(Self { operation })
    }
}
impl<'c> From<FirRegOperation<'c>> for ::melior::ir::operation::Operation<'c> {
    fn from(operation: FirRegOperation<'c>) -> Self {
        operation.operation
    }
}
#[doc = "A [`from_clock`](FromClockOperation) operation. Cast from a clock type to a wire type."]
#[doc = "\n\n"]
#[doc = ""]
pub struct FromClockOperation<'c> {
    operation: ::melior::ir::operation::Operation<'c>,
}
impl<'c> FromClockOperation<'c> {
    #[doc = r" Returns a name."]
    pub fn name() -> &'static str {
        "seq.from_clock"
    }
    #[doc = r" Returns a generic operation."]
    pub fn as_operation(&self) -> &::melior::ir::operation::Operation<'c> {
        &self.operation
    }
    #[doc = r" Creates a builder."]
    pub fn builder(
        context: &'c ::melior::Context,
        location: ::melior::ir::Location<'c>,
    ) -> FromClockOperationBuilder<
        'c,
        ::melior::dialect::ods::__private::Unset,
        ::melior::dialect::ods::__private::Unset,
    > {
        FromClockOperationBuilder::new(context, location)
    }
    pub fn output(
        &self,
    ) -> Result<::melior::ir::operation::OperationResult<'c, '_>, ::melior::Error> {
        self.operation.result(0usize)
    }
    pub fn input(&self) -> Result<::melior::ir::Value<'c, '_>, ::melior::Error> {
        self.operation.operand(0usize)
    }
}
#[doc = "A builder for a [`from_clock`](FromClockOperation) operation."]
pub struct FromClockOperationBuilder<'c, T0, O0> {
    builder: ::melior::ir::operation::OperationBuilder<'c>,
    context: &'c ::melior::Context,
    _state: ::std::marker::PhantomData<(T0, O0)>,
}
impl<'c>
    FromClockOperationBuilder<
        'c,
        ::melior::dialect::ods::__private::Unset,
        ::melior::dialect::ods::__private::Unset,
    >
{
    pub fn new(context: &'c ::melior::Context, location: ::melior::ir::Location<'c>) -> Self {
        Self {
            context,
            builder: ::melior::ir::operation::OperationBuilder::new("seq.from_clock", location),
            _state: Default::default(),
        }
    }
}
impl<'c, O0> FromClockOperationBuilder<'c, ::melior::dialect::ods::__private::Unset, O0> {
    pub fn output(
        self,
        output: ::melior::ir::Type<'c>,
    ) -> FromClockOperationBuilder<'c, ::melior::dialect::ods::__private::Set, O0> {
        FromClockOperationBuilder {
            context: self.context,
            builder: self.builder.add_results(&[output]),
            _state: Default::default(),
        }
    }
}
impl<'c, T0> FromClockOperationBuilder<'c, T0, ::melior::dialect::ods::__private::Unset> {
    pub fn input(
        self,
        input: ::melior::ir::Value<'c, '_>,
    ) -> FromClockOperationBuilder<'c, T0, ::melior::dialect::ods::__private::Set> {
        FromClockOperationBuilder {
            context: self.context,
            builder: self.builder.add_operands(&[input]),
            _state: Default::default(),
        }
    }
}
impl<'c>
    FromClockOperationBuilder<
        'c,
        ::melior::dialect::ods::__private::Set,
        ::melior::dialect::ods::__private::Set,
    >
{
    pub fn build(self) -> FromClockOperation<'c> {
        self.builder
            .build()
            .expect("valid operation")
            .try_into()
            .expect("should be a valid FromClockOperation")
    }
}
#[allow(clippy::too_many_arguments)]
#[doc = "Creates a [`from_clock`](FromClockOperation) operation."]
pub fn from_clock<'c>(
    context: &'c ::melior::Context,
    output: ::melior::ir::Type<'c>,
    input: ::melior::ir::Value<'c, '_>,
    location: ::melior::ir::Location<'c>,
) -> FromClockOperation<'c> {
    FromClockOperation::builder(context, location)
        .output(output)
        .input(input)
        .build()
}
impl<'c> TryFrom<::melior::ir::operation::Operation<'c>> for FromClockOperation<'c> {
    type Error = ::melior::Error;
    fn try_from(operation: ::melior::ir::operation::Operation<'c>) -> Result<Self, Self::Error> {
        Ok(Self { operation })
    }
}
impl<'c> From<FromClockOperation<'c>> for ::melior::ir::operation::Operation<'c> {
    fn from(operation: FromClockOperation<'c>) -> Self {
        operation.operation
    }
}
#[doc = "A [`hlmem`](HLMemOperation) operation. Instantiate a high-level memory.."]
#[doc = "\n\n"]
#[doc = "See the Seq dialect rationale for a longer description\n"]
pub struct HLMemOperation<'c> {
    operation: ::melior::ir::operation::Operation<'c>,
}
impl<'c> HLMemOperation<'c> {
    #[doc = r" Returns a name."]
    pub fn name() -> &'static str {
        "seq.hlmem"
    }
    #[doc = r" Returns a generic operation."]
    pub fn as_operation(&self) -> &::melior::ir::operation::Operation<'c> {
        &self.operation
    }
    #[doc = r" Creates a builder."]
    pub fn builder(
        context: &'c ::melior::Context,
        location: ::melior::ir::Location<'c>,
    ) -> HLMemOperationBuilder<
        'c,
        ::melior::dialect::ods::__private::Unset,
        ::melior::dialect::ods::__private::Unset,
        ::melior::dialect::ods::__private::Unset,
        ::melior::dialect::ods::__private::Unset,
    > {
        HLMemOperationBuilder::new(context, location)
    }
    pub fn handle(
        &self,
    ) -> Result<::melior::ir::operation::OperationResult<'c, '_>, ::melior::Error> {
        self.operation.result(0usize)
    }
    pub fn clk(&self) -> Result<::melior::ir::Value<'c, '_>, ::melior::Error> {
        self.operation.operand(0usize)
    }
    pub fn rst(&self) -> Result<::melior::ir::Value<'c, '_>, ::melior::Error> {
        self.operation.operand(1usize)
    }
    #[allow(clippy::needless_question_mark)]
    pub fn _name(&self) -> Result<::melior::ir::attribute::StringAttribute<'c>, ::melior::Error> {
        Ok(self.operation.attribute("name")?.try_into()?)
    }
    pub fn set_name(&mut self, value: ::melior::ir::attribute::StringAttribute<'c>) {
        self.operation.set_attribute("name", value.into());
    }
}
#[doc = "A builder for a [`hlmem`](HLMemOperation) operation."]
pub struct HLMemOperationBuilder<'c, T0, O0, O1, A0> {
    builder: ::melior::ir::operation::OperationBuilder<'c>,
    context: &'c ::melior::Context,
    _state: ::std::marker::PhantomData<(T0, O0, O1, A0)>,
}
impl<'c>
    HLMemOperationBuilder<
        'c,
        ::melior::dialect::ods::__private::Unset,
        ::melior::dialect::ods::__private::Unset,
        ::melior::dialect::ods::__private::Unset,
        ::melior::dialect::ods::__private::Unset,
    >
{
    pub fn new(context: &'c ::melior::Context, location: ::melior::ir::Location<'c>) -> Self {
        Self {
            context,
            builder: ::melior::ir::operation::OperationBuilder::new("seq.hlmem", location),
            _state: Default::default(),
        }
    }
}
impl<'c, O0, O1, A0>
    HLMemOperationBuilder<'c, ::melior::dialect::ods::__private::Unset, O0, O1, A0>
{
    pub fn handle(
        self,
        handle: ::melior::ir::Type<'c>,
    ) -> HLMemOperationBuilder<'c, ::melior::dialect::ods::__private::Set, O0, O1, A0> {
        HLMemOperationBuilder {
            context: self.context,
            builder: self.builder.add_results(&[handle]),
            _state: Default::default(),
        }
    }
}
impl<'c, T0, O1, A0>
    HLMemOperationBuilder<'c, T0, ::melior::dialect::ods::__private::Unset, O1, A0>
{
    pub fn clk(
        self,
        clk: ::melior::ir::Value<'c, '_>,
    ) -> HLMemOperationBuilder<'c, T0, ::melior::dialect::ods::__private::Set, O1, A0> {
        HLMemOperationBuilder {
            context: self.context,
            builder: self.builder.add_operands(&[clk]),
            _state: Default::default(),
        }
    }
}
impl<'c, T0, A0>
    HLMemOperationBuilder<
        'c,
        T0,
        ::melior::dialect::ods::__private::Set,
        ::melior::dialect::ods::__private::Unset,
        A0,
    >
{
    pub fn rst(
        self,
        rst: ::melior::ir::Value<'c, '_>,
    ) -> HLMemOperationBuilder<
        'c,
        T0,
        ::melior::dialect::ods::__private::Set,
        ::melior::dialect::ods::__private::Set,
        A0,
    > {
        HLMemOperationBuilder {
            context: self.context,
            builder: self.builder.add_operands(&[rst]),
            _state: Default::default(),
        }
    }
}
impl<'c, T0, O0, O1>
    HLMemOperationBuilder<'c, T0, O0, O1, ::melior::dialect::ods::__private::Unset>
{
    pub fn _name(
        self,
        _name: ::melior::ir::attribute::StringAttribute<'c>,
    ) -> HLMemOperationBuilder<'c, T0, O0, O1, ::melior::dialect::ods::__private::Set> {
        HLMemOperationBuilder {
            context: self.context,
            builder: self.builder.add_attributes(&[(
                ::melior::ir::Identifier::new(self.context, "name"),
                _name.into(),
            )]),
            _state: Default::default(),
        }
    }
}
impl<'c>
    HLMemOperationBuilder<
        'c,
        ::melior::dialect::ods::__private::Set,
        ::melior::dialect::ods::__private::Set,
        ::melior::dialect::ods::__private::Set,
        ::melior::dialect::ods::__private::Set,
    >
{
    pub fn build(self) -> HLMemOperation<'c> {
        self.builder
            .build()
            .expect("valid operation")
            .try_into()
            .expect("should be a valid HLMemOperation")
    }
}
#[allow(clippy::too_many_arguments)]
#[doc = "Creates a [`hlmem`](HLMemOperation) operation."]
pub fn hlmem<'c>(
    context: &'c ::melior::Context,
    handle: ::melior::ir::Type<'c>,
    clk: ::melior::ir::Value<'c, '_>,
    rst: ::melior::ir::Value<'c, '_>,
    _name: ::melior::ir::attribute::StringAttribute<'c>,
    location: ::melior::ir::Location<'c>,
) -> HLMemOperation<'c> {
    HLMemOperation::builder(context, location)
        .handle(handle)
        .clk(clk)
        .rst(rst)
        ._name(_name)
        .build()
}
impl<'c> TryFrom<::melior::ir::operation::Operation<'c>> for HLMemOperation<'c> {
    type Error = ::melior::Error;
    fn try_from(operation: ::melior::ir::operation::Operation<'c>) -> Result<Self, Self::Error> {
        Ok(Self { operation })
    }
}
impl<'c> From<HLMemOperation<'c>> for ::melior::ir::operation::Operation<'c> {
    fn from(operation: HLMemOperation<'c>) -> Self {
        operation.operation
    }
}
#[doc = "A [`read`](ReadPortOperation) operation. Structural read access to a seq.hlmem, with an optional read enable signal.."]
#[doc = "\n\n"]
#[doc = ""]
pub struct ReadPortOperation<'c> {
    operation: ::melior::ir::operation::Operation<'c>,
}
impl<'c> ReadPortOperation<'c> {
    #[doc = r" Returns a name."]
    pub fn name() -> &'static str {
        "seq.read"
    }
    #[doc = r" Returns a generic operation."]
    pub fn as_operation(&self) -> &::melior::ir::operation::Operation<'c> {
        &self.operation
    }
    #[doc = r" Creates a builder."]
    pub fn builder(
        context: &'c ::melior::Context,
        location: ::melior::ir::Location<'c>,
    ) -> ReadPortOperationBuilder<
        'c,
        ::melior::dialect::ods::__private::Unset,
        ::melior::dialect::ods::__private::Unset,
        ::melior::dialect::ods::__private::Unset,
        ::melior::dialect::ods::__private::Unset,
    > {
        ReadPortOperationBuilder::new(context, location)
    }
    pub fn read_data(
        &self,
    ) -> Result<::melior::ir::operation::OperationResult<'c, '_>, ::melior::Error> {
        self.operation.result(0usize)
    }
    pub fn memory(&self) -> Result<::melior::ir::Value<'c, '_>, ::melior::Error> {
        let attribute = ::melior::ir::attribute::DenseI32ArrayAttribute::<'c>::try_from(
            self.operation.attribute("operand_segment_sizes")?,
        )?;
        let start = (0..0usize)
            .map(|index| attribute.element(index))
            .collect::<Result<Vec<_>, _>>()?
            .into_iter()
            .sum::<i32>() as usize;
        let group_len = attribute.element(0usize)? as usize;
        self.operation.operand(start)
    }
    pub fn addresses(
        &self,
    ) -> Result<impl Iterator<Item = ::melior::ir::Value<'c, '_>>, ::melior::Error> {
        let attribute = ::melior::ir::attribute::DenseI32ArrayAttribute::<'c>::try_from(
            self.operation.attribute("operand_segment_sizes")?,
        )?;
        let start = (0..1usize)
            .map(|index| attribute.element(index))
            .collect::<Result<Vec<_>, _>>()?
            .into_iter()
            .sum::<i32>() as usize;
        let group_len = attribute.element(1usize)? as usize;
        Ok(self.operation.operands().skip(start).take(group_len))
    }
    pub fn rd_en(&self) -> Result<::melior::ir::Value<'c, '_>, ::melior::Error> {
        let attribute = ::melior::ir::attribute::DenseI32ArrayAttribute::<'c>::try_from(
            self.operation.attribute("operand_segment_sizes")?,
        )?;
        let start = (0..2usize)
            .map(|index| attribute.element(index))
            .collect::<Result<Vec<_>, _>>()?
            .into_iter()
            .sum::<i32>() as usize;
        let group_len = attribute.element(2usize)? as usize;
        if group_len == 0 {
            Err(::melior::Error::OperandNotFound("rdEn"))
        } else {
            self.operation.operand(start)
        }
    }
    #[allow(clippy::needless_question_mark)]
    pub fn latency(
        &self,
    ) -> Result<::melior::ir::attribute::IntegerAttribute<'c>, ::melior::Error> {
        Ok(self.operation.attribute("latency")?.try_into()?)
    }
    pub fn set_latency(&mut self, value: ::melior::ir::attribute::IntegerAttribute<'c>) {
        self.operation.set_attribute("latency", value.into());
    }
}
#[doc = "A builder for a [`read`](ReadPortOperation) operation."]
pub struct ReadPortOperationBuilder<'c, T0, O0, O1, A0> {
    builder: ::melior::ir::operation::OperationBuilder<'c>,
    context: &'c ::melior::Context,
    _state: ::std::marker::PhantomData<(T0, O0, O1, A0)>,
}
impl<'c>
    ReadPortOperationBuilder<
        'c,
        ::melior::dialect::ods::__private::Unset,
        ::melior::dialect::ods::__private::Unset,
        ::melior::dialect::ods::__private::Unset,
        ::melior::dialect::ods::__private::Unset,
    >
{
    pub fn new(context: &'c ::melior::Context, location: ::melior::ir::Location<'c>) -> Self {
        Self {
            context,
            builder: ::melior::ir::operation::OperationBuilder::new("seq.read", location),
            _state: Default::default(),
        }
    }
}
impl<'c, O0, O1, A0>
    ReadPortOperationBuilder<'c, ::melior::dialect::ods::__private::Unset, O0, O1, A0>
{
    pub fn read_data(
        self,
        read_data: ::melior::ir::Type<'c>,
    ) -> ReadPortOperationBuilder<'c, ::melior::dialect::ods::__private::Set, O0, O1, A0> {
        ReadPortOperationBuilder {
            context: self.context,
            builder: self.builder.add_results(&[read_data]),
            _state: Default::default(),
        }
    }
}
impl<'c, T0, O1, A0>
    ReadPortOperationBuilder<'c, T0, ::melior::dialect::ods::__private::Unset, O1, A0>
{
    pub fn memory(
        self,
        memory: ::melior::ir::Value<'c, '_>,
    ) -> ReadPortOperationBuilder<'c, T0, ::melior::dialect::ods::__private::Set, O1, A0> {
        ReadPortOperationBuilder {
            context: self.context,
            builder: self.builder.add_operands(&[memory]),
            _state: Default::default(),
        }
    }
}
impl<'c, T0, A0>
    ReadPortOperationBuilder<
        'c,
        T0,
        ::melior::dialect::ods::__private::Set,
        ::melior::dialect::ods::__private::Unset,
        A0,
    >
{
    pub fn addresses(
        self,
        addresses: &[::melior::ir::Value<'c, '_>],
    ) -> ReadPortOperationBuilder<
        'c,
        T0,
        ::melior::dialect::ods::__private::Set,
        ::melior::dialect::ods::__private::Set,
        A0,
    > {
        ReadPortOperationBuilder {
            context: self.context,
            builder: self.builder.add_operands(addresses),
            _state: Default::default(),
        }
    }
}
impl<'c, T0, O0, O1, A0> ReadPortOperationBuilder<'c, T0, O0, O1, A0> {
    pub fn rd_en(
        mut self,
        rd_en: ::melior::ir::Value<'c, '_>,
    ) -> ReadPortOperationBuilder<'c, T0, O0, O1, A0> {
        self.builder = self.builder.add_operands(&[rd_en]);
        self
    }
}
impl<'c, T0, O0, O1>
    ReadPortOperationBuilder<'c, T0, O0, O1, ::melior::dialect::ods::__private::Unset>
{
    pub fn latency(
        self,
        latency: ::melior::ir::attribute::IntegerAttribute<'c>,
    ) -> ReadPortOperationBuilder<'c, T0, O0, O1, ::melior::dialect::ods::__private::Set> {
        ReadPortOperationBuilder {
            context: self.context,
            builder: self.builder.add_attributes(&[(
                ::melior::ir::Identifier::new(self.context, "latency"),
                latency.into(),
            )]),
            _state: Default::default(),
        }
    }
}
impl<'c>
    ReadPortOperationBuilder<
        'c,
        ::melior::dialect::ods::__private::Set,
        ::melior::dialect::ods::__private::Set,
        ::melior::dialect::ods::__private::Set,
        ::melior::dialect::ods::__private::Set,
    >
{
    pub fn build(self) -> ReadPortOperation<'c> {
        self.builder
            .build()
            .expect("valid operation")
            .try_into()
            .expect("should be a valid ReadPortOperation")
    }
}
#[allow(clippy::too_many_arguments)]
#[doc = "Creates a [`read`](ReadPortOperation) operation."]
pub fn read<'c>(
    context: &'c ::melior::Context,
    read_data: ::melior::ir::Type<'c>,
    memory: ::melior::ir::Value<'c, '_>,
    addresses: &[::melior::ir::Value<'c, '_>],
    latency: ::melior::ir::attribute::IntegerAttribute<'c>,
    location: ::melior::ir::Location<'c>,
) -> ReadPortOperation<'c> {
    ReadPortOperation::builder(context, location)
        .read_data(read_data)
        .memory(memory)
        .addresses(addresses)
        .latency(latency)
        .build()
}
impl<'c> TryFrom<::melior::ir::operation::Operation<'c>> for ReadPortOperation<'c> {
    type Error = ::melior::Error;
    fn try_from(operation: ::melior::ir::operation::Operation<'c>) -> Result<Self, Self::Error> {
        Ok(Self { operation })
    }
}
impl<'c> From<ReadPortOperation<'c>> for ::melior::ir::operation::Operation<'c> {
    fn from(operation: ReadPortOperation<'c>) -> Self {
        operation.operation
    }
}
#[doc = "A [`shiftreg`](ShiftRegOperation) operation. Shift register."]
#[doc = "\n\n"]
#[doc = "The `seq.shiftreg` op represents a shift register. It takes the input\nvalue and shifts it every cycle when `clockEnable` is asserted.\nThe `reset` and `resetValue` operands are optional and if present, every\nentry in the shift register will be initialized to `resetValue` upon\nassertion of the reset signal. Exact reset behavior (sync/async) is\nimplementation defined.\n"]
pub struct ShiftRegOperation<'c> {
    operation: ::melior::ir::operation::Operation<'c>,
}
impl<'c> ShiftRegOperation<'c> {
    #[doc = r" Returns a name."]
    pub fn name() -> &'static str {
        "seq.shiftreg"
    }
    #[doc = r" Returns a generic operation."]
    pub fn as_operation(&self) -> &::melior::ir::operation::Operation<'c> {
        &self.operation
    }
    #[doc = r" Creates a builder."]
    pub fn builder(
        context: &'c ::melior::Context,
        location: ::melior::ir::Location<'c>,
    ) -> ShiftRegOperationBuilder<
        'c,
        ::melior::dialect::ods::__private::Unset,
        ::melior::dialect::ods::__private::Unset,
        ::melior::dialect::ods::__private::Unset,
        ::melior::dialect::ods::__private::Unset,
        ::melior::dialect::ods::__private::Unset,
    > {
        ShiftRegOperationBuilder::new(context, location)
    }
    pub fn data(
        &self,
    ) -> Result<::melior::ir::operation::OperationResult<'c, '_>, ::melior::Error> {
        self.operation.result(0usize)
    }
    pub fn input(&self) -> Result<::melior::ir::Value<'c, '_>, ::melior::Error> {
        let attribute = ::melior::ir::attribute::DenseI32ArrayAttribute::<'c>::try_from(
            self.operation.attribute("operand_segment_sizes")?,
        )?;
        let start = (0..0usize)
            .map(|index| attribute.element(index))
            .collect::<Result<Vec<_>, _>>()?
            .into_iter()
            .sum::<i32>() as usize;
        let group_len = attribute.element(0usize)? as usize;
        self.operation.operand(start)
    }
    pub fn clk(&self) -> Result<::melior::ir::Value<'c, '_>, ::melior::Error> {
        let attribute = ::melior::ir::attribute::DenseI32ArrayAttribute::<'c>::try_from(
            self.operation.attribute("operand_segment_sizes")?,
        )?;
        let start = (0..1usize)
            .map(|index| attribute.element(index))
            .collect::<Result<Vec<_>, _>>()?
            .into_iter()
            .sum::<i32>() as usize;
        let group_len = attribute.element(1usize)? as usize;
        self.operation.operand(start)
    }
    pub fn clock_enable(&self) -> Result<::melior::ir::Value<'c, '_>, ::melior::Error> {
        let attribute = ::melior::ir::attribute::DenseI32ArrayAttribute::<'c>::try_from(
            self.operation.attribute("operand_segment_sizes")?,
        )?;
        let start = (0..2usize)
            .map(|index| attribute.element(index))
            .collect::<Result<Vec<_>, _>>()?
            .into_iter()
            .sum::<i32>() as usize;
        let group_len = attribute.element(2usize)? as usize;
        self.operation.operand(start)
    }
    pub fn reset(&self) -> Result<::melior::ir::Value<'c, '_>, ::melior::Error> {
        let attribute = ::melior::ir::attribute::DenseI32ArrayAttribute::<'c>::try_from(
            self.operation.attribute("operand_segment_sizes")?,
        )?;
        let start = (0..3usize)
            .map(|index| attribute.element(index))
            .collect::<Result<Vec<_>, _>>()?
            .into_iter()
            .sum::<i32>() as usize;
        let group_len = attribute.element(3usize)? as usize;
        if group_len == 0 {
            Err(::melior::Error::OperandNotFound("reset"))
        } else {
            self.operation.operand(start)
        }
    }
    pub fn reset_value(&self) -> Result<::melior::ir::Value<'c, '_>, ::melior::Error> {
        let attribute = ::melior::ir::attribute::DenseI32ArrayAttribute::<'c>::try_from(
            self.operation.attribute("operand_segment_sizes")?,
        )?;
        let start = (0..4usize)
            .map(|index| attribute.element(index))
            .collect::<Result<Vec<_>, _>>()?
            .into_iter()
            .sum::<i32>() as usize;
        let group_len = attribute.element(4usize)? as usize;
        if group_len == 0 {
            Err(::melior::Error::OperandNotFound("resetValue"))
        } else {
            self.operation.operand(start)
        }
    }
    pub fn power_on_value(&self) -> Result<::melior::ir::Value<'c, '_>, ::melior::Error> {
        let attribute = ::melior::ir::attribute::DenseI32ArrayAttribute::<'c>::try_from(
            self.operation.attribute("operand_segment_sizes")?,
        )?;
        let start = (0..5usize)
            .map(|index| attribute.element(index))
            .collect::<Result<Vec<_>, _>>()?
            .into_iter()
            .sum::<i32>() as usize;
        let group_len = attribute.element(5usize)? as usize;
        if group_len == 0 {
            Err(::melior::Error::OperandNotFound("powerOnValue"))
        } else {
            self.operation.operand(start)
        }
    }
    #[allow(clippy::needless_question_mark)]
    pub fn num_elements(
        &self,
    ) -> Result<::melior::ir::attribute::IntegerAttribute<'c>, ::melior::Error> {
        Ok(self.operation.attribute("numElements")?.try_into()?)
    }
    pub fn set_num_elements(&mut self, value: ::melior::ir::attribute::IntegerAttribute<'c>) {
        self.operation.set_attribute("numElements", value.into());
    }
    #[allow(clippy::needless_question_mark)]
    pub fn _name(&self) -> Result<::melior::ir::attribute::StringAttribute<'c>, ::melior::Error> {
        Ok(self.operation.attribute("name")?.try_into()?)
    }
    pub fn set_name(&mut self, value: ::melior::ir::attribute::StringAttribute<'c>) {
        self.operation.set_attribute("name", value.into());
    }
    pub fn remove_name(&mut self) -> Result<(), ::melior::Error> {
        self.operation.remove_attribute("name")
    }
    #[allow(clippy::needless_question_mark)]
    pub fn inner_sym(&self) -> Result<::melior::ir::attribute::Attribute<'c>, ::melior::Error> {
        Ok(self.operation.attribute("inner_sym")?.try_into()?)
    }
    pub fn set_inner_sym(&mut self, value: ::melior::ir::attribute::Attribute<'c>) {
        self.operation.set_attribute("inner_sym", value.into());
    }
    pub fn remove_inner_sym(&mut self) -> Result<(), ::melior::Error> {
        self.operation.remove_attribute("inner_sym")
    }
}
#[doc = "A builder for a [`shiftreg`](ShiftRegOperation) operation."]
pub struct ShiftRegOperationBuilder<'c, T0, O0, O1, O2, A0> {
    builder: ::melior::ir::operation::OperationBuilder<'c>,
    context: &'c ::melior::Context,
    _state: ::std::marker::PhantomData<(T0, O0, O1, O2, A0)>,
}
impl<'c>
    ShiftRegOperationBuilder<
        'c,
        ::melior::dialect::ods::__private::Unset,
        ::melior::dialect::ods::__private::Unset,
        ::melior::dialect::ods::__private::Unset,
        ::melior::dialect::ods::__private::Unset,
        ::melior::dialect::ods::__private::Unset,
    >
{
    pub fn new(context: &'c ::melior::Context, location: ::melior::ir::Location<'c>) -> Self {
        Self {
            context,
            builder: ::melior::ir::operation::OperationBuilder::new("seq.shiftreg", location),
            _state: Default::default(),
        }
    }
}
impl<'c, O0, O1, O2, A0>
    ShiftRegOperationBuilder<'c, ::melior::dialect::ods::__private::Unset, O0, O1, O2, A0>
{
    pub fn data(
        self,
        data: ::melior::ir::Type<'c>,
    ) -> ShiftRegOperationBuilder<'c, ::melior::dialect::ods::__private::Set, O0, O1, O2, A0> {
        ShiftRegOperationBuilder {
            context: self.context,
            builder: self.builder.add_results(&[data]),
            _state: Default::default(),
        }
    }
}
impl<'c, T0, O1, O2, A0>
    ShiftRegOperationBuilder<'c, T0, ::melior::dialect::ods::__private::Unset, O1, O2, A0>
{
    pub fn input(
        self,
        input: ::melior::ir::Value<'c, '_>,
    ) -> ShiftRegOperationBuilder<'c, T0, ::melior::dialect::ods::__private::Set, O1, O2, A0> {
        ShiftRegOperationBuilder {
            context: self.context,
            builder: self.builder.add_operands(&[input]),
            _state: Default::default(),
        }
    }
}
impl<'c, T0, O2, A0>
    ShiftRegOperationBuilder<
        'c,
        T0,
        ::melior::dialect::ods::__private::Set,
        ::melior::dialect::ods::__private::Unset,
        O2,
        A0,
    >
{
    pub fn clk(
        self,
        clk: ::melior::ir::Value<'c, '_>,
    ) -> ShiftRegOperationBuilder<
        'c,
        T0,
        ::melior::dialect::ods::__private::Set,
        ::melior::dialect::ods::__private::Set,
        O2,
        A0,
    > {
        ShiftRegOperationBuilder {
            context: self.context,
            builder: self.builder.add_operands(&[clk]),
            _state: Default::default(),
        }
    }
}
impl<'c, T0, A0>
    ShiftRegOperationBuilder<
        'c,
        T0,
        ::melior::dialect::ods::__private::Set,
        ::melior::dialect::ods::__private::Set,
        ::melior::dialect::ods::__private::Unset,
        A0,
    >
{
    pub fn clock_enable(
        self,
        clock_enable: ::melior::ir::Value<'c, '_>,
    ) -> ShiftRegOperationBuilder<
        'c,
        T0,
        ::melior::dialect::ods::__private::Set,
        ::melior::dialect::ods::__private::Set,
        ::melior::dialect::ods::__private::Set,
        A0,
    > {
        ShiftRegOperationBuilder {
            context: self.context,
            builder: self.builder.add_operands(&[clock_enable]),
            _state: Default::default(),
        }
    }
}
impl<'c, T0, O0, O1, O2, A0> ShiftRegOperationBuilder<'c, T0, O0, O1, O2, A0> {
    pub fn reset(
        mut self,
        reset: ::melior::ir::Value<'c, '_>,
    ) -> ShiftRegOperationBuilder<'c, T0, O0, O1, O2, A0> {
        self.builder = self.builder.add_operands(&[reset]);
        self
    }
}
impl<'c, T0, O0, O1, O2, A0> ShiftRegOperationBuilder<'c, T0, O0, O1, O2, A0> {
    pub fn reset_value(
        mut self,
        reset_value: ::melior::ir::Value<'c, '_>,
    ) -> ShiftRegOperationBuilder<'c, T0, O0, O1, O2, A0> {
        self.builder = self.builder.add_operands(&[reset_value]);
        self
    }
}
impl<'c, T0, O0, O1, O2, A0> ShiftRegOperationBuilder<'c, T0, O0, O1, O2, A0> {
    pub fn power_on_value(
        mut self,
        power_on_value: ::melior::ir::Value<'c, '_>,
    ) -> ShiftRegOperationBuilder<'c, T0, O0, O1, O2, A0> {
        self.builder = self.builder.add_operands(&[power_on_value]);
        self
    }
}
impl<'c, T0, O0, O1, O2>
    ShiftRegOperationBuilder<'c, T0, O0, O1, O2, ::melior::dialect::ods::__private::Unset>
{
    pub fn num_elements(
        self,
        num_elements: ::melior::ir::attribute::IntegerAttribute<'c>,
    ) -> ShiftRegOperationBuilder<'c, T0, O0, O1, O2, ::melior::dialect::ods::__private::Set> {
        ShiftRegOperationBuilder {
            context: self.context,
            builder: self.builder.add_attributes(&[(
                ::melior::ir::Identifier::new(self.context, "numElements"),
                num_elements.into(),
            )]),
            _state: Default::default(),
        }
    }
}
impl<'c, T0, O0, O1, O2, A0> ShiftRegOperationBuilder<'c, T0, O0, O1, O2, A0> {
    pub fn _name(
        mut self,
        _name: ::melior::ir::attribute::StringAttribute<'c>,
    ) -> ShiftRegOperationBuilder<'c, T0, O0, O1, O2, A0> {
        self.builder = self.builder.add_attributes(&[(
            ::melior::ir::Identifier::new(self.context, "name"),
            _name.into(),
        )]);
        self
    }
}
impl<'c, T0, O0, O1, O2, A0> ShiftRegOperationBuilder<'c, T0, O0, O1, O2, A0> {
    pub fn inner_sym(
        mut self,
        inner_sym: ::melior::ir::attribute::Attribute<'c>,
    ) -> ShiftRegOperationBuilder<'c, T0, O0, O1, O2, A0> {
        self.builder = self.builder.add_attributes(&[(
            ::melior::ir::Identifier::new(self.context, "inner_sym"),
            inner_sym.into(),
        )]);
        self
    }
}
impl<'c>
    ShiftRegOperationBuilder<
        'c,
        ::melior::dialect::ods::__private::Set,
        ::melior::dialect::ods::__private::Set,
        ::melior::dialect::ods::__private::Set,
        ::melior::dialect::ods::__private::Set,
        ::melior::dialect::ods::__private::Set,
    >
{
    pub fn build(self) -> ShiftRegOperation<'c> {
        self.builder
            .build()
            .expect("valid operation")
            .try_into()
            .expect("should be a valid ShiftRegOperation")
    }
}
#[allow(clippy::too_many_arguments)]
#[doc = "Creates a [`shiftreg`](ShiftRegOperation) operation."]
pub fn shiftreg<'c>(
    context: &'c ::melior::Context,
    data: ::melior::ir::Type<'c>,
    input: ::melior::ir::Value<'c, '_>,
    clk: ::melior::ir::Value<'c, '_>,
    clock_enable: ::melior::ir::Value<'c, '_>,
    num_elements: ::melior::ir::attribute::IntegerAttribute<'c>,
    location: ::melior::ir::Location<'c>,
) -> ShiftRegOperation<'c> {
    ShiftRegOperation::builder(context, location)
        .data(data)
        .input(input)
        .clk(clk)
        .clock_enable(clock_enable)
        .num_elements(num_elements)
        .build()
}
impl<'c> TryFrom<::melior::ir::operation::Operation<'c>> for ShiftRegOperation<'c> {
    type Error = ::melior::Error;
    fn try_from(operation: ::melior::ir::operation::Operation<'c>) -> Result<Self, Self::Error> {
        Ok(Self { operation })
    }
}
impl<'c> From<ShiftRegOperation<'c>> for ::melior::ir::operation::Operation<'c> {
    fn from(operation: ShiftRegOperation<'c>) -> Self {
        operation.operation
    }
}
#[doc = "A [`to_clock`](ToClockOperation) operation. Cast from a wire type to a clock type."]
#[doc = "\n\n"]
#[doc = ""]
pub struct ToClockOperation<'c> {
    operation: ::melior::ir::operation::Operation<'c>,
}
impl<'c> ToClockOperation<'c> {
    #[doc = r" Returns a name."]
    pub fn name() -> &'static str {
        "seq.to_clock"
    }
    #[doc = r" Returns a generic operation."]
    pub fn as_operation(&self) -> &::melior::ir::operation::Operation<'c> {
        &self.operation
    }
    #[doc = r" Creates a builder."]
    pub fn builder(
        context: &'c ::melior::Context,
        location: ::melior::ir::Location<'c>,
    ) -> ToClockOperationBuilder<
        'c,
        ::melior::dialect::ods::__private::Unset,
        ::melior::dialect::ods::__private::Unset,
    > {
        ToClockOperationBuilder::new(context, location)
    }
    pub fn output(
        &self,
    ) -> Result<::melior::ir::operation::OperationResult<'c, '_>, ::melior::Error> {
        self.operation.result(0usize)
    }
    pub fn input(&self) -> Result<::melior::ir::Value<'c, '_>, ::melior::Error> {
        self.operation.operand(0usize)
    }
}
#[doc = "A builder for a [`to_clock`](ToClockOperation) operation."]
pub struct ToClockOperationBuilder<'c, T0, O0> {
    builder: ::melior::ir::operation::OperationBuilder<'c>,
    context: &'c ::melior::Context,
    _state: ::std::marker::PhantomData<(T0, O0)>,
}
impl<'c>
    ToClockOperationBuilder<
        'c,
        ::melior::dialect::ods::__private::Unset,
        ::melior::dialect::ods::__private::Unset,
    >
{
    pub fn new(context: &'c ::melior::Context, location: ::melior::ir::Location<'c>) -> Self {
        Self {
            context,
            builder: ::melior::ir::operation::OperationBuilder::new("seq.to_clock", location),
            _state: Default::default(),
        }
    }
}
impl<'c, O0> ToClockOperationBuilder<'c, ::melior::dialect::ods::__private::Unset, O0> {
    pub fn output(
        self,
        output: ::melior::ir::Type<'c>,
    ) -> ToClockOperationBuilder<'c, ::melior::dialect::ods::__private::Set, O0> {
        ToClockOperationBuilder {
            context: self.context,
            builder: self.builder.add_results(&[output]),
            _state: Default::default(),
        }
    }
}
impl<'c, T0> ToClockOperationBuilder<'c, T0, ::melior::dialect::ods::__private::Unset> {
    pub fn input(
        self,
        input: ::melior::ir::Value<'c, '_>,
    ) -> ToClockOperationBuilder<'c, T0, ::melior::dialect::ods::__private::Set> {
        ToClockOperationBuilder {
            context: self.context,
            builder: self.builder.add_operands(&[input]),
            _state: Default::default(),
        }
    }
}
impl<'c>
    ToClockOperationBuilder<
        'c,
        ::melior::dialect::ods::__private::Set,
        ::melior::dialect::ods::__private::Set,
    >
{
    pub fn build(self) -> ToClockOperation<'c> {
        self.builder
            .build()
            .expect("valid operation")
            .try_into()
            .expect("should be a valid ToClockOperation")
    }
}
#[allow(clippy::too_many_arguments)]
#[doc = "Creates a [`to_clock`](ToClockOperation) operation."]
pub fn to_clock<'c>(
    context: &'c ::melior::Context,
    output: ::melior::ir::Type<'c>,
    input: ::melior::ir::Value<'c, '_>,
    location: ::melior::ir::Location<'c>,
) -> ToClockOperation<'c> {
    ToClockOperation::builder(context, location)
        .output(output)
        .input(input)
        .build()
}
impl<'c> TryFrom<::melior::ir::operation::Operation<'c>> for ToClockOperation<'c> {
    type Error = ::melior::Error;
    fn try_from(operation: ::melior::ir::operation::Operation<'c>) -> Result<Self, Self::Error> {
        Ok(Self { operation })
    }
}
impl<'c> From<ToClockOperation<'c>> for ::melior::ir::operation::Operation<'c> {
    fn from(operation: ToClockOperation<'c>) -> Self {
        operation.operation
    }
}
#[doc = "A [`write`](WritePortOperation) operation. Structural write access to a seq.hlmem."]
#[doc = "\n\n"]
#[doc = ""]
pub struct WritePortOperation<'c> {
    operation: ::melior::ir::operation::Operation<'c>,
}
impl<'c> WritePortOperation<'c> {
    #[doc = r" Returns a name."]
    pub fn name() -> &'static str {
        "seq.write"
    }
    #[doc = r" Returns a generic operation."]
    pub fn as_operation(&self) -> &::melior::ir::operation::Operation<'c> {
        &self.operation
    }
    #[doc = r" Creates a builder."]
    pub fn builder(
        context: &'c ::melior::Context,
        location: ::melior::ir::Location<'c>,
    ) -> WritePortOperationBuilder<
        'c,
        ::melior::dialect::ods::__private::Unset,
        ::melior::dialect::ods::__private::Unset,
        ::melior::dialect::ods::__private::Unset,
        ::melior::dialect::ods::__private::Unset,
        ::melior::dialect::ods::__private::Unset,
    > {
        WritePortOperationBuilder::new(context, location)
    }
    pub fn memory(&self) -> Result<::melior::ir::Value<'c, '_>, ::melior::Error> {
        self.operation.operand(0usize)
    }
    pub fn addresses(&self) -> impl Iterator<Item = ::melior::ir::Value<'c, '_>> {
        let group_length = self.operation.operand_count() - 4usize + 1;
        self.operation.operands().skip(1usize).take(group_length)
    }
    pub fn in_data(&self) -> Result<::melior::ir::Value<'c, '_>, ::melior::Error> {
        let group_length = self.operation.operand_count() - 4usize + 1;
        self.operation.operand(2usize + group_length - 1)
    }
    pub fn wr_en(&self) -> Result<::melior::ir::Value<'c, '_>, ::melior::Error> {
        let group_length = self.operation.operand_count() - 4usize + 1;
        self.operation.operand(3usize + group_length - 1)
    }
    #[allow(clippy::needless_question_mark)]
    pub fn latency(
        &self,
    ) -> Result<::melior::ir::attribute::IntegerAttribute<'c>, ::melior::Error> {
        Ok(self.operation.attribute("latency")?.try_into()?)
    }
    pub fn set_latency(&mut self, value: ::melior::ir::attribute::IntegerAttribute<'c>) {
        self.operation.set_attribute("latency", value.into());
    }
}
#[doc = "A builder for a [`write`](WritePortOperation) operation."]
pub struct WritePortOperationBuilder<'c, O0, O1, O2, O3, A0> {
    builder: ::melior::ir::operation::OperationBuilder<'c>,
    context: &'c ::melior::Context,
    _state: ::std::marker::PhantomData<(O0, O1, O2, O3, A0)>,
}
impl<'c>
    WritePortOperationBuilder<
        'c,
        ::melior::dialect::ods::__private::Unset,
        ::melior::dialect::ods::__private::Unset,
        ::melior::dialect::ods::__private::Unset,
        ::melior::dialect::ods::__private::Unset,
        ::melior::dialect::ods::__private::Unset,
    >
{
    pub fn new(context: &'c ::melior::Context, location: ::melior::ir::Location<'c>) -> Self {
        Self {
            context,
            builder: ::melior::ir::operation::OperationBuilder::new("seq.write", location),
            _state: Default::default(),
        }
    }
}
impl<'c, O1, O2, O3, A0>
    WritePortOperationBuilder<'c, ::melior::dialect::ods::__private::Unset, O1, O2, O3, A0>
{
    pub fn memory(
        self,
        memory: ::melior::ir::Value<'c, '_>,
    ) -> WritePortOperationBuilder<'c, ::melior::dialect::ods::__private::Set, O1, O2, O3, A0> {
        WritePortOperationBuilder {
            context: self.context,
            builder: self.builder.add_operands(&[memory]),
            _state: Default::default(),
        }
    }
}
impl<'c, O2, O3, A0>
    WritePortOperationBuilder<
        'c,
        ::melior::dialect::ods::__private::Set,
        ::melior::dialect::ods::__private::Unset,
        O2,
        O3,
        A0,
    >
{
    pub fn addresses(
        self,
        addresses: &[::melior::ir::Value<'c, '_>],
    ) -> WritePortOperationBuilder<
        'c,
        ::melior::dialect::ods::__private::Set,
        ::melior::dialect::ods::__private::Set,
        O2,
        O3,
        A0,
    > {
        WritePortOperationBuilder {
            context: self.context,
            builder: self.builder.add_operands(addresses),
            _state: Default::default(),
        }
    }
}
impl<'c, O3, A0>
    WritePortOperationBuilder<
        'c,
        ::melior::dialect::ods::__private::Set,
        ::melior::dialect::ods::__private::Set,
        ::melior::dialect::ods::__private::Unset,
        O3,
        A0,
    >
{
    pub fn in_data(
        self,
        in_data: ::melior::ir::Value<'c, '_>,
    ) -> WritePortOperationBuilder<
        'c,
        ::melior::dialect::ods::__private::Set,
        ::melior::dialect::ods::__private::Set,
        ::melior::dialect::ods::__private::Set,
        O3,
        A0,
    > {
        WritePortOperationBuilder {
            context: self.context,
            builder: self.builder.add_operands(&[in_data]),
            _state: Default::default(),
        }
    }
}
impl<'c, A0>
    WritePortOperationBuilder<
        'c,
        ::melior::dialect::ods::__private::Set,
        ::melior::dialect::ods::__private::Set,
        ::melior::dialect::ods::__private::Set,
        ::melior::dialect::ods::__private::Unset,
        A0,
    >
{
    pub fn wr_en(
        self,
        wr_en: ::melior::ir::Value<'c, '_>,
    ) -> WritePortOperationBuilder<
        'c,
        ::melior::dialect::ods::__private::Set,
        ::melior::dialect::ods::__private::Set,
        ::melior::dialect::ods::__private::Set,
        ::melior::dialect::ods::__private::Set,
        A0,
    > {
        WritePortOperationBuilder {
            context: self.context,
            builder: self.builder.add_operands(&[wr_en]),
            _state: Default::default(),
        }
    }
}
impl<'c, O0, O1, O2, O3>
    WritePortOperationBuilder<'c, O0, O1, O2, O3, ::melior::dialect::ods::__private::Unset>
{
    pub fn latency(
        self,
        latency: ::melior::ir::attribute::IntegerAttribute<'c>,
    ) -> WritePortOperationBuilder<'c, O0, O1, O2, O3, ::melior::dialect::ods::__private::Set> {
        WritePortOperationBuilder {
            context: self.context,
            builder: self.builder.add_attributes(&[(
                ::melior::ir::Identifier::new(self.context, "latency"),
                latency.into(),
            )]),
            _state: Default::default(),
        }
    }
}
impl<'c>
    WritePortOperationBuilder<
        'c,
        ::melior::dialect::ods::__private::Set,
        ::melior::dialect::ods::__private::Set,
        ::melior::dialect::ods::__private::Set,
        ::melior::dialect::ods::__private::Set,
        ::melior::dialect::ods::__private::Set,
    >
{
    pub fn build(self) -> WritePortOperation<'c> {
        self.builder
            .build()
            .expect("valid operation")
            .try_into()
            .expect("should be a valid WritePortOperation")
    }
}
#[allow(clippy::too_many_arguments)]
#[doc = "Creates a [`write`](WritePortOperation) operation."]
pub fn write<'c>(
    context: &'c ::melior::Context,
    memory: ::melior::ir::Value<'c, '_>,
    addresses: &[::melior::ir::Value<'c, '_>],
    in_data: ::melior::ir::Value<'c, '_>,
    wr_en: ::melior::ir::Value<'c, '_>,
    latency: ::melior::ir::attribute::IntegerAttribute<'c>,
    location: ::melior::ir::Location<'c>,
) -> WritePortOperation<'c> {
    WritePortOperation::builder(context, location)
        .memory(memory)
        .addresses(addresses)
        .in_data(in_data)
        .wr_en(wr_en)
        .latency(latency)
        .build()
}
impl<'c> TryFrom<::melior::ir::operation::Operation<'c>> for WritePortOperation<'c> {
    type Error = ::melior::Error;
    fn try_from(operation: ::melior::ir::operation::Operation<'c>) -> Result<Self, Self::Error> {
        Ok(Self { operation })
    }
}
impl<'c> From<WritePortOperation<'c>> for ::melior::ir::operation::Operation<'c> {
    fn from(operation: WritePortOperation<'c>) -> Self {
        operation.operation
    }
}
