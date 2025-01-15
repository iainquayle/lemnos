from __future__ import annotations

from lemnos.schema.components import *
from ..templates.torch import *
from ..templates.python import *

from .generator import ComponentStatements, StatementGeneratorArgs, SourceGenerator,  StatementGenerator, ShapeView


def standard_module(module: str, args: StatementGeneratorArgs) -> ComponentStatements:
	member_identifier = args.member_identifier_generator.get_identifier()
	init_statement = [assign_(member_identifier, self_(nn_(module)))]
	return_expression = call_(member_identifier,  args.input_registers[0])
	return ComponentStatements(init_statement, [], return_expression)


def sum_generator(self: Sum, args: StatementGeneratorArgs) -> ComponentStatements:
	return ComponentStatements([], [], sum_(args.input_registers), )


def concat_generator(self: Concat, args: StatementGeneratorArgs) -> ComponentStatements:
	return ComponentStatements([], [], concat_(args.input_registers, True), )


def conv_generator(self: Conv, args: StatementGeneratorArgs) -> ComponentStatements:
	dimensionality = len(args.input_shape) - 1
	return standard_module(f"Conv{dimensionality}d({args.input_shape[0]}, {args.output_shape[0]}, {self._kernel.expand(dimensionality)}, {self._stride.expand(dimensionality)}, {self._padding.expand(dimensionality)}, {self._dilation.expand(dimensionality)}, {self._groups})", args, )


def maxpool_generator(self: MaxPool, args: StatementGeneratorArgs) -> ComponentStatements:
	dimensionality = len(args.input_shape) - 1
	return standard_module(f"MaxPool{dimensionality}d({self._kernel.expand(dimensionality)}, {self._stride.expand(dimensionality)}, {self._padding.expand(dimensionality)}, {self._dilation.expand(dimensionality)})", args, )


def full_generator(self: Dense, args: StatementGeneratorArgs) -> ComponentStatements:
	return standard_module(f"Linear({args.input_shape.get_product()}, {args.output_shape.get_product()}, bias=True)", args, )


def relu_generator(self: Relu, args: StatementGeneratorArgs) -> ComponentStatements:
	return standard_module("ReLU()", args, )


def relu6_generator(self: Relu6, args: StatementGeneratorArgs) -> ComponentStatements:
	return standard_module("ReLU6()", args, )


def silu_generator(self: Silu, args: StatementGeneratorArgs) -> ComponentStatements:
	return standard_module("SiLU()", args, )


def sigmoid_generator(self: Sigmoid, args: StatementGeneratorArgs) -> ComponentStatements:
	return standard_module("Sigmoid()", args, )


def softmax_generator(self: Softmax, args: StatementGeneratorArgs) -> ComponentStatements:
	return standard_module("Softmax(dim=1)", args, )


def batchnorm_generator(self: BatchNorm, args: StatementGeneratorArgs) -> ComponentStatements:
	return standard_module(f"BatchNorm{len(args.input_shape) - 1}d({args.input_shape[0]}, momentum={self._momentum})", args, )


def layernorm_generator(self: LayerNorm, args: StatementGeneratorArgs) -> ComponentStatements:
	return standard_module(f"LayerNorm({args.input_shape.get_tuple()})", args, )


def rmsnorm_generator(self: RmsNorm, args: StatementGeneratorArgs) -> ComponentStatements:
	return standard_module(f"RMSNorm({args.input_shape.get_tuple()})", args, )


def dropout_generator(self: Dropout, args: StatementGeneratorArgs) -> ComponentStatements:
	return standard_module(f"Dropout(p={self._probability})", args, )


def channel_dropout_generator(self: ChannelDropout, args: StatementGeneratorArgs) -> ComponentStatements:
	return standard_module(f"Dropout{args.input_shape.dimensionality() - 1}d(p={self._probability})", args, )


def glu_generator(self: Glu, args: StatementGeneratorArgs) -> ComponentStatements:
	return standard_module("GLU(dim=1)", args, )


#need to figure out how to deal with grouping, apperently channels need to be divisible by num_groups
#may beed to impl some custom norm layers 
def groupnorm_generator(self: GroupNorm, args: StatementGeneratorArgs) -> ComponentStatements:
	return standard_module(f"GroupNorm({self._num_groups}, {args.input_shape[0]})", args, )	


standard_generator = SourceGenerator({ 
	Concat: StatementGenerator(concat_generator, ShapeView.FLAT),
	Sum: StatementGenerator(sum_generator, ShapeView.FLAT),
	Conv: StatementGenerator(conv_generator, ShapeView.REAL),
	Dense: StatementGenerator(full_generator, ShapeView.FLAT),
	MaxPool: StatementGenerator(maxpool_generator, ShapeView.REAL),
	Relu: StatementGenerator(relu_generator, ShapeView.EITHER),
	Relu6: StatementGenerator(relu6_generator, ShapeView.EITHER),
	Silu: StatementGenerator(silu_generator, ShapeView.EITHER),
	Sigmoid: StatementGenerator(sigmoid_generator, ShapeView.EITHER),
	Softmax: StatementGenerator(softmax_generator, ShapeView.EITHER),
	Glu: StatementGenerator(glu_generator, ShapeView.REAL),
	BatchNorm: StatementGenerator(batchnorm_generator, ShapeView.REAL),
	LayerNorm: StatementGenerator(layernorm_generator, ShapeView.REAL),
	RmsNorm: StatementGenerator(rmsnorm_generator, ShapeView.REAL),
	Dropout: StatementGenerator(dropout_generator, ShapeView.EITHER),
	ChannelDropout: StatementGenerator(channel_dropout_generator, ShapeView.REAL),
})
