from __future__ import annotations

from ...templates.torch import *
from ...templates.python import *
from ...schema.components import *

from .generator import ComponentStatements, StatementGeneratorArgs, SourceGenerator,  StatementGenerator, ShapeView


def standard_module(module: str, args: StatementGeneratorArgs) -> ComponentStatements:
	member_identifier = args.member_identifier_generator.get_identifier()
	init_statement = [assign_(member_identifier, self_(nn_(module)))]
	return_expression = call_(member_identifier,  args.input_registers[0])
	return ComponentStatements(init_statement, [], return_expression)


def sum_generator(self: Sum, args: StatementGeneratorArgs) -> ComponentStatements:
	return ComponentStatements([], [], sum_(args.input_registers), )


def concat_generator(self: Concat, args: StatementGeneratorArgs) -> ComponentStatements:
	return ComponentStatements([], [], self_(concat_(args.input_registers)), )


def conv_generator(self: Conv, args: StatementGeneratorArgs) -> ComponentStatements:
	dimensionality = len(args.input_shape) - 1
	return standard_module(f"Conv{dimensionality}d({args.input_shape[0]}, {args.output_shape[0]}, {self._kernel.expand(dimensionality)}, {self._stride.expand(dimensionality)}, {self._padding.expand(dimensionality)}, {self._dilation.expand(dimensionality)}, {self._groups})", args, )


def maxpool_generator(self: MaxPool, args: StatementGeneratorArgs) -> ComponentStatements:
	dimensionality = len(args.input_shape) - 1
	return standard_module(f"MaxPool{dimensionality}d({self._kernel.expand(dimensionality)}, {self._stride.expand(dimensionality)}, {self._padding.expand(dimensionality)}, {self._dilation.expand(dimensionality)})", args, )


def full_generator(self: Full, args: StatementGeneratorArgs) -> ComponentStatements:
	return standard_module(f"Linear({args.input_shape.get_product()}, {args.output_shape.get_product()}, bias=True)", args, )


def relu_generator(self: ReLU, args: StatementGeneratorArgs) -> ComponentStatements:
	return standard_module("ReLU()", args, )


def relu6_generator(self: ReLU6, args: StatementGeneratorArgs) -> ComponentStatements:
	return standard_module("ReLU6()", args, )


def silu_generator(self: SiLU, args: StatementGeneratorArgs) -> ComponentStatements:
	return standard_module("SiLU()", args, )


def sigmoid_generator(self: Sigmoid, args: StatementGeneratorArgs) -> ComponentStatements:
	return standard_module("Sigmoid()", args, )


def softmax_generator(self: Softmax, args: StatementGeneratorArgs) -> ComponentStatements:
	return standard_module("Softmax(dim=1)", args, )


def batchnorm_generator(self: BatchNorm, args: StatementGeneratorArgs) -> ComponentStatements:
	return standard_module(f"BatchNorm{len(args.input_shape) - 1}d({args.input_shape[0]})", args, )


def layernorm_generator(self: LayerNorm, args: StatementGeneratorArgs) -> ComponentStatements:
	return standard_module(f"LayerNorm({args.input_shape.get_tuple()})", args, )


def rmsnorm_generator(self: RMSNorm, args: StatementGeneratorArgs) -> ComponentStatements:
	return standard_module(f"RMSNorm({args.input_shape.get_tuple()})", args, )


def dropout_generator(self: Dropout, args: StatementGeneratorArgs) -> ComponentStatements:
	return standard_module(f"Dropout(p={self._probability})", args, )


def channel_dropout_generator(self: ChannelDropout, args: StatementGeneratorArgs) -> ComponentStatements:
	return standard_module(f"ChannelDropout(p={self._probability})", args, )


def glu_generator(self: GLU, args: StatementGeneratorArgs) -> ComponentStatements:
	return standard_module("GLU(dim=1)", args, )


standard_generator = SourceGenerator({ 
	Concat: StatementGenerator(concat_generator, ShapeView.FLAT),
	Sum: StatementGenerator(sum_generator, ShapeView.FLAT),
	Conv: StatementGenerator(conv_generator, ShapeView.REAL),
	Full: StatementGenerator(full_generator, ShapeView.FLAT),
	MaxPool: StatementGenerator(maxpool_generator, ShapeView.REAL),
	ReLU: StatementGenerator(relu_generator, ShapeView.EITHER),
	ReLU6: StatementGenerator(relu6_generator, ShapeView.EITHER),
	SiLU: StatementGenerator(silu_generator, ShapeView.EITHER),
	Sigmoid: StatementGenerator(sigmoid_generator, ShapeView.EITHER),
	Softmax: StatementGenerator(softmax_generator, ShapeView.EITHER),
	BatchNorm: StatementGenerator(batchnorm_generator, ShapeView.REAL),
	LayerNorm: StatementGenerator(layernorm_generator, ShapeView.REAL),
	RMSNorm: StatementGenerator(rmsnorm_generator, ShapeView.REAL),
	Dropout: StatementGenerator(dropout_generator, ShapeView.EITHER),
	ChannelDropout: StatementGenerator(channel_dropout_generator, ShapeView.REAL),
	GLU: StatementGenerator(glu_generator, ShapeView.REAL),
})
