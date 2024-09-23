from __future__ import annotations

from ...templates.torch import *
from ...templates.python import *
from ...schema.components import *

from .generation import InitType, StatementGeneratorOutput, StatementGeneratorArgs, SourceGenerator

def standard_module(module: str, args: StatementGeneratorArgs) -> StatementGeneratorOutput:
	member_identifier = args.member_identifier_generator.get_identifier()
	init_statement = [assign_(member_identifier, self_(nn_(module)))]
	return_expression = call_(member_identifier,  args.input_registers[0])
	return StatementGeneratorOutput(init_statement, [], return_expression)

def sum_generator(self: Sum, args: StatementGeneratorArgs) -> StatementGeneratorOutput:
	return StatementGeneratorOutput([], [], sum_(args.input_registers))

def concat_generator(self: Concat, args: StatementGeneratorArgs) -> StatementGeneratorOutput:
	return StatementGeneratorOutput([], [], self_(concat_(args.input_registers)))

def conv_generator(self: Conv, args: StatementGeneratorArgs) -> StatementGeneratorOutput:
	dimensionality = len(args.input_shape) - 1
	return standard_module(f"Conv{dimensionality}d({args.input_shape[0]}, {args.output_shape[0]}, {self._kernel.expand(dimensionality)}, {self._stride.expand(dimensionality)}, {self._padding.expand(dimensionality)}, {self._dilation.expand(dimensionality)}, {self._groups})", args)

def maxpool_generator(self: MaxPool, args: StatementGeneratorArgs) -> StatementGeneratorOutput:
	dimensionality = len(args.input_shape) - 1
	return standard_module(f"MaxPool{dimensionality}d({self._kernel.expand(dimensionality)}, {self._stride.expand(dimensionality)}, {self._padding.expand(dimensionality)}, {self._dilation.expand(dimensionality)})", args)

def full_generator(self: Full, args: StatementGeneratorArgs) -> StatementGeneratorOutput:
	return standard_module(f"Linear({args.input_shape.get_product()}, {args.output_shape.get_product()}, bias=True)", args)

def relu_generator(self: ReLU, args: StatementGeneratorArgs) -> StatementGeneratorOutput:
	return standard_module("ReLU()", args)

def relu6_generator(self: ReLU6, args: StatementGeneratorArgs) -> StatementGeneratorOutput:
	return standard_module("ReLU6()", args)

def silu_generator(self: SiLU, args: StatementGeneratorArgs) -> StatementGeneratorOutput:
	return standard_module("SiLU()", args)

def sigmoid_generator(self: Sigmoid, args: StatementGeneratorArgs) -> StatementGeneratorOutput:
	return standard_module("Sigmoid()", args)

def softmax_generator(self: Softmax, args: StatementGeneratorArgs) -> StatementGeneratorOutput:
	return standard_module("Softmax(dim=1)", args)

def batchnorm_generator(self: BatchNorm, args: StatementGeneratorArgs) -> StatementGeneratorOutput:
	return standard_module(f"BatchNorm{len(args.input_shape) - 1}d({args.input_shape[0]})", args)

def layernorm_generator(self: LayerNorm, args: StatementGeneratorArgs) -> StatementGeneratorOutput:
	raise NotImplementedError
	return [InitStatement( InitType.CALLABLE, '', nn_(f"LayerNorm({input_shape})"))]

def dropout_generator(self: Dropout, args: StatementGeneratorArgs) -> StatementGeneratorOutput:
	return standard_module(f"Dropout(p={self._probability})", args)

def channel_dropout_generator(self: ChannelDropout, args: StatementGeneratorArgs) -> StatementGeneratorOutput:
	return standard_module(f"ChannelDropout(p={self._probability})", args)

def glu_generator(self: GLU, args: StatementGeneratorArgs) -> StatementGeneratorOutput:
	return standard_module("GLU(dim=1)", args)

standard_generator = SourceGenerator({ 
	Concat: concat_generator,
	Sum: sum_generator,
	Conv: conv_generator,
	Full: full_generator,
	MaxPool: maxpool_generator,
	ReLU: relu_generator,
	ReLU6: relu6_generator,
	SiLU: silu_generator,
	Sigmoid: sigmoid_generator,
	Softmax: softmax_generator,
	BatchNorm: batchnorm_generator,
	LayerNorm: layernorm_generator,
	Dropout: dropout_generator,
	ChannelDropout: channel_dropout_generator,
	GLU: glu_generator
})
