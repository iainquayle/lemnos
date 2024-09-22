from __future__ import annotations

from ...templates.torch import *
from ...templates.python import *
from ...schema.components import *

from .generation import InitType, StatementGeneratorOutput, StatementGeneratorArgs, SourceGenerator


def standard_module(module: str, args: StatementGeneratorArgs) -> StatementGeneratorOutput:
	identifier = args.member_identifier_generator.get_identifier()
	init_statement = [assign_(self_(identifier), nn_(module))]
	return_expression = call_(self_(identifier),  args.input_register)
	return StatementGeneratorOutput(init_statement, [], return_expression)

def conv_generator(self: Conv, args: StatementGeneratorArgs) -> StatementGeneratorOutput:
	return standard_module(f"Conv{len(args.input_shape) - 1}d({args.input_shape[0]}, {args.output_shape[0]}, {self._kernel}, {self._stride}, {self._padding}, {self._dilation}, {self._groups})", args)

def maxpool_generator(self: MaxPool, args: StatementGeneratorArgs) -> StatementGeneratorOutput:
	return standard_module(f"MaxPool{len(args.input_shape) - 1}d({self._kernel}, {self._stride}, {self._padding}, {self._dilation})", args)

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
	Conv: conv_generator,
	Full: full_generator,
})
