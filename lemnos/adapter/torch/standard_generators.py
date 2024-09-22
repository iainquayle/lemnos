from ...shared import LockedShape

from ...templates.torch import *
from ...schema.components import *

from .generation import InitType, ComponentGeneratorData, ComponentGenerator

#need to be able to pass out the statement and the variable name

def conv_generator(self: Conv, input_shape: LockedShape, output_shape: LockedShape, register: str, **kwargs: Any) -> Any:
	return [InitStatement( InitType.CALLABLE, '',
		nn_(f"Conv{len(input_shape) - 1}d({input_shape[0]}, {output_shape[0]}, {self._kernel}, {self._stride}, {self._padding}, {self._dilation}, {self._groups})"),
	)]

def maxpool_formatter(self: MaxPool, input_shape: LockedShape, output_shape: LockedShape) -> list[InitStatement]:
	return [InitStatement( InitType.CALLABLE, '',
		nn_(f"MaxPool{len(input_shape) - 1}d({self._kernel}, {self._stride}, {self._padding}, {self._dilation})")
	)]
def full_formatter(self: Full, input_shape: LockedShape, output_shape: LockedShape) -> list[InitStatement]:
	return [InitStatement( InitType.CALLABLE, '',
		nn_(f"Linear({input_shape.get_product()}, {output_shape.get_product()}, bias=True)")
	)]
def relu_formatter(self: ReLU, input_shape: LockedShape, output_shape: LockedShape) -> list[InitStatement]:
	return [InitStatement( InitType.CALLABLE, '', nn_("ReLU()"))]
def relu6_formatter(self: ReLU6, input_shape: LockedShape, output_shape: LockedShape) -> list[InitStatement]:
	return [InitStatement( InitType.CALLABLE, '', nn_("ReLU6()"))]
def silu_formatter(self: SiLU, input_shape: LockedShape, output_shape: LockedShape) -> list[InitStatement]:
	return [InitStatement( InitType.CALLABLE, '', nn_("SiLU()"))]
def sigmoid_formatter(self: Sigmoid, input_shape: LockedShape, output_shape: LockedShape) -> list[InitStatement]:
	return [InitStatement( InitType.CALLABLE, '', nn_("Sigmoid()"))]
def softmax_formatter(self: Softmax, input_shape: LockedShape, output_shape: LockedShape) -> list[InitStatement]:
	return [InitStatement( InitType.CALLABLE, '', nn_("Softmax(dim=1)"))]
def batchnorm_formatter(self: BatchNorm, input_shape: LockedShape, output_shape: LockedShape) -> list[InitStatement]:
	return [InitStatement( InitType.CALLABLE, '',
		nn_(f"BatchNorm{len(input_shape) - 1}d({input_shape[0]})")
	)]
def layernorm_formatter(self: LayerNorm, input_shape: LockedShape, output_shape: LockedShape) -> list[InitStatement]:
	raise NotImplementedError
	return [InitStatement( InitType.CALLABLE, '', nn_(f"LayerNorm({input_shape})"))]
def dropout_formatter(self: Dropout, input_shape: LockedShape, output_shape: LockedShape) -> list[InitStatement]:
	return [InitStatement( InitType.CALLABLE, '', nn_(f"Dropout(p={self._probability})"))]
def channel_dropout_formatter(self: ChannelDropout, input_shape: LockedShape, output_shape: LockedShape) -> list[InitStatement]:
	return [InitStatement( InitType.CALLABLE, '', nn_(f"ChannelDropout(p={self._probability})"))]
def glu_formatter(self: GLU, input_shape: LockedShape, output_shape: LockedShape) -> list[InitStatement]:
	return [InitStatement( InitType.CALLABLE, '', nn_("GLU(dim=1)"))]


def conv_forward(self: Conv,  input_expr: str) -> str:
	return nn_(f"Conv{len(self.input_shape) - 1}d({input_expr})")
