from ...shared import LockedShape

from ...templates.torch import *
from ...schema.components import *

from .formatter import InitType, InitStatement

#need to be able to pass out the statement and the variable name

def conv_init(self: Conv, input_shape: LockedShape, output_shape: LockedShape) -> list[InitStatement]:
	return [InitStatement( InitType.CALLABLE, '',
		nn_(f"Conv{len(input_shape) - 1}d({input_shape[0]}, {output_shape[0]}, {self._kernel}, {self._stride}, {self._padding}, {self._dilation}, {self._groups})"),
	)]
def maxpool_init(self: MaxPool, input_shape: LockedShape, output_shape: LockedShape) -> list[InitStatement]:
	return [InitStatement( InitType.CALLABLE, '',
		nn_(f"MaxPool{len(input_shape) - 1}d({self._kernel}, {self._stride}, {self._padding}, {self._dilation})")
	)]
def full_init(self: Full, input_shape: LockedShape, output_shape: LockedShape) -> list[InitStatement]:
	return [InitStatement( InitType.CALLABLE, '',
		nn_(f"Linear({input_shape.get_product()}, {output_shape.get_product()}, bias=True)")
	)]
def relu_init(self: ReLU, input_shape: LockedShape, output_shape: LockedShape) -> list[InitStatement]:
	return [InitStatement( InitType.CALLABLE, '', nn_("ReLU()"))]
def relu6_init(self: ReLU6, input_shape: LockedShape, output_shape: LockedShape) -> list[InitStatement]:
	return [InitStatement( InitType.CALLABLE, '', nn_("ReLU6()"))]
def silu_init(self: SiLU, input_shape: LockedShape, output_shape: LockedShape) -> list[InitStatement]:
	return [InitStatement( InitType.CALLABLE, '', nn_("SiLU()"))]
def sigmoid_init(self: Sigmoid, input_shape: LockedShape, output_shape: LockedShape) -> list[InitStatement]:
	return [InitStatement( InitType.CALLABLE, '', nn_("Sigmoid()"))]
def softmax_init(self: Softmax, input_shape: LockedShape, output_shape: LockedShape) -> list[InitStatement]:
	return [InitStatement( InitType.CALLABLE, '', nn_("Softmax(dim=1)"))]
def batchnorm_init(self: BatchNorm, input_shape: LockedShape, output_shape: LockedShape) -> list[InitStatement]:
	return [InitStatement( InitType.CALLABLE, '',
		nn_(f"BatchNorm{len(input_shape) - 1}d({input_shape[0]})")
	)]
def layernorm_init(self: LayerNorm, input_shape: LockedShape, output_shape: LockedShape) -> list[InitStatement]:
	raise NotImplementedError
	return [InitStatement( InitType.CALLABLE, '', nn_(f"LayerNorm({input_shape})"))]
def dropout_init(self: Dropout, input_shape: LockedShape, output_shape: LockedShape) -> list[InitStatement]:
	return [InitStatement( InitType.CALLABLE, '', nn_(f"Dropout(p={self._probability})"))]
def channel_dropout_init(self: ChannelDropout, input_shape: LockedShape, output_shape: LockedShape) -> list[InitStatement]:
	return [InitStatement( InitType.CALLABLE, '', nn_(f"ChannelDropout(p={self._probability})"))]
def glu_init(self: GLU, input_shape: LockedShape, output_shape: LockedShape) -> list[InitStatement]:
	return [InitStatement( InitType.CALLABLE, '', nn_("GLU(dim=1)"))]


def conv_forward(self: Conv,  input_expr: str) -> str:
	return nn_(f"Conv{len(self.input_shape) - 1}d({input_expr})")
