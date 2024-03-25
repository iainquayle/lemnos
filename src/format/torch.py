from __future__ import annotations

def torch_nn(name: str) -> str:
	pass

		if isinstance(component, Conv):
			#need to fix up group
			return [torch_nn_(f"Conv{len(input_shape) - 1}d({input_shape[0]}, {output_shape[0]}, {component._kernel}, {component._stride}, {component._padding}, {component._group_size}, bias=True, padding_mode='zeros')")]
		elif isinstance(component, Full):
			return [torch_nn_(f"Linear({input_shape.get_product()}, {output_shape.get_product()}, bias=True)")] 
		elif isinstance(component, ReLU):
			return [torch_nn_("ReLU()")]
		elif isinstance(component, ReLU6):
			return [torch_nn_("ReLU6()")]
		elif isinstance(component, Sigmoid):
			return [torch_nn_("Sigmoid()")]
		elif isinstance(component, Softmax):
			return [torch_nn_("Softmax(dim=1)")]
		elif isinstance(component, BatchNormalization):
			return [torch_nn_(f"BatchNorm{len(input_shape) - 1}d({input_shape[0]})")]
		elif isinstance(component, Dropout):
			return [torch_nn_(f"Dropout(p={component._p})")]
		elif isinstance(component, ChannelDropout):
			return [torch_nn_(f"ChannelDropout(p={component._p})")]
