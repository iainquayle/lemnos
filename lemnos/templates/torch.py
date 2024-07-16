from __future__ import annotations

from ..shared import LockedShape
from .python import *

def view_(expr: str, shape: LockedShape) -> str:
	return f"{expr}.view(-1, {arg_list_(*to_str_list(iter(shape)))})"
def flatten_view_(expr: str, size: int | LockedShape) -> str:
	return f"{expr}.view(-1, {size if isinstance(size, int) else size.get_product()})"
def sum_(exprs: list[str]) -> str:
	if len(exprs) == 1:
		return exprs[0]
	return f"({' + '.join(exprs)})"
def cat_(exprs: list[str]) -> str:
	if len(exprs) == 1:
		return exprs[0]
	return torch_(f"cat(({arg_list_(*exprs)}), dim=1)")
def import_torch_() -> str:
	return "import torch"
def torch_(expr: str) -> str:
	return f"torch.{expr}"
def nn_(expr: str) -> str:
	return f"torch.nn.{expr}"
def functional_(expr: str) -> str:
	return f"torch.nn.functional.{expr}"
def module_(name: str, class_definitions: list[str], init_args: list[str], init_statements: list[str], forward_args: list[str], forward_statments: list[str]) -> list[str]:
	return (class_(name, [nn_("Module")], 
		[import_torch_()] + class_definitions +
		function_("__init__", ["self"] + init_args,["super().__init__()"] + init_statements) +
		function_("forward", ["self"] + forward_args, forward_statments)))
def conv_mix_definition_(dimensions: int) -> list[str]:
	return module_(f"ConvMix{dimensions}d", [], 
		["channels_in", "channels_out", "kernel", "stride", "padding", "dilation", "groups"], 
		[f"self.c = self.torch.nn.Conv{dimensions}d(channels_in, channels_out, kernel, stride, padding, dilation, groups,)",
		assign_("s", "channels_out // groups"),
		assign_(self_("indices"), "self.torch.Tensor([i + j * s for j in range(groups) for i in range(s)]).int()")],
		["x"], 
		[return_("self.c(x)[:, self.indices]")])
def flex_conv_definition(dimensions: int) -> list[str]:
	return module_(f"FlexConv{dimensions}d", [], 
		["conv_splits", "mix_indices", "kernel", "stride", "padding", "dilation"], 
		[f"self.modules_list = self.torch.nn.ModuleList([self.torch.nn.Conv{dimensions}d(channels_in, channels_out, kernel, stride, padding, dilation, groups) for channels_in, channels_out, groups in sub_conv_splits])",
   			"self.input_splits = self.torch.Tensor([channels_in for channels_in, _, _ in sub_conv_splits]).int()",
   			"self.mix_indices = self.torch.Tensor(mix_indices).int()" ],
		["x"], 
		["xs = self.torch.split(x, [self.channels_in_1, self.channels_in_2], dim=1)",
   			"return self.torch.cat([module(x) for module, x in zip(self.modules_list, xs)], dim=1)" ])

def conv_init_(input_shape: LockedShape, output_shape: LockedShape, kernel: tuple[int, ...], stride: tuple[int, ...], padding: tuple[int, ...], dilation: tuple[int, ...], groups: int) -> str:
	return nn_(f"Conv{len(input_shape) - 1}d({input_shape[0]}, {output_shape[0]}, {kernel}, {stride}, {padding}, {dilation}, {groups})")
def flex_conv_init_(input_shape: LockedShape, output_shape: LockedShape, kernel: tuple[int, ...], stride: tuple[int, ...], padding: tuple[int, ...], dilation: tuple[int, ...], groups: int) -> str:
	return (f"{'FlexConv'}{len(input_shape) - 1}d({input_shape[0]}, {output_shape[0]}, {kernel}, {stride}, {padding}, {dilation}, {groups})")
def maxpool_init_(kernel: tuple[int, ...], stride: tuple[int, ...], padding: tuple[int, ...], dilation: tuple[int, ...]) -> str:
	return nn_(f"MaxPool{len(kernel)}d({kernel}, {stride}, {padding}, {dilation})")
def full_init_(input_shape: LockedShape, output_shape: LockedShape) -> str:
	return nn_(f"Linear({input_shape.get_product()}, {output_shape.get_product()}, bias=True)") 
def relu_init_() -> str:
	return nn_("ReLU()")
def relu6_init_() -> str:
	return nn_("ReLU6()")
def silu_init_() -> str:
	return nn_("SiLU()")
def sigmoid_init_() -> str:
	return nn_("Sigmoid()")
def softmax_init_() -> str:
	return nn_("Softmax(dim=1)")
def batchnorm_init_(input_shape: LockedShape) -> str:
	return nn_(f"BatchNorm{len(input_shape) - 1}d({input_shape[0]})")
def layernorm_init_(input_shape: LockedShape) -> str:
	return nn_(f"LayerNorm({input_shape[-1]})")
def dropout_init_(p: float) -> str:
	return nn_(f"Dropout(p={p})")
def channeldropout_init_(input_shape: LockedShape, p: float) -> str:
	return nn_(f"Dropout{len(input_shape) - 1}d(p={p})")
def glu_init_() -> str:
	return nn_("GLU(dim=1)")
