from __future__ import annotations

from ..shared import LockedShape

from .python_src import *
from .src_backend import SrcBackend

class TorchSrc(SrcBackend):
	def module(self, name: str, init_statements: list[str], forward_args: list[str], forward_statments: list[str]) -> str:
		return torch_module_(name, init_statements, forward_args, forward_statments)
	def view(self, expr: str, shape: LockedShape) -> str:
		return view_(expr, shape)
	def flatten_view(self, expr: str, size: int | LockedShape) -> str:
		return flatten_view_(expr, size)
	def sum(self, *exprs: str) -> str:
		return sum_(*exprs)
	def cat(self, *exprs: str) -> str:
		return cat_(*exprs)
	def conv(self, shape_in: LockedShape, shape_out: LockedShape, kernel: tuple[int, ...], stride: tuple[int, ...], padding: tuple[int, ...], group: int) -> str:
		return conv_(shape_in, shape_out, kernel, stride, padding, group)
	def full(self, shape_in: LockedShape, shape_out: LockedShape) -> str:
		return full_(shape_in, shape_out)
	def relu(self) -> str:
		return relu_()
	def relu6(self) -> str:
		return relu6_()
	def softmax(self) -> str:
		return softmax_()
	def sigmoid(self) -> str:
		return sigmoid_()
	def batch_norm(self, shape_in: LockedShape) -> str:
		return batch_norm_(shape_in)
	def dropout(self, p: float) -> str:
		return dropout_(p)
	def channel_dropout(self, p: float, shape_in: LockedShape) -> str:
		return channel_dropout_(p, shape_in)

def import_torch_() -> str:
	return "import torch"
def torch_(expr: str) -> str:
	return f"torch.{expr}"
def torch_nn_(expr: str) -> str:
	return f"torch.nn.{expr}"
def torch_module_(name: str, init_statements: list[str], forward_args: list[str], forward_statments: list[str]) -> str:
	return concat_lines_(*([import_torch_()] + class_(name, [torch_nn_("Module")], 
		function_("__init__", ["self"],["super().__init__()"] + [import_torch_()] + init_statements) +
		function_("forward", ["self"] + forward_args, [import_torch_()] + forward_statments))))

def view_(expr: str, shape: LockedShape) -> str:
	return f"{expr}.view(-1, {arg_list_(*to_str_list(iter(shape)))})"
def flatten_view_(expr: str, size: int | LockedShape) -> str:
	return f"{expr}.view(-1, {size if isinstance(size, int) else size.get_product()})"

def sum_(*exprs: str) -> str:
	return f"({' + '.join(exprs)})"
def cat_(*exprs: str) -> str:
	return torch_(f"cat({arg_list_(*exprs)}, dim=1)")

def conv_(shape_in: LockedShape, shape_out: LockedShape, kernel: tuple[int, ...], stride: tuple[int, ...], padding: tuple[int, ...], group: int) -> str:
	return torch_nn_(f"Conv{len(shape_in) - 1}d({shape_in[0]}, {shape_out[0]}, {kernel}, {stride}, {padding}, {group}, bias=True, padding_mode='zeros')")
def full_(shape_in: LockedShape, shape_out: LockedShape) -> str:
	return torch_nn_(f"Linear({shape_in[0]}, {shape_out[0]}, bias=True)")

def relu_() -> str:
	return torch_nn_("ReLU()")
def relu6_() -> str:
	return torch_nn_("ReLU6()")
def softmax_() -> str:
	return torch_nn_("Softmax(dim=1)")
def sigmoid_() -> str:
	return torch_nn_("Sigmoid()")

def batch_norm_(shape_in: LockedShape) -> str:
	return torch_nn_(f"BatchNorm{len(shape_in) - 1}d({shape_in[0]})")
def dropout_(p: float) -> str:
	return torch_nn_(f"Dropout(p={p})")
def channel_dropout_(p: float, shape_in: LockedShape) -> str:
	return torch_nn_(f"Dropout{len(shape_in) - 1}d(p={p})")

