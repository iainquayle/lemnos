from __future__ import annotations

from ...shared import LockedShape

from ..python_formats import *
from ..target_components import TargetComponents

class TorchComponents(TargetComponents):
	def view(self, expr: str, shape: LockedShape) -> str:
		return f"{expr}.view(-1, {arg_list_(*to_str_list(iter(shape)))})"
	def flatten_view(self, expr: str, size: int | LockedShape) -> str:
		return f"{expr}.view(-1, {size if isinstance(size, int) else size.get_product()})"
	def sum(self, *exprs: str) -> str:
		return f"({' + '.join(exprs)})"
	def cat(self, *exprs: str) -> str:
		if len(exprs) == 1:
			return exprs[0]
		return torch_(f"cat({arg_list_(*exprs)}, dim=1)")
	def conv(self, shape_in: LockedShape, shape_out: LockedShape, kernel: tuple[int, ...], stride: tuple[int, ...], padding: tuple[int, ...], group: int) -> str:
		return torch_nn_(f"Conv{len(shape_in) - 1}d({shape_in[0]}, {shape_out[0]}, {kernel}, {stride}, {padding}, {group}, bias=True, padding_mode='zeros')")
	def full(self, shape_in: LockedShape, shape_out: LockedShape) -> str:
		return torch_nn_(f"Linear({shape_in[0]}, {shape_out[0]}, bias=True)")
	def relu(self) -> str:
		return torch_nn_("ReLU()")
	def relu6(self) -> str:
		return torch_nn_("ReLU6()")
	def softmax(self) -> str:
		return torch_nn_("Softmax(dim=1)")
	def sigmoid(self) -> str:
		return torch_nn_("Sigmoid()")
	def batch_norm(self, shape_in: LockedShape) -> str:
		return torch_nn_(f"BatchNorm{len(shape_in) - 1}d({shape_in[0]})")
	def dropout(self, p: float) -> str:
		return torch_nn_(f"Dropout(p={p})")
	def channel_dropout(self, p: float, shape_in: LockedShape) -> str:
		return torch_nn_(f"Dropout{len(shape_in) - 1}d(p={p})")

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
