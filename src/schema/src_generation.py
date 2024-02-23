from __future__ import annotations

from abc import ABC as Abstract, abstractmethod
from src.shared.shape import LockedShape

from typing import Tuple, List, Iterable, Any

#src generation responsibilities
#	model
#		registers
#		assigning 
#	schema node
#		eval
#		view
#	component
#		init
def _to_str_list(iterable: Iterable[Any]) -> List[str]:
	return [str(i) for i in iterable]

def arg_list_(*exprs: str) -> str:
	return f"{', '.join(exprs)}"
def assign_(register: str, value: str) -> str:
	return f"{register} = {value}"
def self_(register: str) -> str:
	return f"self.{register}"
def return_(*exprs: str) -> str:
	return f"return {arg_list_(*exprs)}"
def call_(function: str, *exprs: str) -> str:
	return f"{function}({arg_list_(*exprs)})"
def function_(name: str, args: List[str], statements: List[str]) -> List[str]:
	return [f"def {name}({arg_list_(*args)}):"] + [f"\t{statement}" for statement in statements]
def class_(name: str, super_classes: List[str], members: List[str]) -> List[str]:
	return [f"class {name}({arg_list_(*super_classes)}):"] + [f"\t{member}" for member in members]
def concat_lines_(*lines: str) -> str:
	return "\n".join(lines)

def import_torch_() -> str:
	return "import torch"
def torch_(expr: str) -> str:
	return f"torch.{expr}"
def torch_nn_(expr: str) -> str:
	return f"torch.nn.{expr}"
def torch_module_(name: str, init_statements: List[str], forward_args: List[str], forward_statments: List[str]) -> str:
	return concat_lines_(*([import_torch_()] + class_(name, [torch_nn_("Module")], 
		function_("__init__", ["self"],["super().__init__()"] + [import_torch_()] + init_statements) +
		function_("forward", ["self"] + forward_args, [import_torch_()] + forward_statments))))

def view_(expr: str, shape: LockedShape) -> str:
	return f"{expr}.view(-1, {arg_list_(*_to_str_list(iter(shape)))})"
def flatten_view_(expr: str, size: int | LockedShape) -> str:
	return f"{expr}.view(-1, {size if isinstance(size, int) else size.get_product()})"

def sum_(*exprs: str) -> str:
	return f"({' + '.join(exprs)})"
def cat_(*exprs: str) -> str:
	return torch_(f"cat({arg_list_(*exprs)}, dim=1)")

def conv_(shape_in: LockedShape, shape_out: LockedShape, kernel: Tuple[int, ...], stride: Tuple[int, ...], padding: Tuple[int, ...], group: int) -> str:
	return torch_nn_(f"Conv{len(shape_in) - 1}d({shape_in[0]}, {shape_out[0]}, {kernel}, {stride}, {padding}, {group}, bias=True, padding_mode='zeros')")

def relu_() -> str:
	return torch_nn_("ReLU()")
def relu6_() -> str:
	return torch_nn_("ReLU6()")
def softmax_() -> str:
	return torch_nn_("Softmax(dim=1)")

def batch_norm_(shape_in: LockedShape) -> str:
	return torch_nn_(f"BatchNorm{len(shape_in) - 1}d({shape_in[0]})")
def dropout_(p: float) -> str:
	return torch_nn_(f"Dropout(p={p})")
def channel_dropout_(p: float, shape_in: LockedShape) -> str:
	return torch_nn_(f"Dropout{len(shape_in) - 1}d(p={p})")

