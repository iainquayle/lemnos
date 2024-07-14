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
