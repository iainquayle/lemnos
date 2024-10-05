from __future__ import annotations

from typing import Iterable, Any


def to_str_list(iterable: Iterable[Any]) -> list[str]:
	return [str(i) for i in iterable]


def concat_lines_(*lines: str) -> str:
	output = "" 
	for i, line in enumerate(lines):
		output += f"{line}{'	' * 2}# ln: {i + 1}\n"
	return output 


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


def function_(name: str, args: list[str], statements: list[str]) -> list[str]:
	return [f"def {name}({arg_list_(*args)}):"] + [f"\t{statement}" for statement in statements]


def class_(name: str, super_classes: list[str], members: list[str]) -> list[str]:
	return [f"class {name}({arg_list_(*super_classes)}):"] + [f"\t{member}" for member in members]


def print_(expr: str) -> str:
	return f"print({expr})"


def if_(condition: str, true: list[str], false: list[str] | None = None) -> list[str]:
	return [f"if {condition}:" ] + [f"\t{statement}" for statement in true] + ([f"else:"] + [f"\t{statement}" for statement in false] if false is not None else [])
