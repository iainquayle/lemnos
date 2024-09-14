from __future__ import annotations

from ...shared import LockedShape

from abc import ABC as Abstract 

from dataclasses import dataclass

@dataclass
class Statements:
	init_statements: list[str]
	intermediate_forward_statements: list[str]
	output_forward_statement: str

class IdentifierGenerator:
	def __init__(self, namespace: str):
		self._namespace = namespace
		self._identifiers: dict[str, int] = {} 
	def get_identifier(self, name: str) -> str:
		if name not in self._identifiers:
			self._identifiers[name] = len(self._identifiers)
		return f"{self._namespace}_{self._identifiers[name]}"

class Component(Abstract):
	def get_statements(self, input_shape: LockedShape, output_shape: LockedShape, indentifier_generator: IdentifierGenerator) -> Statements:
		raise NotImplementedError(f"must bind a statements generator for {self.__class__.__name__}")

#would be best if both took in a name space generator maybe
#	otherwise, would need to pass out a name for the members, and then that would need to be edited to add the namespace

#the forwards need to take in the the input register? cant just have it done by the formatter anymore if there is a chance that it could get used in multiple spots...
