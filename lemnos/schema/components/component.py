from __future__ import annotations

from ...shared import LockedShape

from abc import ABC as Abstract 

class Component(Abstract):
	def get_forward_statements(self, input_shape: LockedShape, output_shape: LockedShape) -> list[str]:
		raise NotImplementedError(f"must bind a forward statments generator for {self.__class__.__name__}")
	def get_backward_statements(self, input_shape: LockedShape, output_shape: LockedShape) -> list[str]: 
		raise NotImplementedError(f"must bind a forward statments generator for {self.__class__.__name__}")

#would be best if both took in a name space generator maybe
#	otherwise, would need to pass out a name for the members, and then that would need to be edited to add the namespace

#the forwards need to take in the the input register? cant just have it done by the formatter anymore if there is a chance that it could get used in multiple spots...
