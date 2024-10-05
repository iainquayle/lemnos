from __future__ import annotations


class ID(int):

	def __new__(cls, value: int | ID) -> ID:
		if value < 0:
			raise ValueError("ID must be non-negative")
		return super().__new__(cls, value)

	def __add__(self, other: int | ID) -> ID:
		if (result := super().__add__(other)) < 0:
			raise ValueError("Resulting ID must be non-negative")
		return ID(result)

	def __sub__(self, other: int | ID) -> ID:
		if (result := super().__sub__(other)) < 0:
			raise ValueError("Resulting ID must be non-negative")
		return ID(result)

	def __mul__(self, other: int | ID) -> ID:
		if (result := super().__mul__(other)) < 0:
			raise ValueError("Resulting ID must be non-negative")
		return ID(result)

	def __floordiv__(self, other: int | ID) -> ID:
		if (result := super().__floordiv__(other)) < 0:
			raise ValueError("Resulting ID must be non-negative")
		return ID(result)
