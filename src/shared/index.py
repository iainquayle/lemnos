class Index:
	MAX_INDEX = 2**16 -1
	def __init__(self, index: int =0) -> None:
		self.index = index
	def to_int(self, mod_factor: int) -> int:
		return self.index % mod_factor if mod_factor > 0 else 0
	def as_ratio(self) -> float:
		return self.index / Index.MAX_INDEX
