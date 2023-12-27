class Node(Module):
	def __init__(self, 
			transform: Module = Identity(),
			activation: Module = nn.ReLU(), 
			batch_norm: Module = Identity(), 
			mould_shape: List[int] | Size = [], 
			merge_method: MergeMethod = MergeMethod.SINGLE, 
			node_children: List[Self] = list(),
			node_parents: List[Self] = list(), 
			node_pattern: NodePattern = NodePattern()
			) -> None:
		super().__init__()
		self.index: int = 0
		self.id: int = 0
		self.transform: Module = transform 
		self.activation: Module = activation	 
		self.batch_norm: Module = batch_norm 
		self.mould_shape: Size = Size(mould_shape)
		self.node_children: ModuleList = ModuleList(node_children) 
		self.node_parents: ModuleList = ModuleList(node_parents) 
		self.inputs: List[Tensor] = [] 
		self.merge = MergeMethod.CONCAT.get_function() if merge_method == MergeMethod.SINGLE and len(node_children) > 1 else merge_method.get_function() 
		self.merge_method: MergeMethod = merge_method
		self.node_pattern: NodePattern = node_pattern 
		self.output_shape: Size = Size()
	def forward(self, x: Tensor) -> Tensor | None:
		self.inputs.append(self.mould(x))
		if len(self.inputs) >= len(self.node_parents):
			x = self.activation(self.batch_norm(self.transform(self.merge(self.inputs))))
			y = x 
			for child in self.node_children:
				y = child(x)
			self.inputs = list() 
			return y
		else:
			return None
	def mould(self, x: Tensor) -> Tensor:
		return mould_features(x, self.mould_shape)
	def compile_flat_module_forward(self, source: str, registers: Dict[str, int]) -> str:
		#will need to make some meta objects to keep track of the registers, and whether all childten have been satsified
		return "" 
	def set_node_children(self, node_children: List[Self]) -> Self:
		self.node_children = ModuleList(node_children)
		for child in node_children:
			child.node_parents.append(self)
		return self
	def add_node_child(self, node_child: Self) -> Self:
		self.node_children.append(node_child)
		node_child.node_parents.append(self)
		return self
	#process for node creation:
	#	init
	#		give pattern, first parent, activation, and merge method
	#			consider leaving all of these until the build, so that nothing is missed
	#			could even not init a node until then, and just track the parents
	#	build(duirng expand)
	#		build shape based on parents and pattern
	#			abviously needs to take into accound parents, however, the exact conversion constraints in the pattern may need to be flushed out
	#		from shape, build transform and batch norm
	@staticmethod
	def build(node_pattern: NodePattern, node_data: Node.StackData, index: int) -> None:
		node = Node(node_pattern=node_pattern,
				activation=node_pattern.node_parameters.get_activation(index),
				merge_method=node_pattern.node_parameters.merge_method)
		output_shape_tensors = list() 
		for _, node_parent in node_data.parents.items():
			node_parent.add_node_child(node)
			output_shape_tensors.append(node.forward(torch.zeros(node_parent.output_shape)))
		if isinstance((output_shape_tensor := node.merge(output_shape_tensors)), Tensor):
			node.output_shape = output_shape_tensor.shape
		else:
			raise Exception("output shape list not handled yet, figure out whether this needs to change")
		node.zero_grad()

