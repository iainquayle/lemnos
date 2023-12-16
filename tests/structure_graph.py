import unittest

from src.model_structures.graph import Node
from src.model_structures.commons import MergeMethod
import torch.nn as nn
import torch.optim as optim
import torch

class TestGraph(unittest.TestCase):
	def setUp(self) -> None:
		pass
	#def tearDown(self) -> None:
	#	self.setUp()


	def test_net_function(self):
		n1 = Node(transform=nn.Linear(2, 2), mould_shape=[2], activation=nn.Sigmoid(), merge_method=MergeMethod.SINGLE)
		n2 = Node(transform=nn.Linear(2, 2), mould_shape=[2], activation=nn.Sigmoid(), merge_method=MergeMethod.SINGLE)
		n3 = Node(transform=nn.Linear(2, 2), mould_shape=[2], activation=nn.Sigmoid(), merge_method=MergeMethod.SINGLE)
		n4 = Node(transform=nn.Linear(4, 2), mould_shape=[2], activation=nn.Sigmoid(), merge_method=MergeMethod.CONCAT)
		n5 = Node(transform=nn.Linear(2, 2), mould_shape=[2], activation=nn.Sigmoid(), merge_method=MergeMethod.ADD)

		n1.node_children = nn.ModuleList([n2, n3, n5]) 
		n2.node_children = nn.ModuleList([n4]) 
		n2.node_parents = nn.ModuleList([n1]) 
		n3.node_children = nn.ModuleList([n4]) 
		n3.node_parents = nn.ModuleList([n1]) 
		n4.node_children = nn.ModuleList([n5])
		n4.node_parents = nn.ModuleList([n2, n3])
		n5.node_parents = nn.ModuleList([n1, n4])

		t_in = torch.Tensor([[1.0, 0.0], [1.0, 1.0]])
		t_out = torch.Tensor([[1.0, 0.0], [0.0, 1.0]])

		criterion = nn.CrossEntropyLoss()
		optimizer = optim.Adam(n1.parameters(), lr=0.1)

		for i in range(2000):
			optimizer.zero_grad()
			output = n1(t_in)
			loss = criterion(output, t_out)
			loss.backward()
			optimizer.step()

		output = n1(t_in)
		self.assertGreaterEqual(output[0][0].item(), 0.8, "output[0][0] should be less than 0.8")
		self.assertLessEqual(output[0][1].item(), 0.2, "output[0][1] should be less than 0.2")
		self.assertLessEqual(output[1][0].item(), 0.2, "output[1][0] should be less than 0.2")
		self.assertGreaterEqual(output[1][1].item(), 0.8, "output[1][1] should be less than 0.8")

	def test_set_children(self):
		m1 = Node()
		m2 = Node()
		m1.set_node_children([m2])
		self.assertEqual(m1.node_children[0], m2, "mod1.node_children[0] should be mod2")
		self.assertIn(m1, m2.node_parents, "mod1.node_children[0] should be mod2")
		
if __name__ == '__main__':
	unittest.main()
