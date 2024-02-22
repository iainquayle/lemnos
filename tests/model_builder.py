import unittest   

from src.model.model_builder import _BuildNode, _BuildStack, _BuildTracker, ModelBuilder, BuildIndices
from src.model.model import ModelNode
from src.schema.schema_node import SchemaNode, Transition
from src.schema.transform import ConvParameters 
from src.schema.merge_method import Concat
from src.shared.shape import Bound, LockedShape, Range
from src.shared.index import Index
from copy import copy

s1 = SchemaNode(Bound(), Concat())
s2 = SchemaNode(Bound(), Concat())
shape = LockedShape(1)
m1s1 = ModelNode(Index(), 0, s1, shape, shape, [])
m2s1 = ModelNode(Index(), 0, s1, shape, shape, [])
m1s2 = ModelNode(Index(), 0, s2, shape, shape, [])
m2s2 = ModelNode(Index(), 0, s2, shape, shape, [])

class TestModelBuilder(unittest.TestCase):
	def test_model_builder(self):
		main = SchemaNode(Bound((1, 1), (1, 10)), Concat(), ConvParameters( Range(.1, 2), kernel=2, stride=2))
		input_shape = LockedShape(1, 8)
		output = SchemaNode(Bound((1, 1), (1, 1)), Concat(), None)
		main.add_group(Bound((2, 10)), (output, 0, False))
		main.add_group(Bound((2, 10)), (main, 0, False))
		builder = ModelBuilder([main], [output])
		model = builder.build([input_shape], BuildIndices())
		if model is not None:
			self.assertEqual(model._input_nodes[0].get_schema_node(), main)
			self.assertEqual(model._output_nodes[0].get_schema_node(), output)
		else:
			self.fail()

class TestBuildTrackerBuilding(unittest.TestCase):
	def test_empty(self):
		self.assertFalse(_BuildTracker.build_nodes({}, BuildIndices(), 0))
	def test_single(self):
		input = SchemaNode(Bound((1, 10)), Concat())
		input_shape = LockedShape(5)
		nodes = _BuildTracker.build_nodes({input: input_shape}, BuildIndices(), 0)
		if nodes is not None:
			self.assertEqual(len(nodes), 1)
		else:
			self.fail()
	def test_double(self):
		input = SchemaNode(Bound((1, 10)), Concat())
		input_shape = LockedShape(5)
		output = SchemaNode(Bound((1, 10)), Concat())
		input.add_group(Bound((1, 10)), (output, 0, False))
		nodes = _BuildTracker.build_nodes({input: input_shape}, BuildIndices(), 10)
		if nodes is not None:
			self.assertEqual(len(nodes), 2)
		else: 
			self.fail()
	def test_split_join(self):
		input = SchemaNode(Bound((1, 10)), Concat(), None)
		input_shape = LockedShape(5)
		mid1 = SchemaNode(Bound((1, 10)), Concat(), None)
		mid2 = SchemaNode(Bound((1, 10)), Concat(), None)
		output = SchemaNode(Bound((1, 10)), Concat(), None)
		input.add_group(Bound((1, 10)), (mid1, 0, False), (mid2, 1, False))
		self.assertEqual(input[0][0].get_next(), mid1)
		self.assertEqual(input[0][1].get_next(), mid2)
		mid1.add_group(Bound((1, 10)), (output, 2, False))
		mid2.add_group(Bound((1, 10)), (output, 2, True))
		nodes = _BuildTracker.build_nodes({input: input_shape}, BuildIndices(), 10)
		if nodes is not None:
			self.assertEqual(len(nodes), 4)
		else:
			self.fail()
	def test_looped(self):
		main = SchemaNode(Bound((1, 1), (1, 16)), Concat(), ConvParameters( Range(.1, 2), kernel=2, stride=2))
		input_shape = LockedShape(1, 8)
		output = SchemaNode(Bound((1, 1), (1, 1)), Concat(), None)
		main.add_group(Bound((2, 10)), (output, 0, False))
		main.add_group(Bound((2, 10)), (main, 0, False))
		nodes = _BuildTracker.build_nodes({main: input_shape}, BuildIndices(), 10)
		if nodes is not None:
			self.assertEqual(nodes[0].get_output_shape(), LockedShape(1, 4))
			self.assertEqual(len(nodes), 4)
			for node in nodes[:-1]:
				self.assertEqual(len(node.get_children()), 1)
		else:
			self.fail()
	def test_infinite_loop_stop(self):
		main = SchemaNode(Bound((1, 10)), Concat(), None)
		input_shape = LockedShape(5)
		main.add_group(Bound((1, 10)), (main, 0, False))
		nodes = _BuildTracker.build_nodes({main: input_shape}, BuildIndices(), 10)
		self.assertFalse(nodes)

class TestBuildTrackerUtils(unittest.TestCase):
	def setUp(self) -> None:
		self.node1 = _BuildNode([m1s1], 5)
		self.node2 = _BuildNode([m1s2], 10)
		self.stack1 = _BuildStack([self.node1])
		self.stack2 = _BuildStack([self.node2])
		self.tracker = _BuildTracker(BuildIndices(), 0, {s1: self.stack1, s2: self.stack2})
	def test_pop_min_full(self):
		self.assertEqual(self.tracker._pop_min_node(), (s1, self.node1))
		self.assertEqual(self.tracker._pop_min_node(), (s2, self.node2))
	def test_pop_min_empty(self):
		tracker = _BuildTracker(BuildIndices(), 0)
		self.assertIsNone(tracker._pop_min_node())
	def test_copy(self):
		new_tracker = copy(self.tracker) 
		self.assertEqual(len(new_tracker), len(self.tracker))
		self.assertEqual(new_tracker[s1].peek().get_parents(), self.tracker[s1].peek().get_parents())
		self.assertEqual(new_tracker[s2].peek().get_parents(), self.tracker[s2].peek().get_parents())
		self.assertNotEqual(id(new_tracker[s1]), id(self.tracker[s1]))
		self.assertNotEqual(id(new_tracker[s2]), id(self.tracker[s2]))
		self.assertNotEqual(id(new_tracker[s1].peek()), id(self.tracker[s1].peek()))
		self.assertNotEqual(id(new_tracker[s2].peek()), id(self.tracker[s2].peek()))
	def test_record_valid(self):
		t2_nj = Transition(s2, 1)
		self.assertTrue(self.tracker.record_transition(t2_nj, m1s1))
		self.assertEqual(self.tracker[s2].peek().get_parents(), [m1s1])
		self.assertEqual(self.tracker[s2].peek().get_priority(), 1)
		self.assertEqual(len(self.tracker[s2]), 2)
		t2_j = Transition(s2, 0, True)
		self.assertTrue(self.tracker.record_transition(t2_j, m2s2))
		self.assertTrue(m2s2 in self.tracker[s2].peek().get_parents())
		self.assertEqual(self.tracker[s2].peek().get_priority(), 0)
		self.assertEqual(len(self.tracker[s2]), 2)
		#t3_nj = Transition()
	def test_record_invalid(self):
		t2_j = Transition(s2, 1, True)
		tracker = _BuildTracker(BuildIndices(), 0, {s1: self.stack1, s2: _BuildStack([])})
		self.assertFalse(tracker.record_transition(t2_j, m1s1))
		self.assertEqual(len(tracker[s2]), 0)
		t1_j = Transition(s1, 0, True)
		self.assertFalse(tracker.record_transition(t1_j, m1s1))
	def test_in(self):
		self.assertTrue(s1 in self.tracker)
		self.assertFalse(SchemaNode(Bound(), Concat()) in self.tracker)

class TestBuildNode(unittest.TestCase):
	def setUp(self) -> None:
		self.node = _BuildNode([m1s1], 0)
	def test_available(self):
		self.assertFalse(self.node.available(m2s1))
		self.assertTrue(self.node.available(m1s2))

class TestBuildStack(unittest.TestCase):
	def setUp(self) -> None:
		self.node1 = _BuildNode([m1s1, m1s2], 0)
		self.node2 = _BuildNode([m1s2], 0)
		self.stack = _BuildStack([self.node1, self.node2])
	def test_available(self):
		self.assertEqual(self.stack.get_available(m2s1), self.node2)
	def test_available_none(self):
		self.assertIsNone(self.stack.get_available(m2s2))
	def test_priority(self):
		self.assertEqual(self.stack.get_priority(), 0)
		self.stack.push(_BuildNode([m1s2], 1))
		self.assertEqual(self.stack.get_priority(), 1)
		self.assertEqual(_BuildStack().get_priority(), Transition.get_max_priority() + 1)
	def test_len(self):
		self.assertEqual(len(self.stack), 2)
		self.stack.pop()
		self.assertEqual(len(self.stack), 1)
