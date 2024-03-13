import unittest   

from src.schema.schema import _BuildNode, _BuildStack, _BuildTracker
from src.model.model import ModelNode
from src.schema import Schema, BreedIndices, SchemaNode, Transition, Conv, Concat, Sum, ReLU, BatchNormalization, JoinType
from src.shared import ShapeBound, LockedShape, Index

from typing import List, Tuple
from copy import copy

class TestSchema(unittest.TestCase):
	def test_model_builder(self):
		main = SchemaNode(ShapeBound((1, 1), (1, 10)), Concat(), Conv( (.1, 2), kernel=2, stride=2))
		input_shape = LockedShape(1, 8)
		output = SchemaNode(ShapeBound((1, 1), (1, 1)), Concat(), None)
		main.add_group((output, 0, JoinType.NEW))
		main.add_group((main, 0, JoinType.NEW))
		builder = Schema([main], [output])
		model = builder.build([input_shape], BreedIndices())
		if model is not None:
			self.assertEqual(model._input_nodes[0].get_schema_node(), main)
			self.assertEqual(model._output_nodes[0].get_schema_node(), output)
		else:
			self.fail()

class TestBuildTrackerBuilding(unittest.TestCase):
	def test_empty(self):
		self.assertFalse(_BuildTracker.build_nodes({}, BreedIndices(), 0))
	def test_single(self):
		input = SchemaNode(ShapeBound((1, 10)), Concat())
		input_shape = LockedShape(5)
		nodes = _BuildTracker.build_nodes({input: input_shape}, BreedIndices(), 0)
		if nodes is not None:
			self.assertEqual(len(nodes), 1)
		else:
			self.fail()
	def test_double(self):
		input = SchemaNode(ShapeBound((1, 10)), Concat())
		input_shape = LockedShape(5)
		output = SchemaNode(ShapeBound((1, 10)), Concat())
		input.add_group((output, 0, JoinType.NEW))
		nodes = _BuildTracker.build_nodes({input: input_shape}, BreedIndices(), 10)
		if nodes is not None:
			self.assertEqual(len(nodes), 2)
		else: 
			self.fail()
	def test_split_join(self):
		input = SchemaNode(ShapeBound((1, 10)), Concat(), None)
		input_shape = LockedShape(5)
		mid1 = SchemaNode(ShapeBound((1, 10)), Concat(), None)
		mid2 = SchemaNode(ShapeBound((1, 10)), Concat(), None)
		output = SchemaNode(ShapeBound((1, 10)), Concat(), None)
		input.add_group((mid1, 0, JoinType.NEW), (mid2, 1, JoinType.NEW))
		self.assertEqual(input[0][0].get_next(), mid1)
		self.assertEqual(input[0][1].get_next(), mid2)
		mid1.add_group((output, 2, JoinType.NEW))
		mid2.add_group((output, 2, JoinType.EXISTING))
		nodes = _BuildTracker.build_nodes({input: input_shape}, BreedIndices(), 10)
		if nodes is not None:
			self.assertEqual(len(nodes), 4)
		else:
			self.fail()
	def test_looped(self):
		main = SchemaNode(ShapeBound((1, 1), (1, 16)), Concat(), Conv( (.1, 2), kernel=2, stride=2))
		input_shape = LockedShape(1, 8)
		output = SchemaNode(ShapeBound((1, 1), (1, 1)), Concat(), None)
		main.add_group((output, 0, JoinType.NEW))
		main.add_group((main, 0, JoinType.NEW))
		nodes = _BuildTracker.build_nodes({main: input_shape}, BreedIndices(), 10)
		if nodes is not None:
			self.assertEqual(nodes[0].get_output_shape(), LockedShape(1, 4))
			self.assertEqual(len(nodes), 4)
			for node in nodes[:-1]:
				self.assertEqual(len(node.get_children()), 1)
		else:
			self.fail()
	def test_split_looped(self):
		main = SchemaNode( ShapeBound((1, 1), (1, 8)), Sum())
		split_1 = SchemaNode( ShapeBound((1, 1), (1, 8)), 
			Concat(), 
			Conv((.1, 2.0), kernel=2, stride=2),
			ReLU(), 
			BatchNormalization())
		split_2 = SchemaNode( ShapeBound((1, 10), (1, 8)), 
			Concat(), 
			Conv((.1, 2.0), kernel=2, stride=2),
			ReLU(), 
			BatchNormalization())
		end_node = SchemaNode( ShapeBound((1, 1), (1, 1)), Concat())
		main.add_group( (split_1, 0, JoinType.NEW), (split_2, 1, JoinType.NEW))
		split_1.add_group( (main, 2, JoinType.NEW))
		split_2.add_group( (main, 2, JoinType.EXISTING))
		main.add_group( (end_node, 0, JoinType.NEW))
		input_shape = LockedShape(1, 8)
		nodes = _BuildTracker.build_nodes({main: input_shape}, BreedIndices(), 10)
		if nodes is not None:
			self.assertEqual(len(nodes), 11)
			id_set = set()
			for node in nodes:
				self.assertNotIn(node.get_id(), id_set)
				id_set.add(node.get_id())
		else:
			self.fail()
	def test_infinite_loop_stop(self):
		main = SchemaNode(ShapeBound((1, 10)), Concat(), None)
		input_shape = LockedShape(5)
		main.add_group((main, 0, JoinType.NEW))
		nodes = _BuildTracker.build_nodes({main: input_shape}, BreedIndices(), 10)
		self.assertFalse(nodes)

class TestBuildTrackerUtils(unittest.TestCase):
	def setUp(self) -> None:
		self.node1 = _BuildNode([m1s1], 5)
		self.node2 = _BuildNode([m1s2], 10)
		self.stack1 = _BuildStack([self.node1])
		self.stack2 = _BuildStack([self.node2])
		self.tracker = _BuildTracker(0, {s1: self.stack1, s2: self.stack2}, 0)
	def test_pop_min_full(self):
		self.assertEqual(self.tracker._pop_min_node(), (s1, self.node1))
		self.assertEqual(self.tracker._pop_min_node(), (s2, self.node2))
	def test_pop_min_empty(self):
		tracker = _BuildTracker(0, dict(), 0)
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
		t2_j = Transition(s2, 0, JoinType.EXISTING)
		self.assertTrue(self.tracker.record_transition(t2_j, m2s2))
		self.assertTrue(m2s2 in self.tracker[s2].peek().get_parents())
		self.assertEqual(self.tracker[s2].peek().get_priority(), 0)
		self.assertEqual(len(self.tracker[s2]), 2)
		#t3_nj = Transition()
	def test_record_invalid(self):
		t2_j = Transition(s2, 1, JoinType.EXISTING)
		tracker = _BuildTracker(0, {s1: self.stack1, s2: _BuildStack([])}, 0)
		self.assertFalse(tracker.record_transition(t2_j, m1s1))
		self.assertEqual(len(tracker[s2]), 0)
		t1_j = Transition(s1, 0, JoinType.EXISTING)
		self.assertFalse(tracker.record_transition(t1_j, m1s1))
	def test_in(self):
		self.assertTrue(s1 in self.tracker)
		self.assertFalse(SchemaNode(ShapeBound(), Concat()) in self.tracker)

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

class TestBreedIndices(unittest.TestCase):
	def setUp(self) -> None:
		self.sequences: List[List[Tuple[Index, SchemaNode, LockedShape]]] = [
			[(Index(1), s2, LockedShape(1, 1)), (Index(2), s2, LockedShape(1, 2))],
			[(Index(3), s1, LockedShape(1, 1)), (Index(4), s1, LockedShape(1, 2))],
		]
	def test_no_mutate(self):
		indices = BreedIndices(0, 0, self.sequences)
		index, _ = indices.get_index(0, 0, s2, LockedShape(1, 1))
		self.assertEqual(index, Index(1))
		index, _ = indices.get_index(0, 0, s2, LockedShape(1, 2))
		self.assertEqual(index, Index(2))
		index, sequence = indices.get_index(0, 0, s1, LockedShape(1, 1))
		self.assertEqual(index, Index(3))
		self.assertEqual(sequence, 1)
	def test_random_index(self):
		indices = BreedIndices(0, 1, self.sequences)
		index, _ = indices.get_index(0, 0, s2, LockedShape(1, 1))
		#could technically fail sometimes, but very unlikely
		for i in range(1, 5):
			self.assertNotEqual(index, Index(i))
	def test_sequence_switch(self):
		indices = BreedIndices(1, 0, self.sequences + [[(Index(5), s2, LockedShape(1, 1))]])
		index, _ = indices.get_index(0, 0, s2, LockedShape(1, 1))
		self.assertEqual(index, Index(5))
		
		
