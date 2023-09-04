from kontrol.transitions import Transition, TransitionGroup, SplitTreeNode 


def test():
	t1 = Transition()
	t2 = Transition()
	
	tg1 = TransitionGroup({t1: False, t2: True})
	
	print(tg1)
	print(tg1.transitions[t1].optional)