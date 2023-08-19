from kontrol.transitions import Transition, TransitionGroup, SplitTreeNode 

def test():
	print("testing splits nodes")
	t1 = Transition()
	t2 = Transition()
	t3 = Transition()
	
	tg1 = TransitionGroup({t2: True, t3: True})	
	tg2 = TransitionGroup({t3: True})

	stn1 = SplitTreeNode()
	stn1.add(tg1, t2)
	stn2 = SplitTreeNode()
	stn2.add(tg1, t3)
	stn3 = SplitTreeNode()
	stn3.add(tg1, t3, stn1)
	stn2.merge(stn3)
	
	print(stn2)