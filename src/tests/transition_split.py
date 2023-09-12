from kontrol.graphs.merger_graph import LayerConstraints, TransitionGroup, SplitTreeNode 

def test():
	print("testing splits nodes")
	t1 = LayerConstraints()
	t2 = LayerConstraints()
	t3 = LayerConstraints()
	
	
	#graph splits,
	#goes thorough 3 and 2, 2 goes to 3
	tg1 = TransitionGroup({t2: True, t3: True})	
	tg2 = TransitionGroup({t3: True})

	stn1 = SplitTreeNode()
	stn1.add(tg1, t2)
	stn2 = SplitTreeNode()
	stn2.add(tg1, t3)
	stn3 = SplitTreeNode()
	stn3.add(tg2, t3, stn1)
	out = stn2.merge(stn3, t3)
	
	print(out)