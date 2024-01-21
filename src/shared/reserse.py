from __future__ import annotations 

from abc import ABC as Abstract, abstractmethod

#TODO: make a reverse index interface
#	would be nice to have it require len and getitem, then have the reverse index be standard
#	have to figure out how to make proper typing work for it
#	maybe just overkill though
class Reverse(Abstract):
	pass
