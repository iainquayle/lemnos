src_class = "class Test:\n\tdef __init__(self):\n\t\tself.a = 1\n\tdef test(self):\n\t\treturn self.a + 1\n"

print(src_class)

exec(src_class)
test = eval("Test()") 
print(f"basic: {test.test()}")

exec(src_class)
test = eval("Test()") 
print(f"redecl: {test.test()}")

class Wrapper:
	def __init__(self, src_class: str) -> None:
		self.src_class = src_class
	def get(self):
		exec(self.src_class)
		return eval("Test()")
wrapper = Wrapper(src_class)
test = wrapper.get()
print(f"wrapped: {test.test()}")


