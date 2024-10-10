from setuptools import setup, find_packages


setup(
	name='lemnos',
	version='0.1',
	packages=find_packages(),
	install_requires=[
		'torch',
		'torchvision',
		'typing_extensions',
	], #move requirements to here...
	author='Iain Quayle',
	#author_email='',
	#description='',
	#long_description='' #open('README.md').read(),
	#long_description_content_type='text/markdown',
	url='github.com/iainquayle/lemnos',
	python_requires='>=3.11',
)

