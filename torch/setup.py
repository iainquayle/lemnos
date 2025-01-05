from setuptools import setup, find_packages

lemnos_address = 'https://github/iainquayle/lemnos.git'
lemnos_name = 'lemnos'

setup(
	name='lemnos-torch',
	version='0.1',
	packages=find_packages(),
	install_requires=[
		'torch',
		'torchvision',
		'typing_extensions',
		f'{lemnos_name} @ {lemnos_address}#egg={lemnos_name}&subdirectory=core',
	], 
	dependency_links=[
		lemnos_address
	],
	author='Iain Quayle',
	#author_email='',
	#description='',
	#long_description='' #open('README.md').read(),
	#long_description_content_type='text/markdown',
	url='github.com/iainquayle/lemnos',
	python_requires='>=3.11',
)

