from setuptools import setup, find_packages

lemnos_address = 'https://github.com/iainquayle/lemnos.git'
lemnos_branch = 'master'
lemnos_name = 'lemnos'
lemnos_dir = 'core'

setup(
	name='lemnos_torch',
	version='0.1',
	packages=find_packages(),
	install_requires=[
		'torch',
		'torchvision',
		'typing_extensions',
		f'{lemnos_name} @ git+{lemnos_address}@{lemnos_branch}#egg={lemnos_name}&subdirectory={lemnos_dir}',
	], 
	#dependency_links=[
	#	lemnos_address
	#],
	author='Iain Quayle',
	#author_email='',
	#description='',
	#long_description='' #open('README.md').read(),
	#long_description_content_type='text/markdown',
	url='github.com/iainquayle/lemnos',
	python_requires='>=3.11',
)

