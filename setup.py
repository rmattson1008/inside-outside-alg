from setuptools import setup

setup(
   name='pcfg',
   version='1.0',
   description='unsupervised Training of PCFGs',
   author='Rachel Mattson',
   author_email='foomail@foo.com',
   packages=['src'],  # would be the same as name
   install_requires=['numpy', 'nltk', 'tqdm', 'pytest', 'nptyping'], #external packages acting as dependencies
)