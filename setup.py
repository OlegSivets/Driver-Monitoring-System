from setuptools import setup
 
setup(
   name='my_package',
   version='1.0',
   description='A useful module',
   author='Author Name',
   author_email='author@gmail.com',
   packages=['my_package'],  #same as name
   install_requires=['numpy', 'pandas'], #external packages as dependencies
)