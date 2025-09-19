from setuptools import setup
setup(name='Gf_Module',
      version='0.1',
      description='Module for building electronic Greens functions',
      url='',
      author='Aleksander Bach Lorentzen',
      author_email='abalo@dtu.dk',
      license='MIT',
      packages=['Gf_Module'],
      zip_safe=False,
      install_requires= ["numpy", "matplotlib", "tqdm", "Block_matrices", "scipy"])

