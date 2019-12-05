#!/usr/bin/env python

import os
from setuptools import setup, Extension

# Load the __version__ variable without importing the package already
exec(open('bunyip/version.py').read())

# Get dependencies
with open('requirements.txt') as f:
    install_requires = f.read().splitlines()

setup(name='bunyip',
      version=__version__,
      description="A package to fit eclipsing binary light curves.",
    #   long_description=open('README.rst').read(),
      license='MIT',
      package_dir={
            'bunyip': 'bunyip',},
      packages=['bunyip'],
      install_requires=install_requires,
      url='https://github.com/danhey/bunyip',
      include_package_data=True,
)