#!/usr/bin/env python

from setuptools import setup

setup(name='infogan',
      version='0.1',
      #packages = find_packages(),
      description='glmnet wrapped in rpy',
      author='OpenAI',
      url='https://github.com/DSLituiev/rpyglmnet',
      packages=['.'],
      install_requires = ["prettytensor", "progressbar" ],
      setup_requires = ["prettytensor", ],
      dependency_links = [ "https://github.com/niltonvolpato/python-progressbar" ],
     )
