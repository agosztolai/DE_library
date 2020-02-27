#!/usr/bin/env python
# -*- coding: UTF-8 -*-

from setuptools import setup, find_packages

setup(
        name = 'DE_library',
        version = '1.0',
        packages=find_packages(),
        install_requires=['numpy', 
                          'scipy', 
                          'matplotlib'],
        include_package_data = True
      )
