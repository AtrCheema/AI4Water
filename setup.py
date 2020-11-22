# -*- coding: utf-8 -*-
# Copyright (C) 2020  Ather Abbas
from setuptools import setup


with open("README.md", "r") as fd:
    long_desc = fd.read()

with open('version.py') as fv:
    exec(fv.read())

setup(
    name='dl4seq',

    version=1.0,

    description='Platform for developing deep learning based for sequential data',
    long_description=long_desc,
    long_description_content_type="text/markdown",

    url='https://github.com/AtrCheema/dl4seq',

    author='Ather Abbas',
    author_email='ather_abbas786@yahoo.com',

    license='GPLv3',

    classifiers=[
        'Development Status :: 4 - Beta',

        'Natural Language :: English',

        'Intended Audience :: Science/Research',
        'Intended Audience :: Developers',

        'Operating System :: MacOS',
        'Operating System :: Microsoft :: Windows',
        'Operating System :: POSIX :: Linux',


        'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',

        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: Implementation :: CPython'
    ],

    packages=['dl4seq'],

    install_requires=[
        'numpy',
        'TSErrors',
        'seaborn',
        'scikit-learn',
        'pandas',
        'matplotlib'
    ],
)
