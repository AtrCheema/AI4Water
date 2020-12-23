# -*- coding: utf-8 -*-
# some rights may be researved by 2020  Ather Abbas
from setuptools import setup
from version import __version__ as ver

with open("README.md", "r") as fd:
    long_desc = fd.read()

with open('version.py') as fv:
    exec(fv.read())

setup(
    name='dl4seq',

    version=ver,

    description='Platform for developing deep learning based for sequential data',
    long_description=long_desc,
    long_description_content_type="text/markdown",

    url='https://github.com/AtrCheema/dl4seq',

    author='Ather Abbas',
    author_email='ather_abbas786@yahoo.com',

    package_data={'dl4seq/data': ['data_30min.csv', "input_target_u1.csv"]},
    include_package_data=True,

    classifiers=[
        'Development Status :: 4 - Beta',

        'Natural Language :: English',

        'Intended Audience :: Science/Research',
        'Intended Audience :: Developers',

        'Operating System :: Microsoft :: Windows',
        'Operating System :: POSIX :: Linux',

        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
    ],

    packages=['dl4seq', 'dl4seq/models', 'dl4seq/utils', 'dl4seq/data'],

    install_requires=[
        'numpy',
        'seaborn',
        'scikit-learn',
        'pandas',
        'matplotlib',
        'scikit-optimize',
        'TSErrors',
    ],
)
