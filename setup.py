# -*- coding: utf-8 -*-
# some rights may be researved by 2020  Ather Abbas
from setuptools import setup
from version import __version__ as ver

with open("README.md", "r") as fd:
    long_desc = fd.read()

with open('version.py') as fv:
    exec(fv.read())

setup(
    name='AI4Water',

    version=ver,

    description='Platform for developing deep learning based for sequential data',
    long_description=long_desc,
    long_description_content_type="text/markdown",

    url='https://github.com/AtrCheema/AI4Water',

    author='Ather Abbas',
    author_email='ather_abbas786@yahoo.com',

    package_data={'AI4Water/data': ['data_30min.csv', "input_target_u1.csv"]},
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

    packages=['AI4Water',
              'AI4Water/models',
              'AI4Water/data',
              'AI4Water/hyper_opt',
              'AI4Water/utils',
              'AI4Water/utils/SeqMetrics',
              'AI4Water/utils/datasets',
              'AI4Water/ETUtil',
              'AI4Water/experiments'
              ],

    install_requires=[
        'numpy',
        'seaborn',
        'scikit-learn',
        'pandas',
        'matplotlib',
        'scikit-optimize'
    ],
)
