# -*- coding: utf-8 -*-
# some rights may be researved by 2020  Ather Abbas
from setuptools import setup
from version import __version__ as ver


with open("README.md", "r") as fd:
    long_desc = fd.read()

with open('version.py') as fv:
    exec(fv.read())

min_requirements = [
    'numpy',
    'seaborn',
    'scikit-learn',
    'pandas',
    'matplotlib',
    'scikit-optimize',
    'joblib',
    'requests',
    ]

extra_requires = [
'tensorflow', # only if you want to use tensorflow-based models, >=1.15, 2.4 having trouble with see-rnn
'scikit-optimize',  # only if you want to use file hyper_opt.py for hyper-parameter optimization
#'pytorch',  # only if you want to use pytorch-based models
'h5py<2.11.0', # only if you want to save batches
'xgboost',
'EMD-signal',  # for emd transformation
'see-rnn',   # for rnn visualizations
'lightgbm',
'catboost',
'plotly',
'tpot',
# spatial processing
'imageio',
# shapely manually download the wheel file and install
'pyshp',

'optuna',
'hyperopt',

# for reading data
'netCDF4',
 'xarray',

# for jsonizing
'wrapt',
]

all_requirements = min_requirements + extra_requires

setup(

    name='AI4Water',

    version=ver,

    description='Platform for developing deep learning based for sequential data',
    long_description=long_desc,
    long_description_content_type="text/markdown",

    url='https://github.com/AtrCheema/AI4Water',

    author='Ather Abbas',
    author_email='ather_abbas786@yahoo.com',

    package_data={'ai4water/datasets': ['arg_busan.csv', "input_target_u1.csv"]},
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
        'Programming Language :: Python :: 3.8',
    ],

    packages=['ai4water',
              'ai4water/models',
              'ai4water/hyper_opt',
              'ai4water/utils',
              'ai4water/pre_processing',
              'ai4water/post_processing/',
              'ai4water/post_processing/SeqMetrics',
              'ai4water/post_processing/explain',
              'ai4water/post_processing/interpret',
              'ai4water/post_processing/visualize',
              'ai4water/datasets',
              'ai4water/ETUtil',
              'ai4water/experiments',
              'ai4water/eda'
              ],

    install_requires=min_requirements,

    extras_require={'all': extra_requires}
)
