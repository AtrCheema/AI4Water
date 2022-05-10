# -*- coding: utf-8 -*-
# Don't know which rights should be reserved  Ather Abbas
from setuptools import setup


import os
fpath = os.path.join(os.getcwd(), "README.md")
if os.path.exists(fpath):
    with open(fpath, "r") as fd:
        long_desc = fd.read()
else:
    long_desc = "https://github.com/AtrCheema/AI4Water"

min_requirements = [
    'numpy',
    'scikit-learn',
    'pandas',
    'matplotlib',
    'scikit-optimize',
    'joblib',
    'requests',
    'easy_mpl>=0.20.4',
    'SeqMetrics>=1.3.3'
    ]

extra_requires = [
'tensorflow', # only if you want to use tensorflow-based models, >=1.15, 2.4 having trouble with see-rnn
'scikit-optimize',  # only if you want to use file hyper_opt.py for hyper-parameter optimization

'h5py<2.11.0', # only if you want to save batches
'xgboost',
'lightgbm',
'catboost',
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

# eda
'seaborn',

# only in some plots
'plotly',
]

tf_requires = ['h5py<2.11.0', 'numpy<=1.19.5', 'easy_mpl', 'tensorflow', 'pandas',
               'matplotlib', 'scikit-learn', 'SeqMetrics>=1.3.3']

tf_hpo_requires = ['h5py<2.11.0', 'numpy<=1.19.5', 'easy_mpl', 'tensorflow', 'pandas',
                   'matplotlib', 'scikit-learn', 'hyperopt', 'scikit-optimize', 'optuna',
                   'SeqMetrics>=1.3.3']

torch_requires = ['h5py', 'numpy', 'easy_mpl>=0.20.4',  'pytorch', 'pandas',
                  'matplotlib', 'scikit-learn', 'SeqMetrics>=1.3.3']

torch_hpo_requires = ['h5py', 'numpy', 'easy_mpl>=0.20.4',  'pytorch', 'pandas',
                  'matplotlib', 'scikit-learn', 'hyperopt', 'scikit-optimize', 'optuna',
                      'SeqMetrics>=1.3.3']

ml_requires = ['numpy', 'matplotlib', 'pandas', 'scikit-learn', 'xgboost', 'catboost',
               'lightgbm', 'easy_mpl>=0.20.4', 'SeqMetrics>=1.3.2']

ml_hpo_requires = ['numpy', 'matplotlib', 'pandas', 'scikit-learn', 'xgboost', 'catboost',
               'lightgbm', 'easy_mpl>=0.20.4', 'hyperopt', 'scikit-optimize', 'optuna',
                   'SeqMetrics>=1.3.3']

hpo_requirements = ['optuna', 'hyperopt', 'scikit-optimize', 'SeqMetrics>=1.3.2']

post_process_requirements = ['lime', 'shap', 'SeqMetrics>=1.3.3']

exp_requirements = ['catboost', 'lightgbm', 'xgboost',
                    'tpot',
                    'optuna', 'hyperopt', 'scikit-optimize',
                    'h5py<2.11.0', 'SeqMetrics>=1.3.3'
                    ]
pre_prcess_requirements = ['netCDF4', 'xarray', 'imageio', 'pyshp', 'SeqMetrics>=1.3.3']

eda_requires = ['seaborn', 'scikit-learn', 'easy_mpl>=0.20.4', 'SeqMetrics>=1.3.3']

all_requirements = min_requirements + extra_requires

setup(

    name='AI4Water',

    version="1.04",

    description='Platform for developing data driven based models for sequential/tabular data',
    long_description=long_desc,
    long_description_content_type="text/markdown",

    url='https://github.com/AtrCheema/AI4Water',

    author='Ather Abbas',
    author_email='ather_abbas786@yahoo.com',

    package_data={'ai4water/datasets': ['arg_busan.csv']},
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
        'Programming Language :: Python :: 3.9',
    ],

    packages=['ai4water',
              'ai4water/models',
              'ai4water/models/tensorflow',
              'ai4water/models/torch',
              'ai4water/hyperopt',
              'ai4water/utils',
              'ai4water/preprocessing',
              'ai4water/preprocessing/transformations',
              'ai4water/preprocessing/dataset',
              'ai4water/postprocessing/',
              'ai4water/postprocessing/SeqMetrics',
              'ai4water/postprocessing/explain',
              'ai4water/postprocessing/interpret',
              'ai4water/postprocessing/visualize',
              'ai4water/datasets',
              'ai4water/et',
              'ai4water/experiments',
              'ai4water/eda'
              ],

    install_requires=min_requirements,

    extras_require={
        'all': extra_requires,
        'hpo': hpo_requirements,
        'post_process': post_process_requirements,
        'exp': exp_requirements,
        'eda': eda_requires,
        'tf': tf_requires,
        'torch': torch_requires,
        'tf_hpo': tf_hpo_requires,
        'torch_hpo_requires': torch_hpo_requires,
        'ml': ml_requires,
        'ml_hpo': ml_hpo_requires,
    }
)
