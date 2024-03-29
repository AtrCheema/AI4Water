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


skopt_version = 'scikit-optimize'
seq_met_ver = 'SeqMetrics>=1.3.3'
easy_mpl_ver = 'easy_mpl[all]>=0.21.3'

min_requirements = [
    'scikit-learn',
    'pandas',

    easy_mpl_ver,
    seq_met_ver,
    ]

extra_requires = [
'tensorflow', # only if you want to use tensorflow-based models, >=1.15, 2.4 having trouble with see-rnn
skopt_version,  # only if you want to use file hyper_opt.py for hyper-parameter optimization

'h5py', # only if you want to save batches
'xgboost',
'lightgbm',
'catboost',
'tpot',
# spatial processing
'imageio',
# shapely manually download the wheel file and install
'pyshp',

'optuna<=2.10.1',
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

    'requests',
]

tf_requires = ['h5py<2.11.0', 'numpy<=1.19.5', easy_mpl_ver, 'tensorflow==1.15', 'pandas',
               'matplotlib', 'scikit-learn', seq_met_ver, 'AttentionLSTM']

tf2_requires = ['h5py', 'numpy', easy_mpl_ver, 'tensorflow<=2.7', 'pandas',
               'matplotlib', 'scikit-learn', seq_met_ver, 'AttentionLSTM']

tf_hpo_requires = ['h5py<2.11.0', 'numpy<=1.19.5', easy_mpl_ver, 'tensorflow==1.15', 'pandas',
                   'matplotlib', 'scikit-learn', 'hyperopt', skopt_version, 'optuna<=2.10.1',
                   seq_met_ver, 'AttentionLSTM']

torch_requires = ['h5py', 'numpy', easy_mpl_ver,  'pytorch', 'pandas',
                  'matplotlib', 'scikit-learn', seq_met_ver]

torch_hpo_requires = ['h5py', 'numpy', easy_mpl_ver,  'pytorch', 'pandas',
                  'matplotlib', 'scikit-learn', 'hyperopt', skopt_version, 'optuna<=2.10.1',
                      seq_met_ver]

ml_requires = ['numpy', 'matplotlib', 'pandas', 'scikit-learn', 'xgboost', 'catboost',
               'lightgbm', easy_mpl_ver, seq_met_ver]

ml_hpo_requires = ['numpy', 'matplotlib', 'pandas', 'scikit-learn', 'xgboost', 'catboost',
               'lightgbm', easy_mpl_ver, 'hyperopt', skopt_version, 'optuna<=2.10.1',
                   seq_met_ver]

hpo_requirements = ['optuna<=2.10.1', 'hyperopt', skopt_version, seq_met_ver]

post_process_requirements = ['lime', 'shap', seq_met_ver]

exp_requirements = ['catboost', 'lightgbm', 'xgboost',
                    'tpot',
                    'optuna<=2.10.1', 'hyperopt', skopt_version,
                    'h5py<2.11.0', seq_met_ver, easy_mpl_ver
                    ]
pre_prcess_requirements = ['netCDF4', 'xarray', 'imageio', 'pyshp', seq_met_ver, easy_mpl_ver]

eda_requires = ['seaborn', 'scikit-learn', easy_mpl_ver, seq_met_ver]

all_requirements = min_requirements + extra_requires

setup(

    name='AI4Water',

    version="1.06",

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
              'ai4water/models/_tensorflow',
              'ai4water/models/_torch',
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
              'ai4water/datasets/water_quality',
              'ai4water/et',
              'ai4water/experiments',
              'ai4water/eda',
              'ai4water/envs'
              ],

    install_requires=min_requirements,

    extras_require={
        'all': extra_requires,
        'hpo': hpo_requirements,
        'post_process': post_process_requirements,
        'exp': exp_requirements,
        'eda': eda_requires,
        'tf': tf_requires,
        'tf2': tf2_requires,
        'torch': torch_requires,
        'tf_hpo': tf_hpo_requires,
        'torch_hpo_requires': torch_hpo_requires,
        'ml': ml_requires,
        'ml_hpo': ml_hpo_requires,
    }
)
