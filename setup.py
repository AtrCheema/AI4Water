# -*- coding: utf-8 -*-
# some rights may be researved by 2020  Ather Abbas
from setuptools import setup
from version import __version__ as ver
from setuptools.command.install import install

requirements=None

class InstallCommand(install):
    user_options = install.user_options + [
        ('requirements', None, "whether to install with all requirements or with minimum requirements"),
    ]

    def initialize_options(self):
        install.initialize_options(self)
        self.requirements = None

    def finalize_options(self):
        if self.requirements not in [None, 'all']:
            raise ValueError(f"Invalid Value {self.requirements} for `requirements`. Allowed value is `all`.")
        if self.requirements == 'all':
            print(f"Install AI4Water version {ver} with all requirements")
        install.finalize_options(self)

    def run(self):
        global requirements
        requirements = self.requirements
        install.run(self)

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
        'scikit-optimize'
    ]
all_requirements = min_requirements + [
'tensorflow' # only if you want to use tensorflow-based models, >=1.15, 2.4 having trouble with see-rnn
'scikit-optimize',  # only if you want to use file hyper_opt.py for hyper-parameter optimization
'pytorch',  # only if you want to use pytorch-based models
'h5py', # only if you want to save batches
'xgboost',
'EMD-signal',  # for emd transformation
'see-rnn',   # for rnn visualizations
'lightgbm',
'catboost',
'plotly',
'tpot',
'joblib',
# spatial processing
'imageio',
# shapely manually download the wheel file and install
'pyshp',

'optuna',
'hyperopt'
]

setup(
    cmdclass={
        'install': InstallCommand,
    },

    name='AI4Water',

    version=ver,

    description='Platform for developing deep learning based for sequential data',
    long_description=long_desc,
    long_description_content_type="text/markdown",

    url='https://github.com/AtrCheema/AI4Water',

    author='Ather Abbas',
    author_email='ather_abbas786@yahoo.com',

    package_data={'AI4Water/utils/datasets': ['mts_30min.csv', "input_target_u1.csv"]},
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
              'AI4Water/hyper_opt',
              'AI4Water/utils',
              'AI4Water/utils/SeqMetrics',
              'AI4Water/utils/datasets',
              'AI4Water/ETUtil',
              'AI4Water/experiments'
              ],

    install_requires=min_requirements if requirements is None else all_requirements,
)
