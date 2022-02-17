Installation
*************

using github link
::
    python -m pip install git+https://github.com/AtrCheema/AI4Water.git

or using setup file, go to folder where repo is downloaded
::
    python setup.py install

The latest code however (possibly with less bugs and more features) can be insalled from ``dev`` branch instead
::
    python -m pip install git+https://github.com/AtrCheema/AI4Water.git@dev

To install the latest branch (`dev`) with all requirements use ``all`` keyword
::
    python -m pip install "AI4Water[all] @ git+https://github.com/AtrCheema/AI4Water.git@dev"

installation options
=====================
In above command ``all`` keyword will install all the dependencies. You can choose the dependencies of particular sub-module
by using the specific keyword. Following keywords are available

 - ``hpo`` if you want hyperparameter optimization
 - ``post_process`` if you want postprocessing
 - ``exp`` for experiments sub-module
 - ``eda`` for exploratory data analysis sub-module