Installation
*************

using pip
=========
The most easy way to install ai4water is using ``pip``
::
    pip install ai4water

However, if you are interested in using only specific module of ai4water, you can
choose to install dependencies related to that module only. For example
to use only machine learning based models use can use ``ml`` option as Following
::
    pip install ai4water[ml]

For list of all options see :ref:`installation_options`.

using github link
=================
You can also use github link to install ai4water.
::
    python -m pip install git+https://github.com/AtrCheema/AI4Water.git

The latest code however (possibly with less bugs and more features) can be installed from ``dev`` branch instead
::
    python -m pip install git+https://github.com/AtrCheema/AI4Water.git@dev

To install the latest branch (`dev`) with all requirements use ``all`` keyword
::
    python -m pip install "AI4Water[all] @ git+https://github.com/AtrCheema/AI4Water.git@dev"

You can also install ai4water from a specific commit using the commit code (SHA) as below
::
    pip install git+https://github.com/AtrCheema/AI4Water.git@e3ec95c560ff5f43a215a2339f6602980ba06f03


using setup.py file
===================
go to folder where repository is downloaded
::
    python setup.py install

.. _installation_options:

installation options
=====================
The ``all`` option will install all the dependencies. You can choose the dependencies
of particular sub-module by using the specific keyword. Following keywords are available

 - ``hpo`` if you want hyperparameter optimization
 - ``post_process`` if you want postprocessing
 - ``exp`` for experiments sub-module
 - ``eda`` for exploratory data analysis sub-module
 - ``ml`` for classical machine learning models
 - ``tf`` for using tensorflow
 - ``torch``  for using pytorch
