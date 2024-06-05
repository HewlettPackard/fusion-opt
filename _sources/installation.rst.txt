========================================
Fusion-opt installation
========================================

Dependencies
------------

1. Linux OS (tested on Ubuntu 20.04)
2. Python 3.9
3. Conda_

.. _Conda: https://docs.conda.io/projects/conda/en/latest/user-guide/install/linux.html

First time setup
----------------

Clone the latest Fusion-opt version from GitHub using:

.. code-block:: bash
    
    git clone https://github.com/HewlettPackard/fusion-opt.git

If using SSH, execute:

.. code-block:: bash
    
    git clone git@github.com:HewlettPackard/fusion-opt.git

Change the current working directory to the fusion-opt folder:

.. code-block:: bash
    
    cd fusion-opt

Create a conda environment and install the needed dependencies:

.. code-block:: bash
    
    conda create -n fusionopt python=3.9
    conda activate fusionopt
    pip install -r requirements.txt