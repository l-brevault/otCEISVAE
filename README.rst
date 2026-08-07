otCEISVAE module
================

otCEISVAE is an `OpenTURNS <https://openturns.github.io/www/>`_ module implementing an algorithm to carry out reliability analysis for high-dimensional problem based on Cross-Entropy Importance Sampling using Variational AutoEncoder. 
This work is an adaptation to the OpenTURNS framework of the Ph.D. thesis work of Julien Demange-Chryst and of an existing `Python implementation <https://github.com/Julien6431/Importance-Sampling-VAE/tree/main>`_
It allows to use classical OpenTURNS  `reliability analysis problem implementations <http://openturns.github.io/openturns/latest/user_manual/reliability.html>`_

The implementation of the Variational AutoEncoder relies on `Tensorflow <https://www.tensorflow.org/>`_ (CPU version ==2.19) and `Tensorflow-Probability <https://www.tensorflow.org/probability>`_ (version ==0.25). Moreover, the problem associated to the reliability analysis is implemented using OpenTURNS (version >=1.25). 

The corresponding `paper <https://openreview.net/forum?id=nzG9KGssSe>`_  is "Demange-Chryst, J., Bachoc, F., Morio, J., & Krauth, T. Variational autoencoder with weighted samples for high-dimensional non-parametric adaptive importance sampling. Transactions on Machine Learning Research."


Prerequisites
=============
This module can only work with python==3.10 (to ensure compatibility between the packages).

Several Python packages are required:
 
* numpy>=1.26.4,
* tensorflow-cpu==2.19.0,
* silence_tensorflow==1.2.3,
* tensorflow_probability==0.25.0,
* keras>=3.10.0,
* tf_keras>=2.19.0,
* scikit-learn>=1.5.1,
* openturns>=1.25

Documentation
=============

A  documentation, including examples is available `here <https://l-brevault.github.io/otCEISVAE/master/index.html>`_.

Build from source
=================

The install procedure is performed as follows:

.. code-block:: shell

    $ pip install .

If you need to install the module in the user folder:

.. code-block:: shell

    $ pip install . --user

To run the tests:

.. code-block:: shell

    $ pytest

Authors
=======

* **Julien Demange-Chryst** - *ONERA* - [mailto](mailto:julien.demange-chryst@onera.fr)
* **Loic Brevault** - *ONERA* - [mailto](mailto:loic.brevault@onera.fr)
* **Mathieu Balesdent** - *ONERA* - [mailto](mailto:mathieu.balesdent@onera.fr)
