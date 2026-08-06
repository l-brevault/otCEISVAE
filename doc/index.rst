otCEISVAE documentation
========================

.. image:: _static/CEISVAE.png
     :align: left
     :scale: 40%

otCEISVAE is an `OpenTURNS <https://openturns.github.io/www/>`_ module implementing an algorithm to carry out reliability analysis for high-dimensional problem based on Cross-Entropy Importance Sampling using Variational AutoEncoder. 
This work is an adaptation to the OpenTURNS framework of the Ph.D. thesis work of Julien Demange-Chryst and of an existing `Python implementation <https://github.com/Julien6431/Importance-Sampling-VAE/tree/main>`_
It allows to use classical OpenTURNS  `reliability analysis problem implementations <http://openturns.github.io/openturns/latest/user_manual/reliability.html>`_

The implementation of the Variational AutoEncoder relies on `Tensorflow <https://www.tensorflow.org/>`_ (CPU version >=2.19), `Tensorflow-Probability <https://www.tensorflow.org/probability>`_ (version >=0.25), `Keras <https://keras.io/>`_  (version >=3.10) and tf_keras (version>=2.19). Moreover, the problem associated to the reliability analysis is implemented using OpenTURNS (version >=1.25). 

The corresponding `paper <https://openreview.net/forum?id=nzG9KGssSe>`_  is "Demange-Chryst, J., Bachoc, F., Morio, J., & Krauth, T. Variational autoencoder with weighted samples for high-dimensional non-parametric adaptive importance sampling. Transactions on Machine Learning Research."




Theory
------

.. toctree::
   :maxdepth: 1  
   
   principle/principle

User documentation
------------------

.. toctree::
   :maxdepth: 2

   user_manual/user_manual

Examples 
--------

.. toctree::
   :maxdepth: 2  
   
   examples/examples

References
----------
- Demange-Chryst, J., Bachoc, F., Morio, J., & Krauth, T. `Variational autoencoder with weighted samples for high-dimensional non-parametric adaptive importance sampling. Transactions on Machine Learning Research. <https://openreview.net/forum?id=nzG9KGssSe>`_ 


Indices and tables
==================

* :ref:`genindex`
* :ref:`search`

