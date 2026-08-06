Cross-Entropy Importance Sampling Using Variational AutoEncoder
===============================================================
The reliability problem to solve is: 
:math:`P_f = \mathbb{P}[\psi\left( \mathbf{X} \right) \leq T)]`

where :math:`\mathbf{X}` is the input random vector of dimension :math:`d`, :math:`f_\mathbf{X}(\mathbf{x})`  is the joint density probability function, 
:math:`D_f = \{\mathbf{X} \in \mathbb{R}^d \, \vert \, \psi(\mathbf{X}) \leq T\}` is the domain definition of the failure event to consider (with :math:`T` a threshold) and :math:`\psi(\cdot) = T` defines the limit state.

Without loss of generality, the failure condition can also be :math:`\psi(\mathbf{X})\geq T`.

The probability of failure can be defined as follows: 
:math:`P_f = \mathbb{P}[\psi(\mathbf{X})\leq T] = \int_{\mathbb{R}^d}  \mathbf{1}_{ \left\{ \psi(\mathbf{X}) \leq T \right\} }f_\mathbf{X}(\mathbf{x}) d\mathbf{x}`

The Cross-Entropy Importance Sampling Using Variational AutoEncoder (VAE) algorithm is based on a Cross-Entropy Importance Sampling technique which is an adaptive algorithm to estimate the auxiliary distribution: 
:math:`P_f^{IS} = \int_{\mathbb{R}^d}  \mathbf{1}_{ \left\{ \psi(\mathbf{X}) \leq T \right\} } \frac{f_\mathbf{X}(\mathbf{x})}{g_\mathbf{X}(\mathbf{x})}g_\mathbf{X}(\mathbf{x}) d\mathbf{x}`

See for instance the OpenTURNS algorithm `PhysicalSpaceCrossEntropyImportanceSampling <https://openturns.github.io/openturns/latest/user_manual/_generated/openturns.PhysicalSpaceCrossEntropyImportanceSampling.html#openturns.PhysicalSpaceCrossEntropyImportanceSampling>`_ for an explanation of the classical Cross-Entropy Importance Sampling algorithm based on a parametric auxiliary distribution. 

However, in the present case, to be adapted to high dimensional problems, a Variational AutoEncoder is used as an auxiliary distribution :math:`g_\mathbf{X}(\cdot)`.  

Variational autoencoders are probabilistic tools able to represent high-dimensional data in a lower dimensional space. 
They constitute a parametric family of distributions with robutness capabilities regarding the increase of problem dimension. In addition, as VAE are based on deep neural networks, they are flexible enough to be considered as non-parametric models. 
A representation of the VAE architecture for Importance Sampling in high dimension is given by the following figure (from the `paper <https://openreview.net/forum?id=nzG9KGssSe>`_): 

.. container:: clearer

  .. image :: VAE_model.png

with :

* :math:`E_{\phi}` the encoder parameterized by the weights :math:`\phi`
* :math:`D_{\theta}` the decoder parameterized by the weights :math:`\theta`
* :math:`\mu_\mathbf{X},\Sigma_\mathbf{X}` the parameters of the variational posterior distribution (Gaussian distribution with diagonal covariance matrix)
* :math:`\mu_\mathbf{Z},\Sigma_\mathbf{Z}` the parameters of the likelihood distribution (Gaussian distribution)
* :math:`VP_{\lambda}` the variational mixture of posteriors prior (VampPrior)

See the `paper <https://openreview.net/forum?id=nzG9KGssSe>`_  for more details on the VAE and the algorithm settings. 

Reference
---------
- Demange-Chryst, J., Bachoc, F., Morio, J., & Krauth, T. `Variational autoencoder with weighted samples for high-dimensional non-parametric adaptive importance sampling. Transactions on Machine Learning Research. <https://openreview.net/forum?id=nzG9KGssSe>`_ 
