"""
Example 1: Use of CEISVAE on Four Branches function
---------------------------------------------------
"""

# %%
# This example aims to illustrate how to use the otCEISVAE module.

# %%
# | Loading python modules

# %%
import numpy as np
import openturns as ot
import otCEISVAE
from matplotlib import pyplot as plt
from matplotlib import cm
ot.RandomGenerator.SetSeed(1)

# %%
# | The considered reliability analysis problem is the classical four branches problem defined for a dimension d=100
def four_branches(x):
    xnp = np.array(x)
    d = len(xnp)
    side_1 = 1/np.sqrt(d)*np.sum(xnp)
    side_2 = -1/np.sqrt(d)*np.sum(xnp)
    side_3 = 1/np.sqrt(d)*(np.sum(xnp[:d//2]) - np.sum(xnp[d//2:]))
    side_4 = 1/np.sqrt(d)*(-np.sum(xnp[:d//2]) + np.sum(xnp[d//2:]))
    return [-np.min([side_1,side_2,side_3,side_4])]

# %%
# | Definition of the parameters for CE-IS using VAE
d = 100 # dimension of the reliability analysis problem
#### Creation of an OpenTURNS Python function of input dimension d and output dimension 1
g = ot.PythonFunction(d,1,four_branches)
#### Creation of the PDF of input random vector 
phi = ot.Normal(d)
#### Definition of threshold
S       = 3.5                                         
#### Definition of the failure event
vect    = ot.RandomVector(phi)                        
output  = ot.CompositeRandomVector(g, vect)
myEvent = ot.ThresholdEvent(output, ot.Greater(), S)  # definition of the threshold event
#### Parameters for the Cross-Entropy Importance Sampling
quantileLevel   = 0.25   # quantile level for the steps
nbIS            = 10000  # number of Importance Sampling sample
maxNbSim        = 1e5    # maximum number of limit state function evaluations 
#### Parameters for the Variational AutoEncoder
latentDim       = 2      # dimension of the latent space for the VAE
nbCompVampPrior = 75     # number of components for the VampPrior

# %%
# | Definition of the algorithm
myAlgo = otCEISVAE.CEISVAE(myEvent,
                  quantileLevel,
                  nbIS=nbIS,
                  latentDim=latentDim,
                  nbCompVampPrior=nbCompVampPrior,
                  maxNbSim=maxNbSim)

# %%
# | Run of the algorithm and get the result
myAlgo.run()
result = myAlgo.getResult()

# %%
# | Probability estimate, variance and simulation budget
print('Probability of failure :',result.getProbabilityEstimate())
print('Number of steps in the CE-IS using VAE algorithm :',result.getStepsNumber())
print('Simulation budget :',result.getOuterSampling())
print('Variance estimate :',result.getVarianceEstimate())

# %%
# | Draw of the VAE latent space (only possible if VAE latent space is of dimension 2)
fig, ax = result.drawLatentSpace()

# %%
# | The figure represents the CE-IS samples at the final iteration in the latent space (the PDF and the samples).
