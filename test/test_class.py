import openturns as ot
import numpy as np
import otCEISVAE
import pytest

ot.RandomGenerator.SetSeed(1)

def four_branches(x):
    xnp = np.array(x)
    d = len(xnp)
    side_1 = 1/np.sqrt(d)*np.sum(xnp)
    side_2 = -1/np.sqrt(d)*np.sum(xnp)
    side_3 = 1/np.sqrt(d)*(np.sum(xnp[:d//2]) - np.sum(xnp[d//2:]))
    side_4 = 1/np.sqrt(d)*(-np.sum(xnp[:d//2]) + np.sum(xnp[d//2:]))
    return [-np.min([side_1,side_2,side_3,side_4])]


def test_class():
    # Definition of test case four branches
    d = 100
    g = ot.PythonFunction(d,1,four_branches)
    phi = ot.Normal(d)
    S       = 3.5                                         
    vect    = ot.RandomVector(phi)                        
    output  = ot.CompositeRandomVector(g, vect)
    myEvent = ot.ThresholdEvent(output, ot.Greater(), S)  # definition of the threshold event
    quantileLevel   = 0.25   # quantile level for the steps
    nbIS            = 10000  # number of Importance Sampling sample
    maxNbSim        = 1e5    # maximum number of limit state function evaluations 
    latentDim       = 2      # dimension of the latent space for the VAE
    nbCompVampPrior = 75     # number of components for the VampPrior
    
    # Defintion of the algorithm
    myAlgo = otCEISVAE.CEISVAE(myEvent,
                  quantileLevel,
                  nbIS=nbIS,
                  latentDim=latentDim,
                  nbCompVampPrior=nbCompVampPrior,
                  maxNbSim=maxNbSim)

    #run algorithm
    myAlgo.run()
    result = myAlgo.getResult()
    print(result.getProbabilityEstimate())
    assert result.getProbabilityEstimate()== pytest.approx(0.0009,abs=1e-4)