#%% Modules
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import numpy as np
import openturns as ot
import tensorflow as tf
from .VAE_IS_VP import fitted_vae
from silence_tensorflow import silence_tensorflow
silence_tensorflow()
tf.get_logger().setLevel('ERROR')
from matplotlib import pyplot as plt
from matplotlib import cm

class CEISVAEResult(ot.SimulationResult):
    """
    Class implementing the result of Cross-Entropy Importance Sampling algorithm based on Variational AutoEncoder (VAE).
    
    """
    def __init__(self):
        self.probabilityEstimate = None
        self.inputAuxiliarySample = None
        self.outputAuxiliarySample = None
        self.quantileLevel = None
        self.thresholdPerStep = None
        
    def getProbabilityEstimate(self):
        """
        Accessor to the probability estimate.
        
        """
        return self.probabilityEstimate
    
    def setProbabilityEstimate(self,proba):
        """
        Set up the probability estimate.
        
        
        """
        self.probabilityEstimate = proba
        return None
            
    def getInputAuxiliarySample(self,step):
        """
        Accessor to the input auxiliary sample at a specific step.
        
        Parameters
        ----------
        step: integer
            Step of interest for the input auxiliary sample. 
                
        """
        return self.inputAuxiliarySample[step]   

    def getOutputAuxiliarySample(self,step):
        """
        Accessor to the output auxiliary sample at a specific step.
        
        Parameters
        ----------
        step: integer
            Step of interest for the output auxiliary sample.
                
        """
        return self.outputAuxiliarySample[step]  

    def setInputAuxiliarySample(self,inputAuxiliarySample):
        """
        Set up the input auxiliary sample.
                
        """
        self.inputAuxiliarySample = inputAuxiliarySample
        return None   
    
    def setOutputAuxiliarySample(self,outputAuxiliarySample):
        """
        Set up the output auxiliary sample.
                
        """
        self.outputAuxiliarySample = outputAuxiliarySample
        return None  
     
    
    def getStepsNumber(self):
        """
        Accessor to the number of steps in Cross-Entropy algorithm.
                
        """
        return self.stepsNumber    

    def setStepsNumber(self,stepsNumber):
        """
        Set up the number of steps in Cross-Entropy algorithm.
                
        """
        self.stepsNumber = stepsNumber
        return None 

    def setThresholdPerStep(self,thresholdPerStep):
        """
        Set up the intermediate thresholds in Cross-Entropy algorithm.
                
        """
        self.thresholdPerStep = thresholdPerStep
        return None 

    def getThresholdPerStep(self):
        """
        Accessor to the intermediate thresholds in Cross-Entropy algorithm.
                
        """
        return self.thresholdPerStep
    
    def setVAE(self,VAE):
        """
        Set up the Variational AutoEncoder at the final step.
                
        """
        self.VAE = VAE
        return None 

    def getVAE(self):
        """
        Accessor to Variational AutoEncoder at the final step.
                
        """
        return self.VAE  
    
    def setOuterSampling(self,outerSampling):
        """
        Set up the total number of limit state function evaluations.
                
        """
        self.outerSampling = outerSampling
        return None 

    def getOuterSampling(self):
        """
        Accessor to the total number of limit state function evaluations.
                
        """
        return self.outerSampling  
    
    def setVarianceEstimate(self,varianceEstimate):
        """
        Set up the variance estimate.
                
        """
        self.varianceEstimate = varianceEstimate
        return None 

    def getVarianceEstimate(self):
        """
        Accessor to the variance estimate.
                
        """
        return self.varianceEstimate  


    def drawLatentSpace(self):
        """
        Draw of the latent space if VAE latent space is in dimension 2
                
        """
        if self.getVAE().latent_dim!=2:
            print('drawLatentSpace only available for VAE with latent space of dimension 2')
            return None
        
        else :
            stepNumber = int(self.getStepsNumber())-1
            X = self.getInputAuxiliarySample(stepNumber)
            y = self.getOutputAuxiliarySample(stepNumber)
            vae = self.getVAE()
            
            Xnp = np.array(X).astype("float32")
            mean_x = vae.mean_x
            std_x = vae.std_x

            X_normed = (Xnp-mean_x)/std_x
            
            Xtf = tf.convert_to_tensor(X_normed)
            
            vae_encoder,vae_decoder = vae.get_encoder_decoder()
            
            fig,ax = plt.subplots(figsize=(20,10))
            
            pseudo_inputs = vae.get_pseudo_inputs()
            _, _, z = vae_encoder(pseudo_inputs)
            Z_mean,Z_log_var,Z = vae_encoder(Xtf)
            ax.scatter(np.array(Z)[:,0],np.array(Z)[:,1],c=y,s=2)
            ax.scatter(np.array(z)[:,0],np.array(z)[:,1],color='white',s=20)
            
            x_min,x_max = ax.get_xlim()
            y_min,y_max = ax.get_ylim()
            
            nb_points = 501
            
            x1 = np.linspace(x_min,x_max, nb_points)
            x2 = np.linspace(y_min,y_max, nb_points)
            
            X1, X2 = np.meshgrid(x1, x2)
            
            values_function = np.zeros((nb_points,nb_points))
            for i in range(nb_points):
                for j in range(nb_points):
                    x = np.array([x1[j],x2[i]])
                    values_function[i,j] = vae.prior.computePDF(x)
            
            ax.contourf(X1, X2, values_function,levels=200,cmap=cm.YlOrBr,alpha=0.7)

            ax.set_xlabel("$z_1$",fontsize=20)
            ax.set_ylabel("$z_2$",fontsize=20)
            ax.tick_params(axis='x',labelsize='xx-large')
            ax.tick_params(axis='y',labelsize='xx-large')
            
            ax.set_title("2-dimensional latent space at the last iteration of the CE-VAE algorithm",fontsize=20)

            return fig,ax 

class CEISVAE(object):
    """
    Class implementing Cross-Entropy Importance Sampling algorithm based on Variational AutoEncoder (VAE).
    
    Parameters
    ----------
    event : :py:class:`openturns.ThresholdEvent`
        ThresholdEvent based on composite vector of input variables on limit state function. 
    
    quantileLevel : float
        Quantile level for Cross-Entropy Importance Sampling. 

    nbIS : integer 
        Number of IS samples at each CE-IS setp. 
            
    latentDim : integer
        Dimension of the latent space of the VAE.
         
    nbCompVampPrior : integer
        Number of components for the VampPrior.
    
    maxNbSim : integer 
        Total simulation budget allowed. 
    
    """
	
    def __init__(self,event,quantileLevel,nbIS=1e4,latentDim=2,nbCompVampPrior=75,maxNbSim=1e5):

        self.event = event        
        self.limitStateFunction = event.getFunction()
        self.S = event.getThreshold()
        self.distrib = event.getAntecedent().getDistribution()
        self.operator =  event.getOperator()
        self.dim = event.getAntecedent().getDimension()

        self.nbIS = nbIS
        self.proba = 0.
        self.nbSim = 0
        self.stepsNumber = 0.
        self.weights = None
        self.outerSampling = 0.

        self.VAE = None
        self.learning_rate = 0.001
        self.beta_1 = 0.9
        self.beta_2 = 0.999
        self.epsilon = 1e-07
        self.epochs = 100
        self.batch_size = 100
        
        self.latentDim = latentDim
        self.nbCompVampPrior = nbCompVampPrior
        
        self.dimLayer1 = np.max([self.latentDim,np.floor(self.dim/3.).astype(int)])
        self.dimLayer2 = np.max([self.latentDim,np.floor(self.dimLayer1/2.).astype(int)])
        
        self.inputAuxiliarySample = []
        self.outputAuxiliarySample = []
        self.maxNbSim = maxNbSim
        self.result = CEISVAEResult()
        
        self.quantileLevel = quantileLevel
        self.thresholdPerStep = []
        
    #Function computing the probability of failure
    def run(self):
        """
        Function computing failure probability using Cross-Entropy Importance Sampling using Variational AutoEncoder.
        
        """
        
        # Adaptation of the quantileLevel depending on the type of Event (Greater or Less)
        if self.operator(1,0) == True:
            self.quantileLevel = 1-self.quantileLevel #definition of quantile if exceedance probability ## type : float
            
        #Drawing of samples using initial density ## type: Sample
        sample = self.distrib.getSample(self.nbIS) 
        #Evaluation on limit state function ## type : Sample
        Ysample = self.limitStateFunction(sample) 
        self.nbSim += self.nbIS
        #Computation of current quantile ## type : float
        quantileCurrent = Ysample.computeQuantile(self.quantileLevel)[0] 
        #Comparison with threshold    
        if self.operator(quantileCurrent,self.S):
                weights = None
                VAE = None
        else:
            IndexWrtQuantileCurrent = np.array([self.operator(Ysample[i][0],quantileCurrent) for i in range(Ysample.getSize())])
            WeightsIndexWrtQuantileCurrent = IndexWrtQuantileCurrent
        
        
        #Append of input and output samples
        self.thresholdPerStep.append(quantileCurrent)        
        self.inputAuxiliarySample.append(sample)
        self.outputAuxiliarySample.append(Ysample)
        self.stepsNumber += 1.
        #Cross-Entropy IS loop using VAE
        while self.operator(self.S,quantileCurrent) and quantileCurrent != self.S and self.nbSim<self.maxNbSim:           
            #Fit of the VAE
            VAE,_,_ = fitted_vae(np.array(sample)[IndexWrtQuantileCurrent.flatten()].astype("float32"),
                                 WeightsIndexWrtQuantileCurrent[IndexWrtQuantileCurrent.flatten()].astype("float32"),
                                 self.latentDim,
                                 self.nbCompVampPrior,
                                 epochs=self.epochs,batch_size=self.batch_size,
                                 learning_rate=self.learning_rate,
                                 beta_1=self.beta_1,beta_2=self.beta_2, epsilon=self.epsilon,
                                 dimLayer1 = self.dimLayer1, dimLayer2 = self.dimLayer2)
            
            #Generation of new input samples and associated weights
            sample,log_gx = VAE.getSample(self.nbIS,with_pdf=True)
            log_fx = self.distrib.computeLogPDF(sample)
            logWeights = log_fx - log_gx
            weights = np.exp(logWeights)

            #Evaluation of the exact limit state function
            Ysample = self.limitStateFunction(sample)            
            self.nbSim += self.nbIS

            #Update of the current quantile and index with respect to current quantile
            quantileCurrent = Ysample.computeQuantile(self.quantileLevel)[0]
            self.thresholdPerStep.append(quantileCurrent)
            IndexWrtQuantileCurrent = np.array([self.operator(Ysample[i][0],quantileCurrent) for i in range(Ysample.getSize())])
            WeightsIndexWrtQuantileCurrent = IndexWrtQuantileCurrent*weights.squeeze()
            
            #Append of input and output samples
            self.inputAuxiliarySample.append(sample)
            self.outputAuxiliarySample.append(Ysample)
            self.stepsNumber += 1.
            
        #Estimation of the current probability of failure
        IndexWrtThreshold = np.array([self.operator(Ysample[i][0],self.S) for i in range(Ysample.getSize())])
        WeightsIndexWrtThreshold = IndexWrtThreshold*weights.squeeze()
        proba = np.mean(WeightsIndexWrtThreshold) #Calculation of failure probability #type : float
        
        #Estimation of the variance
        nbfailed = self.nbIS - np.count_nonzero(WeightsIndexWrtThreshold)
        varianceNonCritic = (self.nbIS-nbfailed)*proba**2
        varianceCritic  = np.sum(WeightsIndexWrtThreshold[WeightsIndexWrtThreshold != 0]-proba)**2
        varianceEstimate = (varianceCritic + varianceNonCritic) / (self.nbIS - 1) / self.nbIS
        
        #Save result
        self.result.setProbabilityEstimate(proba)
        self.result.setInputAuxiliarySample(self.inputAuxiliarySample)
        self.result.setOutputAuxiliarySample(self.outputAuxiliarySample)
        self.result.setThresholdPerStep(self.thresholdPerStep)
        self.result.setStepsNumber(self.stepsNumber)
        self.result.setVAE(VAE)
        self.result.setOuterSampling(self.nbSim)
        self.result.setVarianceEstimate(varianceEstimate)

        return None

    #Accessor to results
    def getResult(self):
        """
        Accessor to simulation algorithm result.     
        
        """
        return self.result
    
    def getQuantileLevel(self):
        """
        Accessor to quantile level.       
        
        """
        return self.quantileLevel    

    def setQuantileLevel(self,quantileLevel):
        """
        Set up the quantile level. 
        
        Parameters
        ----------
        quantileLevel : float
            Quantile level for Importance Sampling. 
        
        """
        self.quantileLevel = quantileLevel
        return None  
    
    def getVAEEpochs(self):
        """
        Accessor to epochs for Variational AutoEncoder.       
        
        """
        return self.epochs    

    def setVAEEpochs(self,epochs):
        """
        Set up the epochs for Variational AutoEncoder. 
        
        Parameters
        ----------
        epochs : integer
            epochs for training of Variational AutoEncoder with Adam optimizer. 
            
        """
        self.epochs = epochs
        return None 
    
    def getVAEBatchSize(self):
        """
        Accessor to batch size for Variational AutoEncoder.      
        
        """
        return self.batch_size    

    def setVAEBatchSize(self,batch_size):
        """
        Set up the batch size for Variational AutoEncoder.       
        
        Parameters
        ----------
        batch_size : integer
            batch size for training of Variational AutoEncoder with Adam optimizer. 
            
        """
        self.batch_size = batch_size
        return None 
    
    def getVAELearningRate(self):
        """
        Accessor to learning rate for training of Variational AutoEncoder with Adam optimizer.       
        
        """
        return self.learning_rate    

    def setVAELearningRate(self,learning_rate):
        """
        Set up the learning rate for training of Variational AutoEncoder with Adam optimizer.      
        
        Parameters
        ----------
        learning_rate : float
            learning rate for training of Variational AutoEncoder with Adam optimizer.
            
        """
        self.learning_rate = learning_rate
        return None 
    
    def getVAEBeta1(self):
        """
        Accessor to beta1 parameter for training of Variational AutoEncoder with Adam optimizer.     
        
        """
        return self.beta_1    

    def setVAEBeta1(self,beta_1):
        """
        Set up the beta1 parameter for training of Variational AutoEncoder with Adam optimizer.      
        
        Parameters
        ----------
        beta_1 : float
            beta_1 parameter for training of Variational AutoEncoder with Adam optimizer.
            
        """
        self.beta_1 = beta_1
        return None 
    
    def getVAEBeta2(self):
        """
        Accessor to beta2 parameter for training of Variational AutoEncoder with Adam optimizer.       
        
        """
        return self.beta_2 

    def setVAEBeta2(self,beta_2):
        """
        Set up the beta2 parameter for training of Variational AutoEncoder with Adam optimizer.      
        
        Parameters
        ----------
        beta_2 : float
            beta_2 parameter for training of Variational AutoEncoder with Adam optimizer.
            
        """
        self.beta_2 = beta_2
        return None 
    
    def getVAEEpsilon(self):
        """
        Accessor to epsilon parameter for training of Variational AutoEncoder with Adam optimizer.       
        
        """
        return self.epsilon 

    def setVAEEpsilon(self,epsilon):
        """
        Set up the epsilon parameter for training of Variational AutoEncoder with Adam optimizer.      
        
        Parameters
        ----------
        epsilon : float
            epsilon parameter for training of Variational AutoEncoder with Adam optimizer.
            
        """
        self.epsilon = epsilon
        return None 
    
    def setVAEdimLayer1(self,dimLayer1):
        """
        Set up the dimension of Layer1 of Variational AutoEncoder. 
        
        Parameters
        ----------
        dimLayer1 : integer
            Dimension of the Layer 1 of the Variational AutoEncoder.
            
        """
        self.dimLayer1 = dimLayer1
        return None 

    def getVAEdimLayer1(self):
        """
        Accessor to dimension of Layer1 of Variational AutoEncoder.      
        
        """
        return self.dimLayer1  

    def setVAEdimLayer2(self,dimLayer2):
        """
        Set up the dimension of Layer2 of Variational AutoEncoder. 
        
        Parameters
        ----------
        dimLayer2 : integer
            Dimension of the Layer 2 of the Variational AutoEncoder.
            
        """
        self.dimLayer2 = dimLayer2
        return None 

    def getVAEdimLayer2(self):
        """
        Accessor to dimension of Layer2 of Variational AutoEncoder.       
        
        """
        return self.dimLayer2  
    
