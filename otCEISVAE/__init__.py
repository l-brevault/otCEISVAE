"""
    otCEISVAE --- An OpenTURNS module
    =================================

    Contents
    --------
      'otCEISVAE' is a module for OpenTURNS

"""

# flake8: noqa

from .CE_VAE    import CEISVAE, CEISVAEResult
from .AE_class  import AutoEncoder, create_encoder_AE, create_decoder_AE
from .VAE_class import VAE, Sampling, create_encoder_VAE, create_decoder_VAE
from .VAE_IS_VP import fitted_vae, fitted_ae, initial_vp_layer, create_pseudo_inputs_layer

__version__ = '0.0'
