#%% Modules
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from silence_tensorflow import silence_tensorflow
silence_tensorflow()
tf.get_logger().setLevel('ERROR')    

#%% Encoder/Decoder architecture
    
def create_encoder_AE(input_dim,latent_dim,dimLayer1,dimLayer2):
    
    encoder_inputs = keras.Input(shape=(input_dim,))
    x = layers.Dense(dimLayer1, activation="relu")(encoder_inputs)
    x = layers.Dense(dimLayer2, activation="relu")(x)
    z = layers.Dense(latent_dim, activation="linear")(x)
    z_log_var = layers.Dense(latent_dim, activation="linear")(x)
    encoder = keras.Model(encoder_inputs, [z,z_log_var], name="encoder")
    return encoder

class EmbeddedLayer(keras.Layer):
    def call(self, x):
        return tf.math.maximum(x, -300.)

def create_decoder_AE(input_dim,latent_dim,dimLayer1,dimLayer2):
    
    threshold = -300
    
    latent_inputs = keras.Input(shape=(latent_dim,))
    x = layers.Dense(dimLayer2, activation="relu")(latent_inputs)
    x = layers.Dense(dimLayer1, activation="relu")(x)
    x_output = layers.Dense(input_dim, activation="linear")(x)
    x_log_var = layers.Dense(input_dim, activation="linear")(x)
    x_log_var = EmbeddedLayer()(x_log_var)
    #x_log_var = tf.math.maximum(x_log_var, threshold)
    decoder = keras.Model(latent_inputs, [x_output,x_log_var], name="decoder")
    return decoder


#%% Autoencoder class

class AutoEncoder(keras.Model):
    def __init__(self, input_dim, latent_dim,dimLayer1,dimLayer2, **kwargs):
        super().__init__(**kwargs)
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.encoder = create_encoder_AE(input_dim,latent_dim,dimLayer1,dimLayer2)
        self.decoder = create_decoder_AE(input_dim,latent_dim,dimLayer1,dimLayer2)
        self.total_loss_tracker = keras.metrics.Mean(name="total_loss")

    @property
    def metrics(self):
        return [self.total_loss_tracker]

    def get_encoder_decoder(self):
        return self.encoder,self.decoder

    def train_step(self, data):
        
        X,y = data
        y = tf.reshape(y,[-1])      
    
        with tf.GradientTape() as tape:
            
            z,z_log_var = self.encoder(X)
            reconstruction,x_log_var = self.decoder(z)
            
            xx = tf.pow(X - reconstruction,2)
            zz = tf.pow(z_log_var,2)
            xx_log_var = tf.pow(x_log_var,2)
            total_loss = tf.reduce_mean(tf.multiply(tf.reduce_mean(xx,axis=1),y)) + tf.reduce_mean(tf.multiply(tf.reduce_mean(zz,axis=1),y)) #+ tf.reduce_mean(tf.multiply(tf.reduce_mean(xx_log_var,axis=1),y))
            
            
            
        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        self.total_loss_tracker.update_state(total_loss)
        
        return {"loss": self.total_loss_tracker.result()}
