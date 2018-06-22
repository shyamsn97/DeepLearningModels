from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D, Lambda
from keras.models import Model
from keras import backend as K
from keras.losses import *
import numpy as np
import matplotlib.pyplot as plt

class VAE():
    """
        Variational Auto Encoder used for data generation and latent data representation
        Model is a simple MLP
        adapted from https://github.com/keras-team/keras/blob/master/examples/variational_autoencoder.py
    """
    def __init__(self):
        
        self.encoder = None
        self.decoder = None
        self.vae = None
        self.z_mean = None
        self.z_log_var = None
        self.z = None
        self.loss = None

    def initialize(self,X):
        # network parameters
        original_dim = X.shape[-1]
        input_shape = (original_dim, )
        intermediate_dim = 512
        batch_size = 128
        latent_dim = 2
        epochs = 50

        # VAE model = encoder + decoder
        # build encoder model
        
        inputs = Input(shape=input_shape, name='encoder_input')
        x = Dense(intermediate_dim, activation='relu')(inputs)
        z_mean = Dense(latent_dim, name='z_mean')(x)
        z_log_var = Dense(latent_dim, name='z_log_var')(x)
        def sample(args):
            """ reparameterization trick
                z ~ N(mu,sigma^2*I) -> z = mu + sigma*e where e ~ N(0,I)
            """
            z_mean, z_log_var = args
            batch = K.shape(z_mean)[0]
            dim = K.int_shape(z_mean)[1]
            epsilon = K.random_normal(shape=(batch, dim))
            return z_mean + K.exp(z_log_var/2) * epsilon

        z = Lambda(sample, output_shape=(latent_dim,), name='z')([z_mean, z_log_var])

        # encoder model
        self.encoder = Model(inputs, [z_mean, z_log_var, z], name='encoder')
        self.encoder.summary()

        # build decoder model
        latent_inputs = Input(shape=(latent_dim,), name='z_sampling')
        x = Dense(intermediate_dim, activation='relu')(latent_inputs)
        outputs = Dense(original_dim, activation='sigmoid')(x)

        # instantiate decoder model
        self.decoder = Model(latent_inputs, outputs, name='decoder')
        self.decoder.summary()
        # instantiate VAE model
        outputs = self.decoder(self.encoder(inputs)[2])
        self.vae = Model(inputs, outputs, name='vae_mlp')

        reconstruction_loss = binary_crossentropy(inputs,
                                                      outputs)
        reconstruction_loss *= original_dim
        kl_loss = 1 + z_log_var - K.square(z_mean) - K.exp(z_log_var)
        kl_loss = K.sum(kl_loss, axis=-1)
        kl_loss *= -0.5
        vae_loss = K.mean(reconstruction_loss + kl_loss)
        self.vae.add_loss(vae_loss)
        self.vae.compile(optimizer='adam')
    
    def mean_squared_error(self,y_true, y_pred):
        return K.mean(K.square(y_pred - y_true), axis=-1)
        
    def fit(self,X_train,X_test,batch_size=1,epochs=30):
        self.vae.fit(X_train, batch_size=batch_size, epochs=epochs, validation_data=(X_test,None))
        
    def predict(self,X):
        return self.vae.predict(X)

