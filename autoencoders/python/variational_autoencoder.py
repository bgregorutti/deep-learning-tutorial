# !/usr/bin/env python
#  -*- coding: utf-8 -*-

from keras.layers import Input, Dense, Lambda
from keras.models import Model
from keras import backend as K
from keras.losses import binary_crossentropy
from keras.utils import plot_model

def sampling(args):
    """Reparameterization trick by sampling from an isotropic unit Gaussian.
    # Arguments
        args (tensor): mean and log of variance of Q(z|X)
    # Returns
        z (tensor): sampled latent vector
    """

    z_mean, z_log_var = args
    batch = K.shape(z_mean)[0]
    dim = K.int_shape(z_mean)[1]
    # by default, random_normal has mean = 0 and std = 1.0
    epsilon = K.random_normal(shape=(batch, dim))
    return z_mean + K.exp(0.5 * z_log_var) * epsilon

def vae_network(input_shape, batch_size, original_dim, intermediate_dim, latent_dim):
    """
    Fully-connected VAE (from Keras documentation)
    """

    # build encoder model
    inputs = Input(shape=input_shape, name='encoder_input')
    x = Dense(intermediate_dim, activation='relu')(inputs)
    z_mean = Dense(latent_dim, name='z_mean')(x)
    z_log_var = Dense(latent_dim, name='z_log_var')(x)

    # use reparameterization trick to push the sampling out as input
    # note that "output_shape" isn't necessary with the TensorFlow backend
    z = Lambda(sampling, output_shape=(latent_dim,), name='z')([z_mean, z_log_var])

    # instantiate encoder model
    encoder = Model(inputs, [z_mean, z_log_var, z], name='encoder')
    encoder.summary()
    # plot_model(encoder, to_file='vae_mlp_encoder.png', show_shapes=True)

    # build decoder model
    latent_inputs = Input(shape=(latent_dim,), name='z_sampling')
    x = Dense(intermediate_dim, activation='relu')(latent_inputs)
    outputs = Dense(original_dim, activation='sigmoid')(x)

    # instantiate decoder model
    decoder = Model(latent_inputs, outputs, name='decoder')
    decoder.summary()
    # plot_model(decoder, to_file='vae_mlp_decoder.png', show_shapes=True)

    # instantiate VAE model
    outputs = decoder(encoder(inputs)[2])
    vae = Model(inputs, outputs, name='vae_mlp')

    # KL loss
    reconstruction_loss = binary_crossentropy(inputs, outputs) * original_dim
    kl_loss = 1 + z_log_var - K.square(z_mean) - K.exp(z_log_var)
    kl_loss = K.sum(kl_loss, axis=-1)
    kl_loss *= -0.5
    vae_loss = K.mean(reconstruction_loss + kl_loss)

    # Compile
    vae.compile(optimizer='rmsprop', loss=vae_loss)
    vae.summary()

    return vae, encoder, decoder

def test_mnist():
    from keras.datasets import mnist
    import numpy as np

    epochs = 5
    batch_size = 128

    intermediate_dim = 512
    latent_dim = 2

    # Get the data
    # (x_train, _), (x_test, _) = mnist.load_data()
    x_train = np.load('MNIST_x_train.npy')
    x_test = np.load('MNIST_x_test.npy')
    
    # Normalize
    x_train = x_train.astype('float32') / 255.
    x_test = x_test.astype('float32') / 255.
    
    # Reshape back to the original shape
    x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
    x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))

    print(x_train.shape, x_test.shape)
    print(np.min(x_train), np.max(x_train))

    original_dim = x_train.shape[1]
    input_shape = (original_dim,)
    
    # Get the network
    vae, _, _ = vae_network(input_shape, batch_size, original_dim, intermediate_dim, latent_dim)

    # Fit the model
    vae.fit(x_train, x_train,
            shuffle=True,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(x_test, x_test))

if __name__ == '__main__':
    test_mnist()

