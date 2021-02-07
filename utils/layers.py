import tensorflow as tf

class GENERATOR(tf.keras.layers.Layer):
    def __init__(self, LATENT_DIM, **kwargs):
        super(GENERATOR, self).__init__(**kwargs)
        self.latent_dim = LATENT_DIM
        self.dense1 = tf.keras.layers.Dense(128, input_dim=self.latent_dim, name='generator_dense1')
        self.dense2 = tf.keras.layers.Dense(256, name='generator_dense2')
        self.dense3 = tf.keras.layers.Dense(512, name='generator_dense3')
        self.dense4 = tf.keras.layers.Dense(28*28, activation='tanh', name='generator_dense4')

    def call(self, inputs):
        x = self.dense1(inputs)
        x = tf.keras.layers.LeakyReLU(0.2)(x)
        x = self.dense2(x)
        x = tf.keras.layers.LeakyReLU(0.2)(x)
        x = self.dense3(x)
        x = tf.keras.layers.LeakyReLU(0.2)(x)
        outputs = self.dense4(x)
        return outputs


class DISCRIMINATOR(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(DISCRIMINATOR).__init__(**kwargs)
        self.dense1 = tf.keras.layers.Dense(512, input_dim=28*28, name='discriminator_dense1')
        self.drop_out1 = tf.keras.layers.Dropout(0.3)
        self.dense2 = tf.keras.layers.Dense(256, name='discriminator_dense2')
        self.drop_out2 = tf.keras.layers.Dropout(0.3)
        self.dense3 = tf.keras.layers.Dense(128, name='discriminator_dense3')
        self.drop_out3 = tf.keras.layers.Dropout(0.3)
        self.dense3 = tf.keras.layers.Dense(1, activation='sigmoid', name='discriminator_dense3')

    def call(self, inputs):
        x = self.dense1(inputs)
        x = tf.keras.layers.LeakyReLU(0.2)(x)
        x = self.dropout1(x)
        x = self.dense2(x)
        x = tf.keras.layers.LeakyReLU(0.2)(x)
        x = self.dropout1(x)
        x = self.dense3(x)
        x = tf.keras.layers.LeakyReLU(0.2)(x)
        x = self.dropout1(x)
        outputs = self.dense4(x)
        return outputs
