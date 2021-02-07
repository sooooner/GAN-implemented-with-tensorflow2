import tensorflow as tf
from utils.layers import GENERATOR, DISCRIMINATOR

class GAN(tf.keras.Model):
    def __init__(self, LATENT_DIM, **kwargs):
        super(GAN, self).__init__(**kwargs)
        self.latent_dim = LATENT_DIM
        self.generator = GENERATOR(self.latent_dim, name='generator')
        self.discriminator = DISCRIMINATOR(name='discriminator')

    def compils(self, discriminator_optimizer, generator_optimizer):
        super(GAN, self).complie()
        self.discriminator_optimizer = discriminator_optimizer
        self.generator_optimizer = generator_optimizer

        self.discriminator_loss_tracker = tf.keras.metrics.Mean(name='discriminator_loss')
        self.generator_loss_tracker = tf.keras.metrics.Mean(name='generator_loss')

    @tf.function
    def Generating(self, num, eps=None):
        if eps == None:
            eps = tf.random.normal(shape=(num, self.latent_dim))
        return self.generator(eps)

    def Generator_Loss(self, z):
        score = self.discriminator(z)
        label = tf.ones_like(z)
        return tf.keras.losses.binary_crossentropy(y_true=label, y_pred=score)

    def Discriminator_Loss(self, x, z):
        real = self.discriminator(x)
        real_label = tf.ones_like(x)
        real_loss = tf.keras.losses.binary_crossentropy(y_true=real_label, y_pred=real)

        generated = self.discriminator(z)
        generated_label = tf.zeros_like(z)
        generated_loss = tf.keras.losses.binary_crossentropy(y_true=generated_label, y_pred=generated)
        return real_loss + generated_loss

    def train_step(self, data):
        batch_size = tf.shape(data)[0]
        
        with tf.GradientTape() as tape:
            generated_image = self.Generating(num=batch_size)
            discriminator_loss = self.Discriminator_Loss(data, generated_image)
        grad = tape.gradient(discriminator_loss, self.discriminator.trainable_weights)
        self.discriminator_optimizer.apply_gradients(zip(grad, self.discriminator.trainable_weights))

        with tf.GradientTape() as tape:
            generated_image = self.Generating(num=batch_size)
            generator_loss = self.Generator_Loss(generated_image)
        grad = tape.gradient(generator_loss, self.generator.trainable_weights)
        self.generator_optimizer.apply_gradients(zip(grad, self.generator.trainable_weights))

        self.generator_loss_tracker.update_state(generator_loss)
        self.discriminator_loss_tracker.update_state(discriminator_loss)

        return {
            'discriminator_loss': self.discriminator_loss_tracker.result(),
            'generator_loss' : self.generator_loss_tracker.result()
        }



