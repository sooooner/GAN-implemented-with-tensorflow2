from IPython import display

import tensorflow as tf
import matplotlib.pyplot as plt
import glob
import imageio

class GANMonitor(tf.keras.callbacks.Callback):
    def __init__(self, num_img=25):
        self.num_img = num_img
        self.random_latent_vectors = None

    def generate_and_images(self, epoch, num=25):
        if self.random_latent_vectors == None:
            self.random_latent_vectors = tf.random.normal(shape=(self.num_img, self.model.latent_dim))
        generated_images = self.model.Generating(num, self.random_latent_vectors)
        generated_images = (generated_images + 1) * 127.5
        fig = plt.figure(figsize=(7, 7))
        for i in range(num):
            plt.subplot(5, 5, i+1)
            plt.imshow(tf.reshape(generated_images, shape=(-1, 28, 28))[i])
            plt.axis('off')
        fig.suptitle(f'image_at_epoch_{epoch+1}')
        plt.savefig(f'./img/image_at_epoch_{epoch+1}.png')
        plt.show()
    
    def on_train_begin(self, logs=None):
        display.clear_output(wait=False)
        self.generate_and_images(-1, self.num_img)

    def on_epoch_end(self, epoch, logs=None):
        display.clear_output(wait=False)
        self.generate_and_images(epoch, self.num_img)

    def on_train_end(self, logs):
        anim_file = './img/gan.gif'
        with imageio.get_writer(anim_file, mode='I') as writer:
            filenames = glob.glob('./img/image_*.png')
            filenames = sorted(filenames)
            for filename in filenames:
                image = imageio.imread(filename)
                writer.append_data(image)