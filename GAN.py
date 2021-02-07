import argparse
import tensorflow as tf

import config
from utils.model import GAN
from utils.callbacks import GANMonitor

ap = argparse.ArgumentParser()
ap.add_argument("--model_save", required=False, type=bool, help="Whether to save the generated model weight")
args = ap.parse_args()

model_save = True
if args.model_save:
    model_save = args.model_save

def load_mnist():
    (train_images, _), _ = tf.keras.datasets.mnist.load_data()
    return train_images.reshape((train_images.shape[0], 28*28)) / 127.5 - 1

def main(model_save=False):
    train_images = load_mnist()

    LATENT_DIM = config.LATENT_DIM
    MODEL_NAME = config.MODEL_NAME
    gan = GAN(LATENT_DIM, name=MODEL_NAME)

    tf.keras.backend.clear_session()
    LEARNING_RATE = config.LEARNING_RATE
    gan.compile(tf.keras.optimizers.Adam(LEARNING_RATE), tf.keras.optimizers.Adam(LEARNING_RATE))

    monitor_callback = GANMonitor()

    EPOCHS = config.EPOCHS
    BATCH_SIZE = config.BATCH_SIZE
    hist = gan.fit(
        x=train_images,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        shuffle=True,
        callbacks=[monitor_callback]
    )

    if model_save:
        gan.save_weights('./gan_save/ckpt')

if __name__ == '__main__':
    main(model_save)