import tensorflow as tf

from attacks.base_attack import BaseAttack


class GaussianNoiseAttack(BaseAttack):
    def __init__(self, **kwargs):
        super(GaussianNoiseAttack, self).__init__()

    def gaussian_noise(self, inputs):
        shp = tf.shape(inputs)[1:]
        noise = tf.random.normal(shape=shp, mean=0.0, stddev=.1, dtype=tf.float32)
        out = inputs + noise
        return out

    def call(self, inputs):
        outputs = self.gaussian_noise(inputs)
        return outputs


def gaussian_noise_function(x):
    return GaussianNoiseAttack()(x)
