import tensorflow as tf

from attacks.base_attack import BaseAttack


class SaltPepperAttack(BaseAttack):
    def __init__(self, **kwargs):
        super(SaltPepperAttack, self).__init__()

    def salt_pepper(self, inputs):
        shp = tf.shape(inputs)[1:]
        # Use random uniform to simulate binomial distribution
        mask_select = tf.cast(
            tf.random.uniform(shape=shp, minval=0, maxval=1) < 0.1,
            tf.float32
        )
        mask_noise = tf.cast(
            tf.random.uniform(shape=shp, minval=0, maxval=1) < 0.5,
            tf.float32
        )
        out = inputs * (1 - mask_select) + mask_noise * mask_select
        return out

    def call(self, inputs):
        outputs = self.salt_pepper(inputs)
        return outputs


def salt_pepper_function(x):
    return SaltPepperAttack()(x)
