import tensorflow as tf
from attacks.base_attack import BaseAttack

class DropOutAttack(BaseAttack):
    def __init__(self, **kwargs):
        super(DropOutAttack, self).__init__()

    def drop_out(self, inputs):
        shp = tf.shape(inputs)

        # Randomized drop rate: 10%–50% of pixels dropped each batch
        # Wider range forces the model to generalize beyond the paper's fixed p=0.3
        drop_rate = tf.random.uniform(shape=[], minval=0.1, maxval=0.5, dtype=tf.float32)

        # Generate random mask 0-1
        mask_select = tf.random.uniform(shape=shp, minval=0, maxval=1, dtype=tf.float32)

        # Keep pixel if mask_select > drop_rate (i.e., drop_rate fraction are zeroed out)
        mask = tf.cast(mask_select > drop_rate, tf.float32)

        out = inputs * mask
        return out

    def call(self, inputs):
        return self.drop_out(inputs)

def drop_out_function(x):
    return DropOutAttack()(x)