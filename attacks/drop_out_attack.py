import tensorflow as tf
from attacks.base_attack import BaseAttack

class DropOutAttack(BaseAttack):
    def __init__(self, **kwargs):
        super(DropOutAttack, self).__init__()

    def drop_out(self, inputs):
        shp = tf.shape(inputs)
        
        # Paper uses p=0.3 (Drop 30% of pixels)
        # Generate random mask 0-1
        mask_select = tf.random.uniform(shape=shp, minval=0, maxval=1, dtype=tf.float32)
        
        # Keep if value > 0.3
        mask = tf.cast(mask_select > 0.3, tf.float32)
        
        out = inputs * mask
        return out

    def call(self, inputs):
        return self.drop_out(inputs)

def drop_out_function(x):
    return DropOutAttack()(x)