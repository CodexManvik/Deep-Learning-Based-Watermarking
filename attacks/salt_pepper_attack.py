import tensorflow as tf
from attacks.base_attack import BaseAttack

class SaltPepperAttack(BaseAttack):
    def __init__(self, **kwargs):
        super(SaltPepperAttack, self).__init__()

    def salt_pepper(self, inputs):
        shp = tf.shape(inputs)
        
        # Paper uses p=0.1 (total noise). Split 50/50 between Salt and Pepper.
        # Thresholds: 0.05 for Salt, 0.05 for Pepper
        
        # Generate one random map
        prob_map = tf.random.uniform(shape=shp, minval=0, maxval=1, dtype=tf.float32)
        
        # Salt: pixels < 0.05 becomes 1.0
        mask_salt = tf.cast(prob_map < 0.05, tf.float32)
        
        # Pepper: pixels > 0.95 becomes 0.0 (equivalent to another 0.05 slice)
        mask_pepper = tf.cast(prob_map > 0.95, tf.float32)
        
        # Apply:
        # If salt (1), add 1 (will be clipped).
        # If pepper (1), multiply by 0.
        out = inputs * (1 - mask_pepper) + mask_salt
        
        return tf.clip_by_value(out, 0.0, 1.0)

    def call(self, inputs):
        return self.salt_pepper(inputs)

def salt_pepper_function(x):
    return SaltPepperAttack()(x)