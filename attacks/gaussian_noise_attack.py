import tensorflow as tf
from attacks.base_attack import BaseAttack

class GaussianNoiseAttack(BaseAttack):
    """
    Gaussian Noise Attack
    
    Paper baseline: σ=0.15 (fixed for evaluation)
    Training: σ ∈ [0.05, 0.15] (randomized for robustness)
    
    The random range improves generalization across noise levels.
    """
    
    def __init__(self, stddev_range=(0.05, 0.15), **kwargs):
        super(GaussianNoiseAttack, self).__init__()
        self.stddev_min = stddev_range[0]
        self.stddev_max = stddev_range[1]
    
    def gaussian_noise(self, inputs):
        shp = tf.shape(inputs)
        
        # Randomly sample stddev for training variety
        stddev = tf.random.uniform(
            shape=[],
            minval=self.stddev_min,
            maxval=self.stddev_max,
            dtype=tf.float32
        )
        
        noise = tf.random.normal(shape=shp, mean=0.0, stddev=stddev, dtype=tf.float32)
        out = inputs + noise
        return tf.clip_by_value(out, 0.0, 1.0)
    
    def call(self, inputs):
        return self.gaussian_noise(inputs)

def gaussian_noise_function(x):
    return GaussianNoiseAttack()(x)

# For evaluation with fixed stddev (paper baseline)
def gaussian_noise_fixed_function(x, stddev=0.15):
    return GaussianNoiseAttack(stddev_range=(stddev, stddev))(x)
