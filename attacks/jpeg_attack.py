import tensorflow as tf

from attacks.base_attack import BaseAttack


class JPEGAttack(BaseAttack):
    def __init__(self, **kwargs):
        super(JPEGAttack, self).__init__()

    def jpeg(self, inputs):
        # Use tf.map_fn to avoid scope issues with loops in graph mode
        def apply_jpeg(image):
            return tf.image.adjust_jpeg_quality(image, 50)
        
        return tf.map_fn(apply_jpeg, inputs, dtype=tf.float32)

    def call(self, inputs):
        outputs = self.jpeg(inputs)
        return outputs


def jpeg_function(x):
    return JPEGAttack()(x)
