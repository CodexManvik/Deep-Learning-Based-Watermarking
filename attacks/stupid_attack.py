import tensorflow as tf
from attacks.base_attack import BaseAttack


class StupidAttack(BaseAttack):
    """
    Median Blur Attack (3x3 approximation via average pooling).

    The original StupidAttack was a pure identity pass-through — identical to
    attack ID 0 (no attack), wasting 12.5% of the attack probability budget.

    Replaced with a 3x3 average pool blur, which:
    - Is fully differentiable (gradients flow through tf.nn.avg_pool)
    - Simulates low-pass / smoothing attacks that destroy high-frequency watermarks
    - Is distinct from Gaussian noise (adds no random noise, only spatial blurring)
    - Is computationally cheap
    """

    def __init__(self, **kwargs):
        super(StupidAttack, self).__init__()

    def call(self, inputs):
        # 3x3 average pooling with stride 1 and SAME padding — equivalent to box blur
        blurred = tf.nn.avg_pool2d(
            inputs,
            ksize=3,
            strides=1,
            padding='SAME',
            data_format='NHWC'
        )
        return tf.cast(blurred, tf.float32)


def stupid_function(x):
    return StupidAttack()(x)