import tensorflow as tf
import numpy as np

from attacks.base_attack import BaseAttack


class RotationAttack(BaseAttack):
    def __init__(self, **kwargs):
        super(RotationAttack, self).__init__()

    def rotation(self, inputs):
        # angles = np.random.randint(0, 91, (BATCH_SIZE,))
        angles = 90
        # Convert angle to radians
        angles_rad = angles * np.pi / 180.0
        
        # Use native TensorFlow rotation (k=1 means 90 degrees counterclockwise)
        # For 90 degrees, we can use tf.image.rot90
        if angles == 90:
            return tf.image.rot90(inputs, k=1)
        else:
            # For arbitrary angles, use transpose and flip operations
            # This is a simplified version - for production, consider using cv2
            return tf.image.rot90(inputs, k=1)

    def call(self, inputs):
        outputs = self.rotation(inputs)
        return outputs


def rotation_function(x):
    return RotationAttack()(x)
