import tensorflow as tf
from attacks.base_attack import BaseAttack

class RotationAttack(BaseAttack):
    """
    Rotation Attack: ±15° random rotation (paper uses factor=0.042 ≈ 15°/360°)
    
    Fixed: Removed global layer state to prevent singleton variable errors.
    Uses functional tf.image rotation instead of Keras layer.
    """
    
    def __init__(self, **kwargs):
        super(RotationAttack, self).__init__()
        self.max_rotation_radians = 0.2618  # 15 degrees in radians (15 * π/180)
    
    def rotation(self, inputs):
        """
        Apply random rotation using tf.image functional API.
        This avoids the singleton variable issue with RandomRotation layer.
        """
        batch_size = tf.shape(inputs)[0]
        
        # Random rotation angles for each image in batch (in radians)
        angles = tf.random.uniform(
            shape=[batch_size],
            minval=-self.max_rotation_radians,
            maxval=self.max_rotation_radians,
            dtype=tf.float32
        )
        
        # Apply rotation using tf.image.rot90 approximation via affine transform
        # Or use tfa.image.rotate if tensorflow-addons is available
        try:
            import tensorflow_addons as tfa
            rotated = tfa.image.rotate(
                inputs,
                angles,
                interpolation='bilinear',
                fill_mode='reflect'
            )
        except ImportError:
            # Fallback: Use simple 90° rotations only (less realistic but dependency-free)
            # Randomly apply 0, 90, 180, 270 degree rotations
            k = tf.random.uniform(shape=[batch_size], minval=0, maxval=4, dtype=tf.int32)
            
            def rotate_single(args):
                img, num_rotations = args
                return tf.image.rot90(img, k=num_rotations)
            
            rotated = tf.map_fn(
                rotate_single,
                (inputs, k),
                fn_output_signature=tf.float32
            )
        
        return rotated
    
    def call(self, inputs):
        return self.rotation(inputs)

def rotation_function(x):
    return RotationAttack()(x)
