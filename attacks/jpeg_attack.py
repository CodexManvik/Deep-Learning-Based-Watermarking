import tensorflow as tf
import numpy as np
from attacks.base_attack import BaseAttack


class JPEGAttack(BaseAttack):
    """
    Differentiable JPEG Compression Attack — Grayscale Only.

    Pipeline: Scale to [0,255] -> 8x8 Block Split -> DCT -> Quantize -> IDCT -> Merge -> Normalize

    Paper baseline: quality=50 (fixed for evaluation)
    Training:       quality in [50, 90] (randomized)

    The three-branch color path (YCbCr) has been removed because this model
    operates exclusively on single-channel (grayscale) images. Keeping dead
    RGB branches caused shape mismatches during Keras graph tracing.
    """

    def __init__(self, quality_range=(50, 90), **kwargs):
        super(JPEGAttack, self).__init__()
        self.quality_min = quality_range[0]
        self.quality_max = quality_range[1]

        # Standard JPEG Luminance Quantization Table
        self.std_luminance_quant_tbl = [
            16, 11, 10, 16, 24, 40, 51, 61,
            12, 12, 14, 19, 26, 58, 60, 55,
            14, 13, 16, 24, 40, 57, 69, 56,
            14, 17, 22, 29, 51, 87, 80, 62,
            18, 22, 37, 56, 68, 109, 103, 77,
            24, 35, 55, 64, 81, 104, 113, 92,
            49, 64, 78, 87, 103, 121, 120, 101,
            72, 92, 95, 98, 112, 100, 103, 99
        ]

    def get_quality_matrix(self, quality):
        """Generates the luminance quantization matrix for a given quality factor (0-100)."""
        q = tf.clip_by_value(tf.cast(quality, tf.float32), 1.0, 100.0)
        scale = tf.where(q < 50, 5000.0 / q, 200.0 - 2.0 * q)

        table = np.array(self.std_luminance_quant_tbl, dtype=np.float32).reshape((8, 8))
        t_table = tf.constant(table)
        final_table = tf.floor((scale * t_table + 50.0) / 100.0)
        final_table = tf.clip_by_value(final_table, 1.0, 255.0)
        return tf.reshape(final_table, (1, 1, 1, 1, 8, 8))  # broadcastable shape

    def dct_2d(self, x):
        """2D DCT via two 1D DCTs on the last two axes of a rank-6 tensor."""
        x1 = tf.signal.dct(x, type=2, norm='ortho')
        x1_t = tf.transpose(x1, perm=[0, 1, 2, 3, 5, 4])
        x2 = tf.signal.dct(x1_t, type=2, norm='ortho')
        return tf.transpose(x2, perm=[0, 1, 2, 3, 5, 4])

    def idct_2d(self, x):
        """2D IDCT — inverse of dct_2d."""
        x1 = tf.signal.idct(x, type=2, norm='ortho')
        x1_t = tf.transpose(x1, perm=[0, 1, 2, 3, 5, 4])
        x2 = tf.signal.idct(x1_t, type=2, norm='ortho')
        return tf.transpose(x2, perm=[0, 1, 2, 3, 5, 4])

    def diff_round(self, x):
        """Straight-through estimator for rounding: forward=round, backward=identity."""
        return x + tf.stop_gradient(tf.round(x) - x)

    def jpeg_simulate(self, inputs, quality_tensor):
        """
        Grayscale JPEG simulation.
        inputs: (Batch, H, W, 1) float32 in [0, 1]
        """
        shape = tf.shape(inputs)
        B, H, W = shape[0], shape[1], shape[2]

        # 1. Scale to [0, 255]
        x = inputs * 255.0

        # 2. Pad H and W to nearest multiple of 8
        pad_h = (8 - H % 8) % 8
        pad_w = (8 - W % 8) % 8
        x = tf.pad(x, [[0, 0], [0, pad_h], [0, pad_w], [0, 0]])
        H_p = H + pad_h
        W_p = W + pad_w

        # 3. Reshape into 8x8 blocks: (B, H//8, W//8, 1, 8, 8)
        x = tf.reshape(x, (B, H_p // 8, 8, W_p // 8, 8, 1))
        x = tf.transpose(x, perm=[0, 1, 3, 5, 2, 4])  # (B, Hb, Wb, 1, 8, 8)

        # 4. 2D DCT
        freq = self.dct_2d(x)

        # 5. Quantize (straight-through grad)
        q_matrix = self.get_quality_matrix(quality_tensor)
        quantized = self.diff_round(freq / q_matrix) * q_matrix

        # 6. 2D IDCT
        spatial = self.idct_2d(quantized)

        # 7. Reshape back: (B, Hb, 8, Wb, 8, 1) -> (B, H_p, W_p, 1)
        spatial = tf.transpose(spatial, perm=[0, 1, 4, 2, 5, 3])  # (B, Hb, 8, Wb, 8, 1)
        reconstructed = tf.reshape(spatial, (B, H_p, W_p, 1))

        # 8. Crop padding and normalize back to [0, 1]
        reconstructed = reconstructed[:, :H, :W, :]
        return tf.clip_by_value(reconstructed / 255.0, 0.0, 1.0)

    def call(self, inputs):
        quality = tf.random.uniform(
            [],
            minval=self.quality_min,
            maxval=self.quality_max,
            dtype=tf.float32
        )
        return self.jpeg_simulate(inputs, quality)


def jpeg_function(x):
    return JPEGAttack()(x)


def jpeg_fixed_function(x, quality=50):
    return JPEGAttack(quality_range=(quality, quality))(x)