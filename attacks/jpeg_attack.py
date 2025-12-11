import tensorflow as tf
import numpy as np
from attacks.base_attack import BaseAttack

class JPEGAttack(BaseAttack):
    """
    True Differentiable JPEG Attack (DiffJPEG).
    Pipeline: RGB -> YCbCr -> 8x8 Block Split -> DCT -> Quantization -> IDCT -> Block Merge -> RGB
    ** FIXED: Dimensions for Grayscale/Color inputs (Rank 6 Tensor support) **
    """
    """
    Differentiable JPEG Compression Attack
    
    Paper baseline: quality=50 (fixed for evaluation)
    Training: quality ∈ [50, 90] (randomized for robustness)
    
    The random range teaches the model to survive various compression levels.
    Lower quality = more aggressive compression = harder attack.
    """

    def __init__(self, quality_range=(50, 90), **kwargs):
        super(JPEGAttack, self).__init__()
        self.quality_min = quality_range[0]
        self.quality_max = quality_range[1]
        # Standard JPEG Quantization Tables
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
        self.std_chrominance_quant_tbl = [
            17, 18, 24, 47, 99, 99, 99, 99,
            18, 21, 26, 66, 99, 99, 99, 99,
            24, 26, 56, 99, 99, 99, 99, 99,
            47, 66, 99, 99, 99, 99, 99, 99,
            99, 99, 99, 99, 99, 99, 99, 99,
            99, 99, 99, 99, 99, 99, 99, 99,
            99, 99, 99, 99, 99, 99, 99, 99,
            99, 99, 99, 99, 99, 99, 99, 99
        ]

    def get_quality_matrix(self, quality):
        """Generates the quantization matrices based on quality factor (0-100)."""
        q = tf.clip_by_value(tf.cast(quality, tf.float32), 1.0, 100.0)
        scale = tf.where(q < 50, 5000.0 / q, 200.0 - 2.0 * q)
        
        def build_matrix(std_table):
            table = np.array(std_table, dtype=np.float32).reshape((8, 8))
            t_table = tf.constant(table)
            final_table = tf.floor((scale * t_table + 50.0) / 100.0)
            final_table = tf.clip_by_value(final_table, 1.0, 255.0)
            return tf.reshape(final_table, (1, 1, 1, 1, 8, 8)) # Fixed shape for broadcasting

        return build_matrix(self.std_luminance_quant_tbl), build_matrix(self.std_chrominance_quant_tbl)

    def rgb_to_ycbcr(self, image):
        matrix = np.array(
            [[0.299, 0.587, 0.114],
             [-0.168736, -0.331264, 0.5],
             [0.5, -0.418688, -0.081312]], dtype=np.float32).T
        bias = np.array([0, 128, 128], dtype=np.float32)
        return tf.tensordot(image * 255.0, matrix, axes=1) + bias

    def ycbcr_to_rgb(self, image):
        matrix = np.array(
            [[1.0, 0.0, 1.402],
             [1.0, -0.344136, -0.714136],
             [1.0, 1.772, 0.0]], dtype=np.float32).T
        bias = np.array([0, 128, 128], dtype=np.float32)
        rgb = tf.tensordot(image - bias, matrix, axes=1)
        return rgb / 255.0

    def dct_2d(self, x):
        # Input shape: (Batch, H_blk, W_blk, C, 8, 8) -> Rank 6
        # Apply DCT on last axis (cols)
        X1 = tf.signal.dct(x, type=2, norm='ortho')
        # Transpose last two dimensions to apply DCT on rows
        # Permutation: 0,1,2,3 stay. Swap 4 and 5.
        X1_t = tf.transpose(X1, perm=[0, 1, 2, 3, 5, 4])
        X2 = tf.signal.dct(X1_t, type=2, norm='ortho')
        # Transpose back
        return tf.transpose(X2, perm=[0, 1, 2, 3, 5, 4])

    def idct_2d(self, x):
        # Input shape: (Batch, H_blk, W_blk, C, 8, 8) -> Rank 6
        X1 = tf.signal.idct(x, type=2, norm='ortho')
        X1_t = tf.transpose(X1, perm=[0, 1, 2, 3, 5, 4])
        X2 = tf.signal.idct(X1_t, type=2, norm='ortho')
        return tf.transpose(X2, perm=[0, 1, 2, 3, 5, 4])

    def diff_round(self, x):
        return x + tf.stop_gradient(tf.round(x) - x)

    def jpeg_simulate(self, inputs, quality_tensor):
        # 0. Check Channels
        is_grayscale = (inputs.shape[-1] == 1)

        # 1. Color Conversion
        if is_grayscale:
            ycbcr = inputs * 255.0
        else:
            ycbcr = self.rgb_to_ycbcr(inputs)

        # 2. Pad to multiple of 8
        shape = tf.shape(ycbcr)
        H, W = shape[1], shape[2]
        pad_h = (8 - H % 8) % 8
        pad_w = (8 - W % 8) % 8
        ycbcr_pad = tf.pad(ycbcr, [[0, 0], [0, pad_h], [0, pad_w], [0, 0]])
        
        H_p = H + pad_h
        W_p = W + pad_w
        
        # 3. Reshape to Blocks
        C = 1 if is_grayscale else 3
        blocks = tf.reshape(ycbcr_pad, (shape[0], H_p // 8, 8, W_p // 8, 8, C))
        # Transpose to (Batch, H_blocks, W_blocks, C, 8, 8)
        blocks = tf.transpose(blocks, perm=[0, 1, 3, 5, 2, 4]) 

        # 4. DCT
        freq_blocks = self.dct_2d(blocks)
        
        # 5. Quantization
        q_y, q_c = self.get_quality_matrix(quality_tensor)
        
        if is_grayscale:
            y_channel = freq_blocks
            y_q = self.diff_round(y_channel / q_y) * q_y
            quantized_freqs = y_q
        else:
            y_channel = freq_blocks[..., 0:1, :, :]
            c_channels = freq_blocks[..., 1:3, :, :]
            y_q = self.diff_round(y_channel / q_y) * q_y
            c_q = self.diff_round(c_channels / q_c) * q_c
            quantized_freqs = tf.concat([y_q, c_q], axis=-3)
        
        # 6. IDCT
        spatial_blocks = self.idct_2d(quantized_freqs)
        
        # 7. Reshape back
        spatial_blocks = tf.transpose(spatial_blocks, perm=[0, 1, 4, 2, 5, 3])
        reconstructed_pad = tf.reshape(spatial_blocks, (shape[0], H_p, W_p, C))
        
        # Crop
        reconstructed = reconstructed_pad[:, :H, :W, :]
        
        # 8. Inverse Color Conversion
        if is_grayscale:
            return tf.clip_by_value(reconstructed / 255.0, 0.0, 1.0)
        else:
            return tf.clip_by_value(self.ycbcr_to_rgb(reconstructed), 0.0, 1.0)

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