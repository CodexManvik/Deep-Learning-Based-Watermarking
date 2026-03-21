from typing import Tuple
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import (
    Input, Conv2D, Conv2DTranspose, BatchNormalization, Activation,
    Reshape, Concatenate, Lambda
)
from tensorflow.keras.models import Model
from wavetf import WaveTFFactory
from configs import *

# Import Attacks
from attacks.gaussian_noise_attack import gaussian_noise_function
from attacks.salt_pepper_attack import salt_pepper_function
from attacks.stupid_attack import stupid_function
from attacks.drop_out_attack import drop_out_function
from attacks.jpeg_attack import jpeg_function
from attacks.rotation_attack import rotation_function

class WaveTFModel:
    """
    Wavelet-based watermarking model targeting the LL BAND.
    Features:
    - Attack Simulator Enabled (Robustness Training)
    - Paper-compliant Architecture (Strided Extraction)
    - Paper-compliant Scaling (/2 and *2)
    - TYPE-SAFE Attack Layer (Fixes int32/float32 mismatch)
    - BATCH-SAFE Attack Application (Fixes Mixed Batch Bugs)
    """

    def __init__(
        self,
        image_size: Tuple[int],
        watermark_size: Tuple[int],
        wavelet_type: str = "haar",
        delta_scale: float = delta_scale
    ):
        self.image_size = image_size
        self.watermark_size = watermark_size
        self.wavelet_type = wavelet_type
        self.delta_scale = float(delta_scale)

        self.wm_side = int(np.sqrt(self.watermark_size[0]))
        self.embed_filters = [64, 64, 64]
        self.extract_filters = [128, 256] 
        self.wm_pre_filters = [256, 128, 64]

    def dwt_forward(self, img):
        full = WaveTFFactory().build(self.wavelet_type, dim=2)(img)
        # [cite_start]LL Band (Channel 0) - Scaled by 0.5 per Paper [cite: 130-131]
        target_band = full[..., 0:1] / 2.0 
        return target_band, full

    def dwt_inverse(self, full):
        return WaveTFFactory().build(self.wavelet_type, dim=2, inverse=True)(full)

    def preprocess_watermark(self, wm_in, target_h: int, target_w: int):
        """
        Paper Section 4.2:
        Reshape -> ConvT(512) -> BN -> ReLU -> ConvT(128) -> BN -> ReLU -> ConvT(1) -> Tanh/ReLU
        The paper uses 3 blocks to upscale 16x16 -> 128x128 (LL band size).
        """
        # 1. Reshape to 2D (Batch, 16, 16, 1)
        x = Reshape((self.wm_side, self.wm_side, 1), name="reshape_watermark")(wm_in)
        
        # 2. Upsample Block 1: 16 -> 32
        x = Conv2DTranspose(512, (3, 3), strides=(2, 2), padding="same")(x)
        x = BatchNormalization()(x)
        x = Activation("relu")(x)

        # 3. Upsample Block 2: 32 -> 64
        x = Conv2DTranspose(128, (3, 3), strides=(2, 2), padding="same")(x)
        x = BatchNormalization()(x)
        x = Activation("relu")(x)

        # 4. Upsample Block 3: 64 -> 128 (Target LL Size)
        x = Conv2DTranspose(1, (3, 3), strides=(2, 2), padding="same")(x)
        
        # Ensure output matches target shape exactly (handles edge cases in padding)
        x = Lambda(lambda t: tf.image.resize(t, (target_h, target_w)), name="wm_resize")(x)
        
        return x

    def embed_cnn(self, x):
        for f in self.embed_filters:
            x = Conv2D(f, (3, 3), padding="same")(x)
            x = BatchNormalization()(x)
            x = Activation("relu")(x)
        
        delta = Conv2D(1, (3, 3), padding="same")(x)
        delta = Activation("tanh")(delta)
        return delta

    def extract_cnn(self, x):
        # Strided Extraction (Paper Match)
        for f in self.extract_filters:
            x = Conv2D(f, (3, 3), strides=2, padding="same")(x)
            x = BatchNormalization()(x)
            x = Activation("relu")(x)

        x = Conv2D(1, (3, 3), strides=2, padding="same")(x) 
        x = Activation("sigmoid")(x)
        return Reshape(self.watermark_size, name="output_watermark")(x)

    def attack_layer(self, img, attack_id):
        """
        Applies ONE randomly selected attack to the entire batch.

        Per-batch attack selection (not per-image) is the standard approach in
        watermarking research. Coverage is identical over training since each
        batch independently draws from the weighted distribution.

        XLA-COMPATIBLE: uses tf.cond chain (no switch_case / map_fn / variant ops).
        FAST: only ONE attack function runs per batch step (~6x faster than running all 7).
        """
        def batch_attack_logic(inputs):
            image_batch, id_batch = inputs

            # All images in the batch share the same attack ID (per-batch sampling)
            # id_batch shape: (B, 1) — take the first element as the batch-level ID
            batch_id = tf.cast(id_batch[0, 0], tf.int32)
            batch_id = tf.clip_by_value(batch_id, 0, 6)

            def apply(attack_fn):
                return tf.cast(attack_fn(image_batch), tf.float32)

            # tf.cond chain: only ONE branch executes per step (XLA-safe)
            result = tf.cond(tf.equal(batch_id, 1),
                lambda: apply(salt_pepper_function),
                lambda: tf.cond(tf.equal(batch_id, 2),
                    lambda: apply(gaussian_noise_function),
                    lambda: tf.cond(tf.equal(batch_id, 3),
                        lambda: apply(jpeg_function),
                        lambda: tf.cond(tf.equal(batch_id, 4),
                            lambda: apply(drop_out_function),
                            lambda: tf.cond(tf.equal(batch_id, 5),
                                lambda: apply(rotation_function),
                                lambda: tf.cond(tf.equal(batch_id, 6),
                                    lambda: apply(stupid_function),
                                    lambda: tf.cast(image_batch, tf.float32)  # ID 0: identity
                                )
                            )
                        )
                    )
                )
            )
            return result

        # Wrap in Lambda so it works in Keras Graph
        attacked = Lambda(batch_attack_logic, name="attack_simulator")([img, attack_id])
        
        # Explicitly restore shape info (Lambda often loses it)
        return Reshape(self.image_size, name="restore_attack_shape")(attacked)

    def get_model(self):
        img_in = Input(self.image_size, name="image_input")
        wm_in = Input(self.watermark_size, name="watermark_input")
        attack_id = Input((1,), name="attack_id_input", dtype="int32")

        # 1. DWT & Preprocess
        target_band, full = self.dwt_forward(img_in)
        h, w = int(target_band.shape[1]), int(target_band.shape[2])
        
        # Corrected Preprocessing (Transpose Conv)
        wm_pre = self.preprocess_watermark(wm_in, h, w)

        # 2. Embed
        merged = Concatenate(axis=-1)([target_band, wm_pre])
        delta = self.embed_cnn(merged)
        new_band = Lambda(lambda t, s=self.delta_scale: t[0] + s * t[1], name="new_band")([target_band, delta])

        # 3. Reconstruct
        rest = Lambda(lambda f: f[..., 1:], name="wavelet_rest")(full)
        # Scale LL back up by 2.0 per Paper
        new_band_scaled = Lambda(lambda x: x * 2.0, name="rescale_band")(new_band)
        full_mod = Concatenate(axis=-1)([new_band_scaled, rest])

        embedded_raw = self.dwt_inverse(full_mod)
        embedded_img = Lambda(lambda x: tf.clip_by_value(x, 0.0, 1.0), name="embedded_image")(embedded_raw)

        # 4. Attack (Robustness Layer)
        attacked_img = self.attack_layer(embedded_img, attack_id)
        attacked_img = Lambda(lambda x: x, name="attacked_image")(attacked_img)
        
        # 5. Extract (From Attacked Image)
        extracted_band, _ = self.dwt_forward(attacked_img)
        extracted_wm = self.extract_cnn(extracted_band)

        # In wavetf_model.py, change the return to:
        return Model(
            inputs=[img_in, wm_in, attack_id],
            outputs=[embedded_img, extracted_wm, attacked_img],  # Add attacked_img
            name="WaveTF_Robust_LL"
        )
