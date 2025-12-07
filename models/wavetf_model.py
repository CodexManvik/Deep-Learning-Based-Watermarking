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
        Applies attacks based on the random ID for Robustness Training.
        Wrapped in Lambda to allow symbolic tensor manipulation.
        Uses tf.map_fn to apply different attacks to different images in the batch.
        """
        def batch_attack_logic(inputs):
            image_batch, id_batch = inputs
            
            # This function runs on a single slice (single image, single ID)
            def single_image_attack(args):
                image, atk_id = args
                idx = tf.cast(atk_id[0], tf.int32)

                # Helper to ensure strict float32 output
                def safe_run(attack_fn, x):
                    # Add batch dim for attack func (H,W,C) -> (1,H,W,C)
                    x_expanded = tf.expand_dims(x, 0) 
                    out = attack_fn(x_expanded)
                    # Remove batch dim (1,H,W,C) -> (H,W,C)
                    return tf.squeeze(tf.cast(out, tf.float32), 0)

                return tf.switch_case(
                    idx,
                    branch_fns={
                        0: lambda: image,  # Identity
                        1: lambda: safe_run(salt_pepper_function, image),
                        2: lambda: safe_run(gaussian_noise_function, image),
                        3: lambda: safe_run(jpeg_function, image),
                        4: lambda: safe_run(drop_out_function, image)
                    },
                    default=lambda: image
                )

            # Apply single_image_attack to every element in the batch
            return tf.map_fn(
                single_image_attack, 
                elems=(image_batch, id_batch),
                fn_output_signature=tf.float32
            )

        # *** CRITICAL FIX: Wrap in Lambda so it works in Keras Graph ***
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
        
        # 5. Extract (From Attacked Image)
        extracted_band, _ = self.dwt_forward(attacked_img)
        extracted_wm = self.extract_cnn(extracted_band)

        return Model(
            inputs=[img_in, wm_in, attack_id],
            outputs=[embedded_img, extracted_wm],
            name="WaveTF_Robust_LL"
        )