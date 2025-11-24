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
class WaveTFModel:
    """
    Wavelet-based watermarking model targeting the LL BAND.
    - Target: LL (Channel 0) - High Robustness / High Sensitivity
    - Extractor: Strided Convolutions (Smart)
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
        # TARGETING LL (Index 0)
        # Paper Code Line 40: Divides by 2!
        target_band = full[..., 0:1] / 2.0 
        return target_band, full

    def dwt_inverse(self, full):
        return WaveTFFactory().build(self.wavelet_type, dim=2, inverse=True)(full)

    def preprocess_watermark(self, wm_in, target_h: int, target_w: int):
        x = Reshape((self.wm_side, self.wm_side, 1), name="reshape_watermark")(wm_in)
        for f in self.wm_pre_filters:
            x = Conv2DTranspose(f, (3, 3), strides=(2, 2), padding="same")(x)
            x = BatchNormalization()(x)
            x = Activation("relu")(x)
        
        x = Conv2D(64, (3, 3), padding="same")(x)
        x = BatchNormalization()(x)
        x = Activation("relu")(x)
        x = Lambda(lambda t: tf.image.resize(t, (target_h, target_w)), name="wm_resize")(x)
        x = Conv2D(64, (3, 3), padding="same")(x)
        x = BatchNormalization()(x)
        x = Activation("relu")(x)
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
        # Strided Extraction (Crucial for LL)
        for f in self.extract_filters:
            x = Conv2D(f, (3, 3), strides=2, padding="same")(x)
            x = BatchNormalization()(x)
            x = Activation("relu")(x)

        x = Conv2D(1, (3, 3), strides=2, padding="same")(x) 
        x = Activation("sigmoid")(x)
        return Reshape(self.watermark_size, name="output_watermark")(x)

    def attack_layer(self, img, attack_id):
        return img

    def get_model(self):
        img_in = Input(self.image_size, name="image_input")
        wm_in = Input(self.watermark_size, name="watermark_input")
        attack_id = Input((1,), name="attack_id_input", dtype="int32")

        # 1. Get LL Band
        target_band, full = self.dwt_forward(img_in)
        
        h = int(target_band.shape[1])
        w = int(target_band.shape[2])

        # 2. Preprocess
        wm_pre = self.preprocess_watermark(wm_in, h, w)

        # 3. Embed
        merged = Concatenate(axis=-1)([target_band, wm_pre])
        delta = self.embed_cnn(merged)
        
        # Apply Delta to LL
        new_band = Lambda(lambda t, s=self.delta_scale: t[0] + s * t[1], name="new_band")([target_band, delta])

        # 4. Reconstruct
        # Keep LH, HL, HH (Indices 1, 2, 3)
        rest = Lambda(lambda f: f[..., 1:], name="wavelet_rest")(full)
        
        # Scale LL back up by 2
        new_band_scaled = Lambda(lambda x: x * 2.0, name="rescale_band")(new_band)
        
        # Combine [New LL] + [LH, HL, HH]
        full_mod = Concatenate(axis=-1)([new_band_scaled, rest])

        embedded_raw = self.dwt_inverse(full_mod)
        embedded_img = Lambda(lambda x: tf.clip_by_value(x, 0.0, 1.0), name="embedded_image")(embedded_raw)

        # 5. Attack & Extract
        attacked_img = self.attack_layer(embedded_img, attack_id)
        extracted_band, _ = self.dwt_forward(attacked_img)
        extracted_wm = self.extract_cnn(extracted_band)

        return Model(
            inputs=[img_in, wm_in, attack_id],
            outputs=[embedded_img, extracted_wm],
            name="WaveTF_Paper_LL"
        )