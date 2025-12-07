from typing import List, Tuple, Optional
from pathlib import Path
import tensorflow as tf

from data_loaders.base_data_loader import BaseDataLoader
from data_loaders.image_data_loaders.image_data_loader import ImageDataLoader
from data_loaders.watermark_data_loaders.watermark_data_loader import WatermarkDataLoader
from data_loaders.attack_id_data_loader.attack_id_data_loader import AttackIdDataLoader
from data_loaders.configs import PREFETCH

class MergedDataLoader(BaseDataLoader):
    """
    Fixed Data Loader:
    - Zips (Image, Watermark, Attack) FIRST to ensure synchronization.
    - Prevents the 'Random Stream Fork' bug where input and target watermarks differed.
    """

    def __init__(
        self,
        image_base_path: str,
        image_channels: List[int],
        image_convert_type,
        watermark_size: Tuple[int],
        attack_min_id: int,
        attack_max_id: int,
        batch_size: int,
        prefetch=PREFETCH,
        max_images: Optional[int] = None
    ):
        super(MergedDataLoader, self).__init__()

        self.batch_size = batch_size
        self.prefetch = prefetch
        self.max_images = max_images
        self.image_base_path = image_base_path

        # 1) Finite clean image dataset
        self.image_stream = ImageDataLoader(
            base_path=image_base_path,
            channels=image_channels,
            convert_type=image_convert_type,
            max_images=None 
        ).get_data_loader()

        # 2) Infinite watermark generator
        self.watermark_stream = WatermarkDataLoader(
            watermark_size=watermark_size
        ).get_data_loader()

        # 3) Infinite attack-id generator
        self.attack_stream = AttackIdDataLoader(
            min_value=attack_min_id,
            max_value=attack_max_id
        ).get_data_loader()

    def _count_image_files(self) -> int:
        try:
            p = Path(self.image_base_path)
            return len([f for f in p.glob("*") if f.is_file()])
        except Exception:
            return 0

    def get_data_loader(self):
        if self.max_images is None:
            guessed = self._count_image_files()
            if guessed <= 0:
                raise ValueError(f"No images found at {self.image_base_path}")
            self.max_images = guessed

        # Limit images to the finite set
        img_ds = self.image_stream.take(int(self.max_images))

        # Zip strictly ONCE: (Image, Watermark, AttackID)
        # This locks the random watermark to the image for this step
        combined_ds = tf.data.Dataset.zip((img_ds, self.watermark_stream, self.attack_stream))

        # Now map to the format Keras expects: ((Inputs), (Outputs))
        # Inputs: [Image, Watermark, AttackID]
        # Outputs: [OriginalImage (for MSE), Watermark (for Extract Loss)]
        def format_batch(img, wm, atk):
            inputs = (img, wm, atk)
            targets = (img, wm) # <--- wm here is IDENTICAL to wm in inputs
            return inputs, targets

        ds = combined_ds.map(format_batch, num_parallel_calls=tf.data.AUTOTUNE)

        ds = ds.shuffle(buffer_size=2048, reshuffle_each_iteration=True)
        ds = ds.batch(self.batch_size)
        ds = ds.prefetch(self.prefetch)

        return ds