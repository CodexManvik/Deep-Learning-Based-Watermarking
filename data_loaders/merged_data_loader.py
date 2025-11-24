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
    Clean, stable, repo-faithful dataloader:
    - Images: finite dataset
    - Watermarks: infinite random bit vectors
    - Attack IDs: infinite random attack choices
    - Perfect alignment: (input → output) watermark matches 1:1
    - Supports Phase-1: attacks disabled
    - Supports Phase-2: attacks enabled
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

        # 1) Finite clean image dataset (exactly like original repo)
        self.image_stream = ImageDataLoader(
            base_path=image_base_path,
            channels=image_channels,
            convert_type=image_convert_type,
            max_images=None  # slicing happens below
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
        """Counts files in directory just like original repo logic."""
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

        # finite image set
        img_ds = self.image_stream.take(int(self.max_images))

        # infinite watermark stream
        wm_ds = self.watermark_stream

        # infinite attack stream
        atk_ds = self.attack_stream

        # zip watermark ONCE so input and output use the same watermark
        # (wm_in, wm_out) are the same tensor
        wm_pair = wm_ds.map(lambda w: (w, w))

        # extract only once, not twice
        wm_in = wm_pair.map(lambda w_in, w_out: w_in)
        wm_out = wm_pair.map(lambda w_in, w_out: w_out)

        x_ds = tf.data.Dataset.zip((img_ds, wm_in, atk_ds))
        y_ds = tf.data.Dataset.zip((img_ds, wm_out))


        ds = tf.data.Dataset.zip((x_ds, y_ds))

        ds = ds.shuffle(buffer_size=2048, reshuffle_each_iteration=True)
        ds = ds.batch(self.batch_size)
        ds = ds.prefetch(self.prefetch)

        return ds

