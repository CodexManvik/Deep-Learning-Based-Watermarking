from pathlib import Path
from typing import List, Optional
import tensorflow as tf

from data_loaders.base_data_loader import BaseDataLoader
from data_loaders.configs import IMAGE_FORMATS


class ImageDataLoader(BaseDataLoader):
    """
    Clean, stable, repo-faithful image loader:
    - Loads images from disk
    - Resizes to (256, 256)
    - Converts to grayscale using the first channel (consistent with paper)
    - Returns float32 normalized images
    - Optimized for tf.data (parallel map + AUTOTUNE)
    """

    def __init__(
        self,
        base_path: str,
        channels: List[int],
        convert_type=None,
        image_size=(256, 256),
        images_format=IMAGE_FORMATS,
        max_images: Optional[int] = None
    ):
        super(ImageDataLoader, self).__init__()
        self.base_path = base_path
        self.channels = channels     # unused but kept for API compatibility
        self.convert_type = convert_type
        self.image_size = image_size
        self.images_format = images_format
        self.max_images = max_images

    # ------------------------------------------------------------
    # 1. Get list of file paths (with max_images cap)
    # ------------------------------------------------------------
    def _collect_files(self):
        p = Path(self.base_path)

        # Accept multiple formats from IMAGE_FORMATS list
        file_paths = []
        if isinstance(self.images_format, (list, tuple)):
            for fmt in self.images_format:
                file_paths += list(p.glob(f"*.{fmt}"))
        else:
            file_paths = list(p.glob(f"*.{self.images_format}"))

        file_paths = sorted(map(str, file_paths))

        # Cap dataset if max_images is set
        if self.max_images is not None:
            file_paths = file_paths[:self.max_images]

        if len(file_paths) == 0:
            raise ValueError(f"No images found in {self.base_path}")

        return file_paths

    # ------------------------------------------------------------
    # 2. Load single image -> grayscale -> resize -> normalize
    # ------------------------------------------------------------
    def _load_image(self, path: tf.Tensor) -> tf.Tensor:
        img_bytes = tf.io.read_file(path)

        # decode jpeg/png/etc. 3 channels
        img = tf.image.decode_image(img_bytes, channels=3, expand_animations=False)

        # use channel 0 (consistent with your code)
        img = img[:, :, 0:1]   # shape: (H, W, 1)

        # resize to target size
        img = tf.image.resize(img, self.image_size)

        # normalize / convert dtype
        if self.convert_type is not None:
            img = tf.image.convert_image_dtype(img, self.convert_type)
        else:
            img = tf.cast(img, tf.float32) / 255.0

        return img

    # ------------------------------------------------------------
    # 3. Build final dataset
    # ------------------------------------------------------------
    def get_data_loader(self):
        files = self._collect_files()

        ds = tf.data.Dataset.from_tensor_slices(files)

        # map loader with AUTOTUNE for parallel decoding
        ds = ds.map(
            lambda p: self._load_image(p),
            num_parallel_calls=tf.data.AUTOTUNE
        )

        return ds
