import tensorflow as tf
from data_loaders.base_data_loader import BaseDataLoader


class WatermarkDataLoader(BaseDataLoader):
    """
    Generates infinite random binary watermark vectors.
    Uses pure tf.data ops (NO Python generator → NO PyFunc → NO RNG key errors).
    Fully GPU-safe and compatible with batching.
    """

    def __init__(self, watermark_size):
        super(WatermarkDataLoader, self).__init__()
        self.watermark_size = watermark_size

    def get_data_loader(self):
        # Infinite dummy counter dataset
        ds = tf.data.Dataset.range(10**12)

        # For each index, generate a random 0/1 watermark vector
        ds = ds.map(
            lambda _: tf.random.uniform(
                shape=self.watermark_size,
                minval=0,
                maxval=2,      # binary → values 0 or 1
                dtype=tf.int32
            ),
            num_parallel_calls=tf.data.AUTOTUNE
        )

        # Convert to float (model expects float)
        ds = ds.map(lambda x: tf.cast(x, tf.float32))

        return ds
