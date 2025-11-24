import tensorflow as tf
from data_loaders.base_data_loader import BaseDataLoader


class AttackIdDataLoader(BaseDataLoader):
    """
    Infinite attack-ID generator.
    Uses tf.data + tf.random.uniform (GPU-safe).
    No Python generator, no stateless RNG issues.
    """

    def __init__(self, min_value, max_value):
        super(AttackIdDataLoader, self).__init__()
        self.min_value = min_value
        self.max_value = max_value

    def get_data_loader(self):
        # Infinite index stream
        ds = tf.data.Dataset.range(10**12)

        # Map to random attack IDs (int32)
        ds = ds.map(
            lambda _: tf.random.uniform(
                shape=(1,),
                minval=self.min_value,
                maxval=self.max_value,
                dtype=tf.int32
            ),
            num_parallel_calls=tf.data.AUTOTUNE
        )

        return ds
