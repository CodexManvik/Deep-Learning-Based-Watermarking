from typing import Tuple
import tensorflow as tf
from data_loaders.base_data_loader import BaseDataLoader

class AttackIdDataLoader(BaseDataLoader):
    """
    Attack ID Generator with Paper-Compliant Distribution.
    
    Paper uses:
    - 1/3 (33%) No Attack (ID 0)
    - 1/6 (16.7%) each for 4 attacks: Salt&Pepper, Gaussian, JPEG, Dropout
    
    This translates to weights: [2, 1, 1, 1, 1, 0, 0] (sum=6)
    """
    
    def __init__(
        self,
        min_value: int,
        max_value: int,
        use_paper_distribution: bool = True
    ):
        super(AttackIdDataLoader, self).__init__()
        self.min_value = min_value
        self.max_value = max_value
        self.use_paper_distribution = use_paper_distribution
        
        # Paper-compliant attack weights (Table 1 in paper)
        # Attack IDs: 0=None, 1=Salt, 2=Gauss, 3=JPEG, 4=Dropout, 5=Rotation, 6=Stupid
        self.paper_weights = [2.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0]  # Rotation/Stupid not in paper
    
    def get_data_loader(self):
        # Handle fixed attack ID case (for evaluation)
        if self.min_value == self.max_value:
            return tf.data.Dataset.from_tensors(
                tf.constant([[self.min_value]], dtype=tf.int32)
            ).repeat()
        
        # Training case: weighted or uniform random
        if self.use_paper_distribution:
            return self._weighted_attack_generator()
        else:
            return self._uniform_attack_generator()
    
    def _weighted_attack_generator(self):
        """
        Paper-compliant weighted sampling.
        Matches Table 1: 1/3 no-attack, 1/6 each for 4 attacks.
        """
        def sample_weighted_attack():
            # Sample from categorical distribution
            logits = tf.constant(self.paper_weights[:self.max_value + 1])
            attack_id = tf.random.categorical(
                tf.math.log([logits + 1e-10]),  # Add epsilon to avoid log(0)
                num_samples=1,
                dtype=tf.int32
            )
            return tf.reshape(attack_id, [1])
        
        return tf.data.Dataset.from_tensors(0).repeat().map(
            lambda _: sample_weighted_attack(),
            num_parallel_calls=tf.data.AUTOTUNE
        )
    
    def _uniform_attack_generator(self):
        """Uniform random attack selection (legacy mode)."""
        def random_attack_id():
            return tf.random.uniform(
                shape=[1],
                minval=self.min_value,
                maxval=self.max_value + 1,
                dtype=tf.int32
            )
        
        return tf.data.Dataset.from_tensors(0).repeat().map(
            lambda _: random_attack_id(),
            num_parallel_calls=tf.data.AUTOTUNE
        )
