from typing import Tuple
import tensorflow as tf
from data_loaders.base_data_loader import BaseDataLoader
from configs import ATTACKS_DISABLED

class AttackIdDataLoader(BaseDataLoader):
    """
    Attack ID Generator with Paper-Compliant Distribution.

    Distribution (when attacks enabled):
    - 25%   No Attack       (ID 0, weight 2)
    - 12.5% Salt & Pepper   (ID 1, weight 1)
    - 12.5% Gaussian Noise  (ID 2, weight 1)
    - 12.5% JPEG            (ID 3, weight 1)
    - 12.5% Dropout         (ID 4, weight 1)
    - 12.5% Rotation        (ID 5, weight 1)
    - 12.5% Stupid Attack   (ID 6, weight 1)

    When attacks_disabled=True (or ATTACKS_DISABLED=True in configs),
    always yields ID 0 (identity pass-through).
    """

    def __init__(
        self,
        min_value: int,
        max_value: int,
        use_paper_distribution: bool = True,
        attacks_disabled: bool = ATTACKS_DISABLED
    ):
        super(AttackIdDataLoader, self).__init__()
        self.min_value = min_value
        self.max_value = max_value
        self.use_paper_distribution = use_paper_distribution
        self.attacks_disabled = attacks_disabled

        # Attack IDs: 0=None, 1=Salt, 2=Gauss, 3=JPEG, 4=Dropout, 5=Rotation, 6=Stupid
        # Weight 2 for no-attack gives ~25% no-attack, ~12.5% per real attack.
        self.paper_weights = [2.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
    
    def get_data_loader(self):
        # Attacks disabled: always stream identity (no-attack) ID
        if self.attacks_disabled:
            return tf.data.Dataset.from_tensors(
                tf.constant([[0]], dtype=tf.int32)
            ).repeat()

        # Handle fixed attack ID case (used during evaluation)
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
