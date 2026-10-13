import numpy as np
from scvi.train import TrainingPlan
from typing import Optional


def _compute_kl_weight(
    epoch: int,
    step: int,
    n_epochs_kl_warmup: Optional[int],
    n_steps_kl_warmup: Optional[int],
    max_kl_weight: float = 1.0,
    min_kl_weight: float = 0.0,
) -> float:
    """Computes the kl weight for the current step or epoch.

    If both `n_epochs_kl_warmup` and `n_steps_kl_warmup` are None `max_kl_weight` is returned.

    Parameters
    ----------
    epoch
        Current epoch.
    step
        Current step.
    n_epochs_kl_warmup
        Number of training epochs to scale weight on KL divergences from
        `min_kl_weight` to `max_kl_weight`
    n_steps_kl_warmup
        Number of training steps (minibatches) to scale weight on KL divergences from
        `min_kl_weight` to `max_kl_weight`
    max_kl_weight
        Maximum scaling factor on KL divergence during training.
    min_kl_weight
        Minimum scaling factor on KL divergence during training.
    """
    if n_epochs_kl_warmup < 400:
        max_beta = (n_epochs_kl_warmup / 400)
    else:
        max_beta = 1
    return kl_weight_scheduler(epoch, cycle_length=int(n_epochs_kl_warmup/4), max_beta=max_beta)


def kl_weight_scheduler(epoch, cycle_length=100, ramp_up_fraction=0.5, max_beta=1.0, min_beta=0.0):
    """
    Compute KL divergence weight using a sigmoid function.

    Args:
        epoch (int): Current training epoch.
        total_epochs (int): Total number of epochs for training.
        midpoint (float): The midpoint (as a fraction of total epochs) where the weight is 0.5.
        scale (float): Controls the steepness of the sigmoid curve.
    
    Returns:
        float: KL divergence weight for the current epoch.
    """
    # progress = epoch / total_epochs
    # return 1 / (1 + np.exp(-scale * (progress - midpoint)))
    cycle_step = epoch % cycle_length
    ramp_length = int(cycle_length * ramp_up_fraction)
    
    if cycle_step < ramp_length:
        beta = min_beta + (max_beta - min_beta) * (cycle_step / ramp_length)
    else:
        beta = max_beta

    return beta

class AjustTrainingPlan(TrainingPlan):
    @property
    def kl_weight(self):
        """Scaling factor on KL divergence during training."""
        return _compute_kl_weight(
            self.current_epoch,
            self.global_step,
            self.n_epochs_kl_warmup,
            self.n_steps_kl_warmup,
            self.max_kl_weight,
            self.min_kl_weight,
        )
    
