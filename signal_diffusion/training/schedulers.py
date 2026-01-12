"""Learning rate schedulers with warmup for classification training."""

import math
from typing import Literal

from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR


SchedulerType = Literal["constant", "linear", "cosine"]


def get_constant_schedule_with_warmup(
    optimizer: Optimizer,
    num_warmup_steps: int,
) -> LambdaLR:
    """
    Create a schedule with a constant learning rate preceded by a warmup period.

    Args:
        optimizer: The optimizer to schedule
        num_warmup_steps: Number of steps for the warmup phase

    Returns:
        LambdaLR scheduler
    """
    def lr_lambda(current_step: int) -> float:
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        return 1.0

    return LambdaLR(optimizer, lr_lambda)


def get_linear_schedule_with_warmup(
    optimizer: Optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    min_lr_ratio: float | None = None,
) -> LambdaLR:
    """
    Create a schedule with a linear decay preceded by a warmup period.

    The learning rate linearly increases during warmup, then linearly decays
    to min_lr_ratio * initial_lr over the remaining training steps.

    Args:
        optimizer: The optimizer to schedule
        num_warmup_steps: Number of steps for the warmup phase
        num_training_steps: Total number of training steps
        min_lr_ratio: Minimum learning rate as a ratio of initial lr (default None = 0.0)

    Returns:
        LambdaLR scheduler
    """
    if min_lr_ratio is None:
        min_lr_ratio = 0.0

    def lr_lambda(current_step: int) -> float:
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(num_training_steps - current_step) / float(max(1, num_training_steps - num_warmup_steps))
        return max(min_lr_ratio, progress * (1.0 - min_lr_ratio) + min_lr_ratio)

    return LambdaLR(optimizer, lr_lambda)


def get_cosine_schedule_with_warmup(
    optimizer: Optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    num_cycles: float = 0.5,
    min_lr_ratio: float = 0.0,
) -> LambdaLR:
    """
    Create a schedule with a cosine annealing decay preceded by a warmup period.

    The learning rate linearly increases during warmup, then follows a cosine
    curve down to min_lr_ratio * initial_lr over the remaining training steps.

    Args:
        optimizer: The optimizer to schedule
        num_warmup_steps: Number of steps for the warmup phase
        num_training_steps: Total number of training steps
        num_cycles: Number of cosine cycles (default 0.5 for half cycle)
        min_lr_ratio: Minimum learning rate as a ratio of initial lr (default 0.0)

    Returns:
        LambdaLR scheduler
    """
    def lr_lambda(current_step: int) -> float:
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))

        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        cosine_decay = 0.5 * (1.0 + math.cos(math.pi * num_cycles * 2.0 * progress))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine_decay

    return LambdaLR(optimizer, lr_lambda)


def create_scheduler(
    optimizer: Optimizer,
    scheduler_type: SchedulerType,
    num_warmup_steps: int,
    num_training_steps: int,
    **kwargs,
) -> LambdaLR:
    """
    Create a learning rate scheduler with warmup.

    Args:
        optimizer: The optimizer to schedule
        scheduler_type: Type of scheduler ("constant", "linear", or "cosine")
        num_warmup_steps: Number of steps for the warmup phase
        num_training_steps: Total number of training steps
        **kwargs: Additional arguments for specific schedulers:
            - min_lr_ratio (float): For linear/cosine schedulers, minimum lr ratio (default None/0.0)
            - num_cycles (float): For cosine scheduler, number of cycles (default 0.5)

    Returns:
        LambdaLR scheduler

    Raises:
        ValueError: If scheduler_type is not recognized
    """
    if scheduler_type == "constant":
        return get_constant_schedule_with_warmup(optimizer, num_warmup_steps)
    elif scheduler_type == "linear":
        min_lr_ratio = kwargs.get("min_lr_ratio", None)
        return get_linear_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, min_lr_ratio)
    elif scheduler_type == "cosine":
        num_cycles = kwargs.get("num_cycles", 0.5)
        min_lr_ratio = kwargs.get("min_lr_ratio", 0.0)
        return get_cosine_schedule_with_warmup(
            optimizer, num_warmup_steps, num_training_steps, num_cycles, min_lr_ratio
        )
    else:
        raise ValueError(
            f"Unknown scheduler_type: {scheduler_type}. "
            f"Must be one of: 'constant', 'linear', 'cosine'"
        )
