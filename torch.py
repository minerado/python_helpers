import torch
from timeit import default_timer as timer


def print_train_time(start: float, end: float, device: torch.device = None):
    total_time = end - start

    print(f"Train time on {device}: {total_time:.3f} seconds")

    return total_time
