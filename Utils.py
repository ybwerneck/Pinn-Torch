from dependencies import *
def default_batch_generator(size, ranges):
    batch = torch.empty((len(ranges),size)).uniform_(0, 1)  # Initialize with random values within [-1, 1]
    
    for i, (min_val, max_val) in enumerate(ranges):
        batch[:, i] = batch[:, i] * (max_val - min_val) + (min_val)
    return batch.requires_grad_().to(torch.device("cuda")).T

