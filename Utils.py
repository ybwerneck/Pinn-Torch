from dependencies import *
def default_batch_generator(size, ranges):
    batch = torch.empty((len(ranges),size)).uniform_(0, 1)  # Initialize with random values within [-1, 1]
    
    for i, (min_val, max_val) in enumerate(ranges):
        batch[:, i] = batch[:, i] * (max_val - min_val) + (min_val)
    return batch.requires_grad_().to(torch.device("cuda")).T

def cp_batch_generator(size, ranges):
    distribution = []

    for min_val, max_val in ranges:
        if min_val == max_val:
            # If min_val equals max_val, set the value manually
            value = min_val  # or max_val, they are the same
            distribution.append(cp.Normal(value, 0))  # Use a normal distribution with std=0 for a fixed value
        else:
            distribution.append(cp.Uniform(min_val, max_val))

    samples = cp.sample(distribution, size)
    batch = torch.tensor(samples).T  # Transpose to match the shape of the output
    return batch.requires_grad_().to(torch.device("cuda"))
