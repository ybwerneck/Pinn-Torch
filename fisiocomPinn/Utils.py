from .dependencies import *
from .Loss import *
from .Validator import *


def default_batch_generator(size, ranges, device):
    batch = (
        torch.empty((len(ranges), size)).uniform_(0, 1).T
    )  # Initialize with random values within [-1, 1]

    for i, (min_val, max_val) in enumerate(ranges):
        batch[:, i] = batch[:, i] * (max_val - min_val) + (min_val)
    return batch.requires_grad_().to(device)


def cp_batch_generator(size, ranges):
    distribution = []

    for min_val, max_val in ranges:
        if min_val == max_val:
            # If min_val equals max_val, set the value manually
            value = min_val  # or max_val, they are the same
            distribution.append(
                cp.Normal(value, 0)
            )  # Use a normal distribution with std=0 for a fixed value
        else:
            distribution.append(cp.Uniform(min_val, max_val))

    samples = cp.sample(distribution, size)
    batch = torch.tensor(samples).T  # Transpose to match the shape of the output
    return batch.requires_grad_().to(torch.device("cuda"))


def LoadDataSet(
    folder,
    data_in=["T.npy", "U.npy", "V.npy"],
    data_out=["SOLs.npy", "SOLw.npy"],
    device=torch.device("cpu"),
    dtype=torch.float64,
):
    data_folder = folder
    ind = []
    outd = []

    for file in data_in:
        ind.append(np.load(data_folder + file))

        # print(f'READ {file}, with shape {np.shape(ind[-1])}')
    for file in data_out:
        outd.append(np.load(data_folder + file))
        # print(f'READ {file}, with shape {np.shape(outd[-1])}')

    data_in = torch.tensor(np.stack(ind), dtype=dtype).T.to(device)
    data_out = torch.tensor(np.stack(outd), dtype=dtype).T.to(device)

    return data_in, data_out


def FHN_loos_fromDataSet(
    data_in,
    data_out,
    batch_size=10000,
    device=torch.device("cpu"),
    loss_type="MSE",
    shuffle=True,
    dtype=torch.float64,
):

    if loss_type == "MSE":
        return MSE(data_in, data_out, batch_size, shuffle)
    elif loss_type == "MAE":
        return MAE(data_in, data_out, batch_size, shuffle)
    elif loss_type == "KLDivergenceLoss":
        return KLDivergenceLoss(data_in, data_out, batch_size, shuffle)
    elif loss_type == "CosineSimilarityLoss":
        return CosineSimilarityLoss(data_in, data_out, batch_size, shuffle)
    elif loss_type == "CosineSimilarityLoss":
        return CosineSimilarityLoss(data_in, data_out, batch_size, shuffle)
    elif loss_type.startswith("L"):
        try:
            p = int(loss_type[1:])
            return LPthLoss(
                data_in, data_out, batch_size, shuffle=shuffle, device=device, p=p
            )
        except ValueError:
            raise ValueError(f"Invalid value for p in L-pth loss: {loss_type}")

    else:
        raise ValueError(f"Unknown loss type: {loss_type}")


import torch
import numpy as np
from scipy.integrate import solve_ivp


def generate_dataset(ode_func, t_span, y0, num_points):
    t = np.linspace(t_span[0], t_span[1], num_points)
    sol = solve_ivp(ode_func, t_span, y0, t_eval=t)
    return sol.t, sol.y.T


def default_file_val_plot(val_obj, dump, file="Results_plotter.py"):
    dump_func = lambda val_obj, dump: [
        val_obj.dump_f_def(dump=dump),
        (
            0
            if dump == False
            else subprocess.Popen(
                f"python {file} {val_obj.folder}/", shell=True, stdout=subprocess.PIPE
            ).stdout.read()
        ),
    ]
    return dump_func(val_obj, dump=dump)


def FHN_VAL_fromODE(
    ode_func,
    t_span,
    y0,
    num_points,
    name="Val",
    device=torch.device("cpu"),
    dtype=torch.float64,
    dump_factor=0,
):
    T, Y = generate_dataset(ode_func, t_span, y0, num_points)
    data_in = torch.tensor(T, dtype=dtype).to(device)
    data_out = torch.tensor(Y, dtype=dtype).to(device)

    dump_func = lambda val_obj, dump: [
        val_obj.dump_f_def(dump=dump),
        subprocess.Popen(
            f"python Results_plotter.py {val_obj.folder}/",
            shell=True,
            stdout=subprocess.PIPE,
        ).stdout.read(),
    ]

    return Validator(data_in, data_out, name, device, dump_factor, dump_func)


def FHN_LOSS_fromODE(
    ode_func,
    t_span,
    y0,
    num_points=10240,
    batch_size=1024,
    shuffle=True,
    device=torch.device("cpu"),
    dtype=torch.float64,
    folder=0,
):
    T, Y = generate_dataset(ode_func, t_span, y0, num_points)
    data_in = torch.tensor(T, dtype=dtype).to(device).view(-1, 1)
    data_out = torch.tensor(Y, dtype=dtype).to(device)
    #   print(np.shape(data_out))
    # print(np.shape(data_in))
    if folder != 0:
        plt.scatter(data_in.cpu(), data_out.T[0].cpu(), label="Training  points")
        plt.savefig(f"{folder}/traininig_data.png")
    return LPthLoss(data_in, data_out, batch_size, shuffle=True, device=device, p=2)


def grad(outputs, inputs):
    return torch.autograd.grad(
        outputs, inputs, grad_outputs=torch.ones_like(outputs), create_graph=True
    )
