from .dependencies import *
from .Loss import *
from .Validator import *

def default_batch_generator(size, ranges,device):
    batch = torch.empty((len(ranges),size)).uniform_(0, 1).T  # Initialize with random values within [-1, 1]
    
    for i, (min_val, max_val) in enumerate(ranges):
        batch[:, i] = batch[:, i] * (max_val - min_val) + (min_val)
    return batch.requires_grad_().to(device)

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

def FHN_VAL_fromDataSet(folder,name="Val",device=torch.device("cpu"),dtype=torch.float64):
        data_folder=folder
        T = np.load(data_folder + "T.npy")
        K = np.load(data_folder + "K.npy")
        U = np.load(data_folder + "U.npy")
        V = np.load(data_folder + "V.npy")
        SOLs = np.load(data_folder + "SOLs.npy")
        SOLw = np.load(data_folder + "SOLw.npy")
        print(np.shape(T))
        print(np.shape(U))
        print(np.shape(T))
        print(np.shape(K))
        data_in=torch.tensor(np.stack((T)),dtype=dtype).T.to(device)
        data_out=torch.tensor(np.stack((SOLs,SOLw)),dtype=dtype).T.to(device)
        return Validator(data_in,data_out,name,device)

def FHN_loos_fromDataSet(folder, batch_size=10000, device=torch.device("cpu"), loss_type="MSE",shuffle=True,dtype=torch.float64):
        data_folder = folder
        T = np.load(data_folder + "T.npy")
        K = np.load(data_folder + "K.npy")
        U = np.load(data_folder + "U.npy")
        V = np.load(data_folder + "V.npy")
        SOLs = np.load(data_folder + "SOLs.npy")
        SOLw = np.load(data_folder + "SOLw.npy")
        data_in=torch.tensor(np.stack((T)),dtype=dtype).T.to(device)
        data_out=torch.tensor(np.stack((SOLs,SOLw)),dtype=dtype).T.to(device)
        
        if loss_type == "MSE":
            return MSE(data_in, data_out, batch_size,shuffle)
        elif loss_type == "MAE":
            return MAE(data_in, data_out, batch_size,shuffle)
        elif loss_type == "KLDivergenceLoss":
            return KLDivergenceLoss(data_in, data_out, batch_size,shuffle)
        elif loss_type == "CosineSimilarityLoss":
            return CosineSimilarityLoss(data_in, data_out, batch_size,shuffle)
        elif loss_type == "CosineSimilarityLoss":
            return CosineSimilarityLoss(data_in, data_out, batch_size,shuffle)
        elif loss_type.startswith("L"):
            try:
                p = int(loss_type[1:])
                return LPthLoss(data_in, data_out, batch_size,shuffle=shuffle,device=device, p=p)
            except ValueError:
                raise ValueError(f"Invalid value for p in L-pth loss: {loss_type}")
   
        else:
            raise ValueError(f"Unknown loss type: {loss_type}")