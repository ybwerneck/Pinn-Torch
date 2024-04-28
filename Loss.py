import torch
import numpy as np
import torch.nn.functional as F
import random
#Loss base class
class LOSS(torch.nn.Module):
    
    @staticmethod
    def fromDataSet(folder, batch_size=10000, device=torch.device("cpu"), loss_type="MSE",shuffle=True):
        data_folder = folder
        T = np.load(data_folder + "T.npy")
        K = np.load(data_folder + "K.npy")
        U = np.load(data_folder + "U.npy")
        V = np.load(data_folder + "V.npy")
        SOLs = np.load(data_folder + "SOLs.npy")
        SOLw = np.load(data_folder + "SOLw.npy")
        data_in=torch.tensor(np.stack((T,K,V,U)),dtype=torch.float32).T.to(device)
        data_out=torch.tensor(np.stack((SOLs,SOLw)),dtype=torch.float32).T.to(device)
        
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
                return LPthLoss(data_in, data_out, batch_size,shuffle=shuffle, p=p)
            except ValueError:
                raise ValueError(f"Invalid value for p in L-pth loss: {loss_type}")
   
        else:
            raise ValueError(f"Unknown loss type: {loss_type}")

    
    def __init__(self,data_in,target,batch_size=10000,shuffle=False):
        
        if shuffle:
            combined = torch.cat((data_in, target), dim=1)
            combined = combined[torch.randperm(combined.size(0))]
            data_in, target = torch.split(combined, [4, 2], dim=1)

        self.data_in=data_in
        self.target=target
        self.batch_size=batch_size
        self.i=0
        self.len_d=len(target)
        
        
        super(LOSS, self).__init__()

    def getBatch(self):
        batch=self.data_in[self.i:self.i+self.batch_size]
        tgt=self.target[self.i:self.i+self.batch_size]
        self.i+=self.batch_size

        return batch,tgt
    
    def forward(self,model):
        batch,tgt= self.getBatch()
        pred= model(batch)
        
        if(self.i>self.len_d):
            self.i=0
        return self.loss(tgt,pred)
    
    def loss(self,tgt,pred):
        print("Not defined")
        

    
       
    
class MAE(LOSS):
    def __init__(self, data_in, target, batch_size=10000, shuffle=False):
        super(MAE, self).__init__(data_in, target, batch_size, shuffle)

    def loss(self, tgt, pred):
        return torch.mean(torch.abs(tgt - pred))


class MSE(LOSS):
    def __init__(self, data_in, target, batch_size=10000, shuffle=False):
        super(MSE, self).__init__(data_in, target, batch_size, shuffle)

    def loss(self, tgt, pred):
        return torch.mean((tgt - pred) ** 2)


class KLDivergenceLoss(LOSS):
    def __init__(self, data_in, target, batch_size=10000, shuffle=False):
        super(KLDivergenceLoss, self).__init__(data_in, target, batch_size, shuffle)

    def loss(self, tgt, pred):
        return F.kl_div(pred, tgt)


class CosineSimilarityLoss(LOSS):
    def __init__(self, data_in, target, batch_size=10000, shuffle=False):
        super(CosineSimilarityLoss, self).__init__(data_in, target, batch_size, shuffle)

    def loss(self, tgt, pred):
        return 1 - F.cosine_similarity(pred, tgt)


class LPthLoss(LOSS):
    def __init__(self, data_in, target, batch_size=10000, p=2, shuffle=False):
        super(LPthLoss, self).__init__(data_in, target, batch_size, shuffle)
        self.p = p

    def loss(self, tgt, pred):
        return torch.mean(torch.pow(torch.abs(tgt - pred), self.p))
