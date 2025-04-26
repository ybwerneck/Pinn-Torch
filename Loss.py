from .dependencies import *
#Loss base class
def ensure_at_least_one_column(x):
    # If x is 1-dimensional, reshape to (len(x), 1)
    if x.ndim == 1:
        return x.reshape(-1, 1)
    # If x is already 2-dimensional, return it unchanged
    elif x.ndim == 2:
        return x
    else:
        raise ValueError("Input array must be 1D or 2D")
    

class LOSS(torch.nn.Module):
    
    def __init__(self,data_in,target,batch_size=10000,shuffle=False,dtype=torch.float32,device=-1,name="Loss"):
        self.name=name
        
        if shuffle:
            datac,tc=ensure_at_least_one_column(data_in.to("cpu")),ensure_at_least_one_column(target.to("cpu"))
            
            combined = torch.cat((datac, tc), dim=1)
            combined = combined[torch.randperm(combined.size(0))]
            
       
            data_in, target = torch.split(combined, [len(datac.T), len(tc.T)], dim=1)
     

        self.device=device
        print(device)
        self.data_in=data_in.to(device )
        self.target=target.to(device)
        self.batch_size=batch_size
        self.i=0
        self.len_d=len(target)
        
        
        super(LOSS, self).__init__()

    def getBatch(self):
        batch=self.data_in[self.i:self.i+self.batch_size]
        tgt=self.target[self.i:self.i+self.batch_size]
        self.i+=self.batch_size
        if(self.i+self.batch_size>len(self.data_in)):
            self.i=0

        return batch.to(self.device),tgt.to(self.device)
    
    def forward(self,model):
        batch,tgt=self.getBatch()
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
    def __init__(self, data_in, target, batch_size=10000, p=2, shuffle=False, device=torch.device("cpu"),name="LPthLoss"):
        super(LPthLoss, self).__init__(data_in, target, batch_size, shuffle,device=device,name=name)
        self.p = p

    def loss(self, tgt, pred):
        return torch.mean(torch.pow(torch.sum(torch.pow(torch.abs(tgt - pred), self.p), axis=1), 1/self.p))
