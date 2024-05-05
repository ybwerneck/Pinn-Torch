from dependencies import *

#Loss base class
class LOSS_PINN(torch.nn.Module):
  

    def __init__(self,Pinn_func,batch_generator,batch_size=10000,shuffle=False):
        self.f=Pinn_func
        self.batch_generator=batch_generator

        self.batch_size=batch_size
        self.shuffle=shuffle
        
        
        super(LOSS_PINN, self).__init__()


    def getBatch(self):
        batch=self.batch_generator(self.batch_size)

        return batch
    
    def forward(self,model):
        batch  = self.getBatch()
  

        return torch.mean((self.f(batch,model))) +torch.max((self.f(batch,model))) 
    
    
    def loss(self,tgt,pred):
        print("Not defined")
        
