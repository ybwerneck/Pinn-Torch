import torch
import numpy as np
import torch.nn.functional as F
import random
#Loss base class
class LOSS_PINN(torch.nn.Module):
    


    ## PDE as loss function. Thus would use the network which we call as u_theta
    def f(self,data_in, model):
        x,w = model(data_in).T # the dependent variable u is given by the network based on independent variables x,t
        ## Based on our f = du/dx - 2du/dt - u, we need du/dx and du/dt
        t,u,v,k=data_in.T
        ## Based on our f = du/dx - 2du/dt - u, we need du/dx and du/dt
        dx_dt= torch.autograd.grad(
        x,
        data_in,
        grad_outputs=torch.ones_like(t),
        create_graph=True,
        retain_graph=True)[0].T[0]
        
        dw_dt= torch.autograd.grad(
        x,
        data_in,
        grad_outputs=torch.ones_like(t),
        create_graph=True,
        retain_graph=True)[0].T[0]
        
        
        pde = torch.abs( 10*((k + 1)*(u*(u-0.4)*(1-u))-v ) - dx_dt)
        return pde

    def batch_generator(self,size):
        return torch.rand(size,4,requires_grad=True).to(torch.device("cuda"))
        

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
  

        return torch.mean((self.f(batch,model)))
    
    
    def loss(self,tgt,pred):
        print("Not defined")
        

    
