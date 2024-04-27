import torch.optim as optim
import torch
import tensorflow as tf
import torch.nn as nn
import matplotlib.pyplot as plt
import csv
import numpy as np
import os
class MSE(torch.nn.Module):
    
    
    def __init__(self,data_in,target,batch_size=10000):
        self.data_in=torch.tensor(data_in,dtype=torch.float32)
        self.target=torch.tensor(target,dtype=torch.float32)
        self.batch_size=batch_size
        self.i=0
        self.len_d=len(target)
        super(MSE, self).__init__()

    
    def forward(self,model):
        batch=self.data_in[self.i:self.i+self.batch_size]
        tgt=self.target[self.i:self.i+self.batch_size]
        pred= model(batch)
        self.i+=self.batch_size
        if(self.i>self.len_d):
            self.i=0
        return torch.mean((torch.abs(tgt) -torch.abs(pred))**2)

class Validator():
    
    def __init__(self,data_intarget,target,name="val"):
        self.data_in=torch.tensor(data_intarget,dtype=torch.float32)
        self.target=torch.tensor(target,dtype=torch.float32)
        self.name=name
 
    def setFolder(self,folder):
        self.folder=folder
        with open(self.folder+"/"+self.name+".csv", mode='w', newline='') as file:
            writer = csv.writer(file)

    def val(self,model):
        p=model(self.data_in)
        e=(self.target - p).detach().numpy()
        plt.plot(self.target[:400,0])
        plt.plot(p[:400,0].detach().numpy())
        plt.savefig(self.folder+"/"+self.name+".png")
        plt.clf()
                # Write data to CSV file
        with open(self.folder+"/"+self.name+".csv", mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([np.mean(e),np.max(e)])
        with open(self.folder+"/"+"target"+".csv", mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerows(self.target.detach().numpy())  
        with open(self.folder+"/"+"pred"+".csv", mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerows(p.detach().numpy())     
        with open(self.folder+"/"+"input"+".csv", mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerows(self.data_in.detach().numpy())  

        
import os

class Trainer:
    def __init__(self, model,val_steps=100,print_steps=100,output_folder="trainer/"):
        self.model = model
        self.losses = []
        self.validators=[]
        self.val_steps=100
        self.print_steps=100
        self.output_folder=output_folder
        self.optimizer =  optim.Adam(model.parameters(), lr=1e-3)
        try:
            os.mkdir(self.output_folder)
        except:
            print("Folder already there")

    def add_loss(self, loss_obj):
        self.losses.append(loss_obj)
    def add_validator(self,val_obj):
        val_obj.setFolder(self.output_folder)
        self.validators.append(val_obj)
    def train(self,num_iterations):
        for it in range(num_iterations):

            total_loss = 0
            for loss_obj in self.losses:
                loss = loss_obj.forward(self.model)
                total_loss += loss

            # Backward pass
            self.optimizer.zero_grad()  # Reset gradients

            total_loss.backward()

            # Update weights
            self.optimizer.step()

            if it % self.print_steps == 0:
                print("Iteration ", it, ": total loss ", total_loss.item())
            if it % self.val_steps ==0:
                for val_obj in self.validators:
                    val_obj.val(self.model)
 

