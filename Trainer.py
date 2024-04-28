from dependencies import *

class Trainer:
    def __init__(self, model,val_steps=1000,print_steps=100,output_folder="trainer/"):
        self.model = model
        self.losses = []
        self.lossesW=[]
        self.validators=[]
        self.val_steps=val_steps
        self.print_steps=print_steps
        self.output_folder=output_folder
        self.optimizer =  optim.Adam(model.parameters(), lr=1e-3)
        try:
            os.mkdir(self.output_folder)
        except:
            print("Folder already there")

    def add_loss(self, loss_obj,weigth=1):
        self.losses.append(loss_obj)
        self.lossesW.append(weigth)
    def add_validator(self,val_obj):
        val_obj.setFolder(self.output_folder)
        self.validators.append(val_obj)
    def train(self,num_iterations):
        for it in range(num_iterations):

            total_loss = 0
            for weighth,loss_obj in zip(self.lossesW,self.losses):
                loss = loss_obj.forward(self.model)
                total_loss += loss*weighth

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
 

