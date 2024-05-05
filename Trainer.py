
class Trainer:
    def __init__(self, model,val_steps=1000,print_steps=100,output_folder="trainer/"):
        self.model = model
        self.losses = []
        self.lossesW=[]
        self.validators=[]
        self.val_steps=val_steps
        self.print_steps=print_steps
        self.output_folder=output_folder
        self.optimizer =  optim.Adam(model.parameters(), lr=1e-2)
        self.scheduler = ReduceLROnPlateau(self.optimizer, mode='min', factor=0.5, patience=5, threshold=0.00001, threshold_mode='rel', cooldown=0, min_lr=0.00001, eps=1e-08)
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
            self.model.zero_grad()
            total_loss = 0
            losses=[]
            for weighth,loss_obj in zip(self.lossesW,self.losses):
                loss = loss_obj.forward(self.model)
                total_loss += loss*weighth
                losses.append((loss*weighth).item())

            # Backward pass
            self.optimizer.zero_grad()  # Reset gradients

            total_loss.backward()

            # Update weights
            self.optimizer.step()
           
            if it % self.print_steps == 0:
              print("Iteration {}: total loss {:.4f}, losses: {}, learning rate: {:.10f}".format(it, total_loss.item(), losses, self.scheduler.get_last_lr()[0]))
                
                
            if it % self.val_steps ==0:
                for val_obj in self.validators:
                    vloss=val_obj.val(self.model)
                    self.scheduler.step(vloss)

                    
 

