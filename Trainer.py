from .dependencies import *
import csv
class Trainer:

    def writeLosses(self, losses,it):
        # Writing each loss to a CSV file
        file_exists = os.path.isfile(self.output_folder+'/slosses.csv')

        with open(self.output_folder+'/losses.csv', 'a', newline='') as csvfile:
            writer = csv.writer(csvfile)
        
        # Write the header if the file does not exist
            if it==0:
                los=[]
                los.append("Iter")
                for loss in self.losses:
                    los.append(loss.name)
                writer.writerow(los)
            los=[]
            los.append(it)
            for loss in losses:
                los.append(loss)
            writer.writerow(los)



    def __init__(self, model,val_steps=5000,print_steps=5000,output_folder="trainer/"):
        self.model = model
        self.losses = []
        self.lossesW=[]
        self.validators=[]
        self.val_steps=val_steps
        self.print_steps=print_steps
        self.output_folder=output_folder
        self.optimizer =  optim.Rprop(model.parameters(), lr=1e-1)
        self.scheduler = ReduceLROnPlateau(self.optimizer, mode='min', factor=0.999, patience=1000, threshold=1e-2, threshold_mode='rel', cooldown=0, min_lr=1e-5, eps=1e-08)
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
    import time

    def train(self, num_iterations):
        best_val = 1e5
        for it in range(num_iterations):
            start_time = time.time()  # Start timing the iteration

            self.model.zero_grad()
            self.optimizer.zero_grad()
            total_loss = 0
            losses = []
            for weighth, loss_obj in zip(self.lossesW, self.losses):
                loss = loss_obj.forward(self.model)
                total_loss += loss * weighth
                losses.append((loss * weighth).item())

            # Backward pass
            total_loss.backward()

            # Update weights
            self.optimizer.step()
            flag = False if (total_loss > 1e-9) else True
            iteration_time = time.time() - start_time  # Calculate iteration duration

            if it % self.print_steps == 0 or flag:
                print("Iteration {}: total loss {:.4f}, losses: {}, learning rate: {:.10f}, time: {:.4f}s".format(
                    it, total_loss.item(), losses, self.scheduler.get_last_lr()[0], iteration_time))
                self.writeLosses(losses, it)

            if it % self.val_steps == 0 or flag:
                for val_obj in self.validators:
                    vloss = val_obj.val(self.model,True)
                    if vloss < best_val:
                        torch.save(self.model, self.output_folder + "/model")
                        best_val = loss
                    print("Val loss ", vloss)
            self.scheduler.step(total_loss)
            if flag:
                break

        #for p in self.model.parameters():
           # print(p.name, p.data)
        for val_obj in self.validators:
            vloss = val_obj.val(self.model, True)

                    
 

