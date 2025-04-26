from .dependencies import *
import csv
class Trainer:

    def writeLosses(self, losses,it,it_time):
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

        with open(self.output_folder+'/it_time.csv', 'a', newline='') as csvfile:
                writer2 = csv.writer(csvfile)
                if it==0:
                    writer2.writerow(["ITTIME"])

                writer2.writerow([it_time])

    def dump_model_stats_and_histograms(self, it,model, filename="model_stats_hist.txt"):
        all_grads = []
        all_weights = []

        with open(filename, "a") as f:
            f.write(f"it{it}\n")
            f.write("Layer-wise Statistics:\n")
            
            # Iterate over model parameters
            for name, param in model.named_parameters():
                if param.grad is not None:
                    grad_abs = param.grad.detach().abs().cpu().flatten()  # Absolute value of gradients
                    grad_mean = grad_abs.mean().item()
                    grad_std = grad_abs.std().item()
                    grad_min = grad_abs.min().item()
                    grad_max = grad_abs.max().item()
                    all_grads.append(grad_abs)  # Collect for model-wide histogram
                else:
                    grad_mean = grad_std = grad_min = grad_max = None  # No gradients computed yet

                weight_abs = param.data.detach().abs().cpu().flatten()  # Absolute value of weights
                weight_mean = weight_abs.mean().item()
                weight_std = weight_abs.std().item()
                weight_min = weight_abs.min().item()
                weight_max = weight_abs.max().item()
                all_weights.append(weight_abs)  # Collect for model-wide histogram

                # Write per-layer statistics
                f.write(f"\nLayer: {name}\n")
                f.write(f"  Weights - mean={weight_mean:.6e}, std={weight_std:.6e}, min={weight_min:.6e}, max={weight_max:.6e}\n")
                if grad_mean is not None:
                    f.write(f"  Gradients - mean={grad_mean:.6e}, std={grad_std:.6e}, min={grad_min:.6e}, max={grad_max:.6e}\n")
                else:
                    f.write(f"  Gradients - No gradient computed\n")

            # Concatenate tensors for model-wide histograms
            all_grads = torch.cat(all_grads) if all_grads else torch.tensor([], dtype=torch.float32)
            all_weights = torch.cat(all_weights)

            f.write("\nModel-Wide Statistics:\n")
            
            # Compute histograms
            if all_grads.numel() > 0:
                hist_grad, edges_grad = torch.histogram(all_grads, bins=10)
                f.write("\n  Gradient Magnitude Histogram:\n")
                for i in range(len(hist_grad)):
                    f.write(f"    Bin {i}: {hist_grad[i].item()} (range: {edges_grad[i].item():.6f} - {edges_grad[i+1].item():.6f})\n")

            hist_weight, edges_weight = torch.histogram(all_weights, bins=10)
            f.write("\n  Weight Magnitude Histogram:\n")
            for i in range(len(hist_weight)):
                f.write(f"    Bin {i}: {hist_weight[i].item()} (range: {edges_weight[i].item():.6f} - {edges_weight[i+1].item():.6f})\n")


    def __init__(self, model,val_steps=5000,print_steps=5000,output_folder="trainer/"):
        self.model = model
        
        self.losses = []
        self.lossesW=[]
        self.validators=[]
        self.val_steps=val_steps
        self.print_steps=print_steps
        self.output_folder=output_folder
        self.optimizer = optim.Adam(model.parameters(), lr=1e-3)
        self.scheduler = ReduceLROnPlateau(self.optimizer, mode='min', factor=0.9999, patience=1000, threshold=1e-3, threshold_mode='rel', cooldown=0, min_lr=1e-5, eps=1e-08)
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
        patience = 1e2          # Allowed consecutive validations with no improvement
        tol = 1e-3             # Minimum improvement to reset patience
        counter = 0
        if(self.losses==[]):
            print("No loss function added")
            return
        for it in range(num_iterations):
            start_time = time.time()  # Start timing the iteration

            self.model.zero_grad()
            self.optimizer.zero_grad()
            total_loss = 0
            losses = []
            for weighth, loss_obj in zip(self.lossesW, self.losses):
                loss = loss_obj.forward(self.model)
                if((weighth==5 and it >50000) or weighth!=5):
                    total_loss += loss * weighth
                losses.append((loss * weighth).item())

            # Backward pass
            total_loss.backward()

            # Update weights
            self.optimizer.step()
            flag = False if (total_loss > 1e-15) else True
            iteration_time = time.time() - start_time  # Calculate iteration duration

            if it % self.print_steps == 0 or flag:
                print("Iteration {}: total loss {:.4f}, losses: {}, learning rate: {:.10f}, time: {:.4f}s".format(
                    it, total_loss.item(), losses, self.scheduler.get_last_lr()[0], iteration_time))
                self.writeLosses(losses, it,iteration_time)

            if it % self.val_steps == 0 or flag:
                self.dump_model_stats_and_histograms(it,self.model,self.output_folder+"/grad.txt")

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

                    
 

        if it % self.val_steps == 0:
            for val_obj in self.validators:
                vloss = val_obj.val(self.model, True)
                if vloss < best_val - tol:
                    best_val = vloss
                    counter = 0
                    torch.save(self.model, self.output_folder + "/model")
                else:
                    counter += 1
                print("Val loss:", vloss)
                if counter >= patience:
                    print(f"No improvement for {patience} validations. Early stopping.")
                    break