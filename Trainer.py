import torch.optim as optim
import torch
import tensorflow as tf
class Loss:

    def compute_loss(self, output):
        raise NotImplementedError("Subclass must implement abstract method")

class MSE(Loss):
    def __init__(self, data_in,data_out):
        self.data_in = data_in
        self.data_out= data_out

    def compute_loss(self,model):
        target = self.data_out
        output=model(self.data_in)
        return torch.mean((output - target)**2)


class Trainer:
    def __init__(self, model):
        self.model = model
        self.losses = []
        self.optimizer =  optim.Adam(model.parameters(), lr=1e-3)

    def add_loss(self, loss_obj):
        self.losses.append(loss_obj)

    def train(self,num_iterations):
        for it in range(num_iterations):

            # Sample data points

            # Forward pass
            # Compute total loss
            total_loss = 0
            for loss_obj in self.losses:
                loss = loss_obj.compute_loss(self.model)
                total_loss += loss

            # Backward pass
            total_loss.backward()

            # Update weights
            self.optimizer.step()

            if it % 100 == 0:
                print("Iteration ", it, ": total loss ", total_loss.item())

