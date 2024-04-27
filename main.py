import tensorflow as tf
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from Net import FullyConnectedNetwork
from Trainer import Trainer
from Trainer import Validator
from Trainer import MSE
# Check if GPU is availabls


input_shape = 4  # Example input shape (28x28 image flattened)
output_shape = 2   # Example output shape (10 classes)
hidden_sizes = [6]  # Example hidden layer sizes
model = FullyConnectedNetwork(input_shape, output_shape, hidden_sizes)
print(model)

data_folder="training_data/treino/"
T = np.load(data_folder + "T.npy")
K = np.load(data_folder + "K.npy")
U = np.load(data_folder + "U.npy")
V = np.load(data_folder + "V.npy")
SOLs = np.load(data_folder + "SOLs.npy")
SOLw = np.load(data_folder + "SOLw.npy")

trainer=Trainer(model)

data_in=torch.tensor(np.stack((T,K,V,U))).T
data_out=torch.tensor(np.stack((U + K*T,V + K*T))).T

loss=MSE(data_in,data_out)

trainer.add_loss(loss)

loss=MSE(data_in,data_out)

trainer.add_loss(loss)
trainer.add_validator(Validator(data_in,data_out,"val"))


trainer.train(1000)

print(model)
