import tensorflow as tf
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from Net import FullyConnectedNetworkMod
from Trainer import Trainer
from Validator import Validator
from Loss import *

# Check if GPU is availabls

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)


input_shape = 4  
output_shape = 2   
hidden_layer = [(nn.Tanh,64), (nn.ReLU,32),(nn.ReLU,64),(nn.Tanh,32)]
model = FullyConnectedNetworkMod(input_shape, output_shape, hidden_layer).to(device)



print(model)

data_folder="training_data/treino/"

trainer=Trainer(model)




trainer.add_loss(LOSS.fromDataSet(data_folder,1024,device=device,loss_type="L4"),weigth=100)
trainer.add_loss(LOSS.fromDataSet(data_folder,1024,device=device,loss_type="L2"),weigth=1)

trainer.add_validator(Validator.fromDataSet("training_data/validation/",device=device))


trainer.train(100000)

print(model)
