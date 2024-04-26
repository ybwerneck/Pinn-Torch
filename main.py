import tensorflow as tf
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import seaborn as sns
import matplotlib.pyplot as plt
from Net import FullyConnectedNetwork
from Trainer import Trainer
from Trainer import MSE
# Check if GPU is availabl

if tf.test.gpu_device_name():
    print('GPU found')
else:
    print("No GPU found. Please ensure you have TensorFlow-GPU installed.")

input_shape = 4  # Example input shape (28x28 image flattened)
output_shape = 2   # Example output shape (10 classes)
hidden_sizes = [128, 64]  # Example hidden layer sizes
model = FullyConnectedNetwork(input_shape, output_shape, hidden_sizes)
print(model)


trainer=Trainer(model)

data_in=np.zeros((4,100)).T
data_out=np.zeros((2,100)).T

loss=MSE(data_in,data_out)

trainer.add_loss(loss)

trainer.train(1000)

print(model)
