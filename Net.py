from dependencies import *

class FullyConnectedNetwork(nn.Module):
    def __init__(self, input_shape, output_shape,hidden_sizes=0):
        super(FullyConnectedNetwork, self).__init__()
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.layers = nn.ModuleList()
        if(hidden_sizes!=0):
            self.initFullyConnected(input_shape, output_shape,hidden_sizes)
        
    def initFullyConnected(self, input_shape, output_shape,hidden_sizes):
        in_features = input_shape
        for hidden_size in hidden_sizes:
            self.layers.append(nn.Linear(in_features, hidden_size))
            
            self.layers.append(nn.Tanh())  # Adding ReLU activation function
            in_features = hidden_size
        self.layers.append(nn.Linear(in_features, output_shape))

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
    
class FullyConnectedNetworkMod(FullyConnectedNetwork):
    def __init__(self, input_shape, output_shape, hidden_sizes):
        super(FullyConnectedNetworkMod, self).__init__(input_shape, output_shape)
        in_features = input_shape
        for act,hidden_size in hidden_sizes:
            self.layers.append(nn.Linear(in_features, hidden_size))
            
            self.layers.append(act())  # Adding ReLU activation function
            in_features = hidden_size
            
            
        self.layers.append(nn.Linear(in_features, output_shape))
