import torch.nn as nn
import torch

# Base Deep Neural Network
def  SingularLayer(input_size, output):
    out = nn.Sequential(
        nn.Linear(input_size, output),
        nn.ReLU(True)
    ).to('cuda')
    return out

class DeepNeuralNetwork(nn.Module):
    def __init__(self, input_size, output_size, *args, activation=None):
        super(DeepNeuralNetwork, self).__init__()
        
        self.overall_structure = nn.Sequential().to('cuda')
        #Model input and hidden layer
        for num, output in enumerate(args):
            self.overall_structure.add_module(name = f'layer_{num+1}', module = SingularLayer(input_size, output))
            input_size = output

        #Model output layer
        self.output_layer = nn.Sequential(nn.Linear(input_size, output_size)).to('cuda')
        if activation is not None:
            self.output_layer.add_module(activation)
    def forward(self, xb):
        out = self.overall_structure(xb)
        out = self.output_layer(out)
        return out
    
# Attention based RNNs
class Attention(nn.Module):
    def __init__(self, hidden_size):
        super(Attention, self).__init__()
        self.hidden_size = hidden_size
        self.attention_layer = nn.Linear(hidden_size, hidden_size).to('cuda')
        
    def forward(self, hidden_states):
        attention_weights = self.attention_layer(hidden_states)
        attention_weights = torch.tanh(attention_weights)
        attention_weights = torch.softmax(attention_weights, dim=0)
        
        context_vector = torch.sum(attention_weights * hidden_states, dim=0)
        
        return context_vector