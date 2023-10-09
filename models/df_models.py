from models.macro_architectures import *
import torch.nn as nn
import torch
from models.utils import *

class Simple1DCNN(nn.Module):
    def __init__(self,architecture, input_size, hidden_size, num_heads = None, kernel_size=3, stride=2):
        super(Simple1DCNN, self).__init__()
        self.hidden_size = hidden_size
        self.conv1d = nn.Conv1d(input_size, 10, kernel_size, stride).to('cuda')
        self.relu = nn.ReLU().to('cuda')
        self.maxpool = nn.MaxPool1d(kernel_size=2, stride=2).to('cuda')
        self.fc = DeepNeuralNetwork(40, hidden_size,*architecture).to('cuda')
        ##add attention
        self.num_heads = num_heads
        if num_heads is not None:
            self.attention = nn.MultiheadAttention(input_size, num_heads, batch_first=True).to('cuda')
    def forward(self, x):
        if self.num_heads is not None:
            x,_ = self.attention(x,x,x)
        x = self.conv1d(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = self.fc(x)
        return x


class DeepVanillaRNN(nn.Module):
    def __init__(self, hidden_size, input_size, mlp_architecture, attention):
        super(DeepVanillaRNN, self).__init__()
        self.hidden_size = hidden_size
        self.at = attention
        self.hidden_mlp = DeepNeuralNetwork(hidden_size, hidden_size, *mlp_architecture).to('cuda')
        self.input_mlp = DeepNeuralNetwork(input_size, hidden_size, *mlp_architecture).to('cuda')
        self.attention = Attention(self.hidden_size).to('cuda')
        ##add attention
    def forward(self,x):
        batch_size, seq_len, _ = x.size()

        hn = torch.zeros(batch_size, self.hidden_size, requires_grad=True)
        hn_list = []

        #create here loop for training the entire sequence
        for t in range(seq_len):
            xt = x[:, t, :]# Extract the input at time t
            
            a_t = self.hidden_mlp(hn) + self.input_mlp(xt)
            
            hn = torch.tanh(a_t)

            hn_list.append(hn)
        if self.at:
            hidden_states = torch.stack(hn_list)
            context_vector = self.attention(hidden_states)
        
            return context_vector
        else:
            return hn

class DeepLSTM(nn.Module):
    def __init__(self, hidden_size, input_size, mlp_architecture, attention):
        super(DeepLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.at = attention
        #Forget gate
        self.F_h = DeepNeuralNetwork(hidden_size, hidden_size, *mlp_architecture).to('cuda')
        self.F_x = DeepNeuralNetwork(input_size, hidden_size, *mlp_architecture).to('cuda')
        #Input gate
        self.I_h = DeepNeuralNetwork(hidden_size, hidden_size, *mlp_architecture).to('cuda')
        self.I_x = DeepNeuralNetwork(input_size, hidden_size, *mlp_architecture).to('cuda')
        #Ouput gate
        self.O_h = DeepNeuralNetwork(hidden_size, hidden_size, *mlp_architecture).to('cuda')
        self.O_x = DeepNeuralNetwork(input_size, hidden_size, *mlp_architecture).to('cuda')
        #Input node
        self.C_hat_h = DeepNeuralNetwork(hidden_size, hidden_size, *mlp_architecture).to('cuda')
        self.C_hat_x = DeepNeuralNetwork(input_size, hidden_size, *mlp_architecture).to('cuda')
        self.attention = Attention(self.hidden_size).to('cuda')
    def forward(self,x):
        batch_size, sequence_size, _ = x.size()
        hn = torch.zeros(batch_size, self.hidden_size, requires_grad = True).to('cuda')
        cn = torch.zeros(batch_size, self.hidden_size, requires_grad = True).to('cuda')
        hn_list = []

        for t in range(sequence_size):
            xt = x[:, t, :]
            #forward
            a_F = self.F_h(hn) + self.F_x(xt)
            F = torch.sigmoid(a_F) #forget gate
            a_I = self.I_h(hn) + self.I_x(xt)
            I = torch.sigmoid(a_I) #input gate
            a_O = self.O_h(hn) + self.O_x(xt)
            O = torch.sigmoid(a_O) #output gate
            a_C_hat = self.C_hat_h(hn) + self.C_hat_x(xt)
            C_hat = torch.tanh(a_C_hat)
            cn = F*cn + I*C_hat
            hn = O*torch.tanh(cn)
            hn_list.append(hn)

        if self.at:
            hidden_states = torch.stack(hn_list)
            context_vector = self.attention(hidden_states)
        
            return context_vector
        else:
            return hn

class DeepGRU(nn.Module):
    def __init__(self, hidden_size, input_size, mlp_architecture, attention):
        super(DeepGRU, self).__init__()
        self.hidden_size = hidden_size
        self.attention = Attention(self.hidden_size).to('cuda')
        self.at = attention
        #Update gate
        self.Z_h = DeepNeuralNetwork(hidden_size, hidden_size, *mlp_architecture).to('cuda')
        self.Z_x = DeepNeuralNetwork(input_size, hidden_size, *mlp_architecture).to('cuda')
        #Reset gate
        self.R_h = DeepNeuralNetwork(hidden_size, hidden_size, *mlp_architecture).to('cuda')
        self.R_x = DeepNeuralNetwork(input_size, hidden_size, *mlp_architecture).to('cuda')
        #Possible hidden state
        self.H_hat_h = DeepNeuralNetwork(hidden_size, hidden_size, *mlp_architecture).to('cuda')
        self.H_hat_x = DeepNeuralNetwork(input_size, hidden_size, *mlp_architecture).to('cuda')
    def forward(self, x):
        batch_size, sequence_size, _ = x.size()
        hn = torch.zeros(batch_size, self.hidden_size, requires_grad = True).to('cuda')
        hn_list = []
        for t in range(sequence_size):
            xt = x[:, t, :]
            Z = torch.sigmoid(self.Z_h(hn)+self.Z_x(xt))
            R = torch.sigmoid(self.R_h(hn)+self.R_x(xt))
            H_hat = torch.tanh(self.H_hat_h(hn*R)+self.H_hat_x(xt))
            hn = hn*Z + (torch.ones_like(Z)-Z)*H_hat
            hn_list.append(hn)
        if self.at:
            hidden_states = torch.stack(hn_list)
            context_vector = self.attention(hidden_states)
        
            return context_vector
        else:
            return hn
    

class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, attention = True):
        super(LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.lstm = nn.LSTMCell(input_size, hidden_size).to('cuda')
        self.attention = Attention(self.hidden_size).to('cuda')
        self.at = attention
    def forward(self, x):
        batch_size, sequence_size, _ = x.size()
        hn = torch.zeros(batch_size, self.hidden_size, requires_grad = True).to('cuda')
        cn = torch.zeros(batch_size, self.hidden_size, requires_grad = True).to('cuda')
        hn_list = []
        for t in range(sequence_size):
            
            xt = x[:, t, :]

            hn,cn = self.lstm(xt, (hn,cn))
            hn_list.append(hn)
        if self.at:
            hidden_states = torch.stack(hn_list)
            context_vector = self.attention(hidden_states)
        
            return context_vector
        else:
            return hn
class GRU(nn.Module):
    def __init__(self, input_size, hidden_size, attention):
        super(GRU, self).__init__()
        self.hidden_size = hidden_size
        self.gru = nn.GRUCell(input_size, hidden_size).to('cuda')
        self.attention = Attention(self.hidden_size).to('cuda')
        self.at = attention
    def forward(self, x):
        batch_size, sequence_size, _ = x.size()
        hn = torch.zeros(batch_size, self.hidden_size, requires_grad = True).to('cuda')
        hn_list = []
        for t in range(sequence_size):
            
            xt = x[:, t, :]

            hn = self.gru(xt, hn) 

            hn_list.append(hn)
        if self.at:
            hidden_states = torch.stack(hn_list)
            context_vector = self.attention(hidden_states)
        
            return context_vector
        else:
            return hn
    
class VanillaRNN(nn.Module):
    def __init__(self, input_size, hidden_size, attention = True):
        super(VanillaRNN, self).__init__()
        self.hidden_size = hidden_size
        self.rnn = nn.RNNCell(input_size, hidden_size).to('cuda')
        self.attention = Attention(self.hidden_size).to('cuda')
        self.at = attention
    
    def forward(self, x):
        batch_size, sequence_size, _ = x.size()
        hn = torch.zeros(batch_size, self.hidden_size, requires_grad = True).to('cuda')
        hn_list = []
        for t in range(sequence_size):
            
            xt = x[:, t, :]

            hn = self.rnn(xt, hn)

            hn_list.append(hn)

        if self.at:
            hidden_states = torch.stack(hn_list)
            context_vector = self.attention(hidden_states)
        
            return context_vector
        else:
            return hn
    
class BidirectionalRNN(nn.Module):
    def __init__(self, rnn1, rnn2):
        super(BidirectionalRNN, self).__init__()
        self.rnn1 = rnn1
        self.rnn2 = rnn2
        
    def forward(self, x):
        # Forward pass through the first RNN
        hidden1 = self.rnn1(x)
        # Reverse the input sequence for the second RNN
        x_backward = torch.flip(x, [1])
        
        # Forward pass through the second RNN
        hidden2 = self.rnn2(x_backward)

        # Concatenate to bidirectional output
        hidden_bidirectional = torch.cat((hidden1,hidden2), dim = 1)
        
        return hidden_bidirectional