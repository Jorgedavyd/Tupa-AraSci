from models.macro_architectures import get_lr
from models.utils import *
import torch.nn.functional as F
import torch.nn as nn
import torch

#training for seq2seq with feature separation
class MultiHead2MultiHeadBase(nn.Module):
    def training_step(self, batch):
        fc, mg, target = batch #decompose batch
        pred = self(fc, mg)        
        #dst index or kp
        loss = F.mse_loss(pred, target)
        return loss

    def validation_step(self, batch):
        fc, mg, target = batch #decompose batch
        pred = self(fc, mg)        
        #dst index or kp
        loss = F.mse_loss(pred, target)
        return {'val_loss': loss.detach()}

    def validation_epoch_end(self, outputs):
        batch_losses = [x['val_loss'] for x in outputs]
        epoch_loss = torch.stack(batch_losses).mean()   # Sacar el valor expectado de todo el conjunto de costos
        return {'val_loss': epoch_loss.item()}

    def epoch_end(self, epoch, result): # Seguimiento del entrenamiento
        print("Epoch [{}]:\n\tlast_lr: {:.5f}\n\ttrain_loss: {:.4f}\n\tval_loss: {:.4f}".format(
            epoch, result['lrs'][-1], result['train_loss'], result['val_loss']))
    
    def evaluate(self, val_loader):
        self.eval()
        outputs = [self.validation_step(batch) for batch in val_loader]
        return self.validation_epoch_end(outputs)
    
    def fit(self, epochs, max_lr, train_loader, val_loader,
                  weight_decay=0, grad_clip=False, opt_func=torch.optim.Adam):
        torch.cuda.empty_cache()
        history = [] # Seguimiento de entrenamiento

        # Poner el método de minimización personalizado
        optimizer = opt_func(self.parameters(), max_lr, weight_decay=weight_decay)
        # Learning rate scheduler, le da momento inicial al entrenamiento para converger con valores menores al final
        sched = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr, epochs=epochs,
                                                    steps_per_epoch=len(train_loader))

        for epoch in range(epochs):
            # Training Phase
            self.train()  #Activa calcular los vectores gradiente
            train_losses = []
            lrs = [] # Seguimiento
            for batch in train_loader:
                # Calcular el costo
                loss = self.training_step(batch)
                #Seguimiento
                train_losses.append(loss)
                #Calcular las derivadas parciales
                loss.backward()

                # Gradient clipping, para que no ocurra el exploding gradient
                if grad_clip:
                    nn.utils.clip_grad_value_(self.parameters(), grad_clip)

                #Efectuar el descensod e gradiente y borrar el historial
                optimizer.step()
                optimizer.zero_grad()

                # Guardar el learning rate utilizado en el cycle.
                lrs.append(get_lr(optimizer))
                #Utilizar el siguiente valor de learning rate dado OneCycle scheduler
                sched.step()

            # Fase de validación
            result = self.evaluate(val_loader)
            result['train_loss'] = torch.stack(train_losses).mean().item() #Stackea todos los costos de las iteraciones sobre los batches y los guarda como la pérdida general de la época
            result['lrs'] = lrs #Guarda la lista de learning rates de cada batch
            self.epoch_end(epoch, result) #imprimir en pantalla el seguimiento
            history.append(result) # añadir a la lista el diccionario de resultados
        return history
#LSTMs with multihead attention incorporated
class EncoderMultiheadAttentionLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_heads, architecture):
        super(EncoderMultiheadAttentionLSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size= hidden_size
        #attention
        self.attention = nn.MultiheadAttention(input_size, num_heads, batch_first=True).to('cuda')
        self.layer_norm = nn.LayerNorm(input_size).to('cuda')
        
        #encoder
        self.lstm = nn.LSTMCell(input_size,hidden_size).to('cuda')
        self.fc = DeepNeuralNetwork(hidden_size,input_size, *architecture).to('cuda')

    def forward(self, x):
        batch_size, seq_length, _ = x.size()
        hn = torch.zeros(batch_size, self.hidden_size, requires_grad = True).to('cuda')
        cn = torch.zeros(batch_size, self.hidden_size, requires_grad = True).to('cuda')
        #Multihead attention
        attn_out, _ = self.attention(x,x,x)
        #residual connection and layer_norm
        attn_out = self.layer_norm(attn_out+x)
        #encoder
        out_list = []
        for t in range(seq_length):
            xt = attn_out[:,t,:]
            hn,cn = self.lstm(xt, (hn,cn))
            out = self.fc(hn)
            out_list.append(out)
        
        out = torch.stack(out_list, dim = 1)
        out = self.layer_norm(out+attn_out)

        return out, (hn, cn)
    
class MultiHeaded2MultiheadAttentionLSTM(MultiHead2MultiHeadBase):
    def __init__(self, encoder_fc, encoder_mg,num_heads: list, architecture, output_size):
        super(MultiHeaded2MultiheadAttentionLSTM, self).__init__()
        #hidden
        self.hidden_size = encoder_fc.hidden_size + encoder_mg.hidden_size
        #encoder(LSTMWithMultiHeadAttention)
        self.encoder_fc = encoder_fc
        self.encoder_mg = encoder_mg
        #MultiheadAttention
        self.attention_1 = nn.MultiheadAttention(encoder_fc.input_size, num_heads[0], batch_first=True).to('cuda')
        self.attention_2 = nn.MultiheadAttention(encoder_mg.input_size, num_heads[1], batch_first=True).to('cuda')
        #Decoder arch
        self.lstm_1 = nn.LSTMCell(encoder_fc.input_size, encoder_fc.hidden_size).to('cuda')
        self.lstm_2 = nn.LSTMCell(encoder_fc.input_size, encoder_fc.hidden_size).to('cuda')
        self.linear_1 = DeepNeuralNetwork(encoder_fc.hidden_size, encoder_fc.input_size, *architecture).to('cuda')
        self.lstm_3 = nn.LSTMCell(encoder_mg.input_size, encoder_mg.hidden_size).to('cuda')
        self.lstm_4 = nn.LSTMCell(encoder_mg.input_size, encoder_mg.hidden_size).to('cuda')
        self.linear_2 = DeepNeuralNetwork(encoder_mg.hidden_size, encoder_mg.input_size, *architecture).to('cuda')
        self.fc = DeepNeuralNetwork(self.hidden_size, output_size, *architecture).to('cuda')
        #layer norm with residual connections(AttentionIsAllYouNeed uses several times on arch)
        self.layer_norm_1 = nn.LayerNorm(encoder_fc.input_size).to('cuda')
        self.layer_norm_2 = nn.LayerNorm(encoder_mg.input_size).to('cuda')
    def forward(self, fc, mg):
        hn_list = []
        #get dim
        _, seq_length, _ = fc.size()
        #encoder for faraday cup
        out, (hn,cn) = self.encoder_fc(fc)
        #Attention mechanism
        attn_out, _ = self.attention_1(out,out,out)
        #Layer norm and residual
        attn_out = self.layer_norm_1(attn_out+fc)
        #Decoder
        out_list = []
        for t in range(seq_length):
            xt = attn_out[:,t,:]
            hn,cn = self.lstm_1(xt, (hn,cn))
            out = self.linear_1(hn)
            out_list.append(out)
        #Getting the output sequences from lstm processing
        out = torch.stack(out_list, dim = 1) #seq dim
        #last layer norm with residual connection
        out = self.layer_norm_1(out+attn_out)
        #second step decoder
        for t in range(seq_length):
            xt = out[:,t,:]
            hn,cn = self.lstm_2(xt, (hn,cn))
        #add to last hn 
        hn_list.append(hn)
        #encoder for magnetometer
        out, (hn,cn) = self.encoder_mg(mg)
        #Attention mechanism
        attn_out, _ = self.attention_2(out,out,out)
        #Layer norm and residual
        attn_out = self.layer_norm_2(attn_out+mg)
        #Decoder
        out_list = []
        for t in range(seq_length):
            xt = attn_out[:,t,:]
            hn,cn = self.lstm_3(xt, (hn,cn))
            out = self.linear_2(hn)
            out_list.append(out)
        #Getting the output sequences from lstm processing
        out = torch.stack(out_list, dim = 1) #seq dim
        #last layer norm with residual connection
        out = self.layer_norm_2(out+attn_out)
        #second step decoder
        for t in range(seq_length):
            xt = out[:,t,:]
            hn,cn = self.lstm_4(xt, (hn,cn))
        hn_list.append(hn)
        
        hn = torch.cat(hn_list, dim = 1)
        #inference with Deep Neural Network

        out = self.fc(hn)
        return out
#GRUs with multihead attention incorporated
class EncoderMultiheadAttentionGRU(nn.Module):
    def __init__(self, input_size, hidden_size, num_heads, architecture):
        super(EncoderMultiheadAttentionGRU, self).__init__()
        self.hidden_size=hidden_size
        self.input_size = input_size
        #attention
        self.attention = nn.MultiheadAttention(input_size, num_heads, batch_first = True).to('cuda')
        self.layer_norm = nn.LayerNorm(input_size).to('cuda')
        
        #encoder
        self.gru = nn.GRUCell(input_size,hidden_size).to('cuda')
        self.fc = DeepNeuralNetwork(hidden_size,input_size, *architecture).to('cuda')

    def forward(self, x):
        batch_size, seq_length, _ = x.size()
        hn = torch.zeros(batch_size, self.hidden_size, requires_grad = True).to('cuda')
        #Multihead attention
        attn_out, _ = self.attention(x,x,x)
        #residual connection and layer_norm
        attn_out = self.layer_norm(attn_out+x)
        
        #encoder
        out_list = []
        for t in range(seq_length):
            xt = attn_out[:,t,:]
            hn = self.gru(xt, hn)
            out = self.fc(hn)
            out_list.append(out)
        
        out = torch.stack(out_list, dim = 1)
        
        out = self.layer_norm(out+attn_out)

        return out, hn
class MultiHeaded2MultiheadAttentionGRU(MultiHead2MultiHeadBase):
    def __init__(self, encoder_fc, encoder_mg,num_heads: list, architecture, output_size):
        super(MultiHeaded2MultiheadAttentionGRU, self).__init__()
        #hidden
        self.hidden_size = encoder_fc.hidden_size + encoder_mg.hidden_size
        #encoder(LSTMWithMultiHeadAttention)
        self.encoder_fc = encoder_fc
        self.encoder_mg = encoder_mg
        #MultiheadAttention
        self.attention_1 = nn.MultiheadAttention(encoder_fc.input_size, num_heads[0], batch_first=True).to('cuda')
        self.attention_2 = nn.MultiheadAttention(encoder_mg.input_size, num_heads[1], batch_first=True).to('cuda')
        #Decoder arch
        self.gru_1 = nn.GRUCell(encoder_fc.input_size, encoder_fc.hidden_size).to('cuda')
        self.gru_2 = nn.GRUCell(encoder_fc.input_size, encoder_fc.hidden_size).to('cuda')
        self.linear_1 = DeepNeuralNetwork(encoder_fc.hidden_size, encoder_fc.input_size, *architecture).to('cuda')
        self.gru_3 = nn.GRUCell(encoder_mg.input_size, encoder_mg.hidden_size).to('cuda')
        self.gru_4 = nn.GRUCell(encoder_mg.input_size, encoder_mg.hidden_size).to('cuda')
        self.linear_2 = DeepNeuralNetwork(encoder_mg.hidden_size, encoder_mg.input_size, *architecture).to('cuda')
        self.fc = DeepNeuralNetwork(self.hidden_size, output_size, *architecture).to('cuda')
        #layer norm with residual connections(AttentionIsAllYouNeed uses several times on arch)
        self.layer_norm_1 = nn.LayerNorm(encoder_fc.input_size).to('cuda')
        self.layer_norm_2 = nn.LayerNorm(encoder_mg.input_size).to('cuda')
    def forward(self, fc, mg):
        hn_list = []
        #get dim
        _, seq_length, _ = fc.size()
        #encoder for faraday cup
        out, hn = self.encoder_fc(fc)
        #Attention mechanism
        attn_out, _ = self.attention_1(out,out,out)
        #Layer norm and residual
        attn_out = self.layer_norm_1(attn_out+fc)
        #Decoder
        out_list = []
        for t in range(seq_length):
            xt = attn_out[:,t,:]
            hn = self.gru_1(xt, hn)
            out = self.linear_1(hn)
            out_list.append(out)
        #Getting the output sequences from lstm processing
        out = torch.stack(out_list, dim = 1) #seq dim
        #last layer norm with residual connection
        out = self.layer_norm_1(out+attn_out)
        #second step decoder
        for t in range(seq_length):
            xt = out[:,t,:]
            hn = self.gru_2(xt, hn)
        #add to last hn 
        hn_list.append(hn)
        #encoder for magnetometer
        out, hn = self.encoder_mg(mg)
        #Attention mechanism
        attn_out, _ = self.attention_2(out,out,out)
        #Layer norm and residual
        attn_out = self.layer_norm_2(attn_out+mg)
        #Decoder
        out_list = []
        for t in range(seq_length):
            xt = attn_out[:,t,:]
            hn = self.gru_3(xt, hn)
            out = self.linear_2(hn)
            out_list.append(out)
        #Getting the output sequences from lstm processing
        out = torch.stack(out_list, dim = 1) #seq dim
        #last layer norm with residual connection
        out = self.layer_norm_2(out+attn_out)
        #second step decoder
        for t in range(seq_length):
            xt = out[:,t,:]
            hn = self.gru_4(xt, hn)
        hn_list.append(hn)
        
        hn = torch.cat(hn_list, dim = 1)
        #inference with Deep Neural Network

        out = self.fc(hn)
        return out