import torch.nn as nn
import torch.nn.functional as F
from models.macro_architectures import get_lr
import torch
#not multihead to multiheadattention
class SingleHead2MultiHead(nn.Module):
    def training_step(self, batch):
        feature, target = batch #decompose batch
        pred = self(feature)        
        #dst index or kp
        loss = F.mse_loss(pred, target)
        return loss

    def validation_step(self, batch):
        feature, target = batch #decompose batch
        pred = self(feature)        
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
class Seq2SeqLSTM(SingleHead2MultiHead):
    def __init__(self, input_size, hidden_size, output_size, num_heads): #input 20, #hidden 10, #pred_length with transformation
        super(Seq2SeqLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.lstm_1 = nn.LSTMCell(input_size, hidden_size).to('cuda')
        self.fc_1 = nn.Linear(hidden_size, input_size).to('cuda')

        ## Attention mechanism 
        self.attention_2 = nn.MultiheadAttention(input_size, num_heads, batch_first = True).to('cuda')
        # Decoder

        self.lstm_2 = nn.LSTMCell(input_size, hidden_size).to('cuda') #encoders[0].hidden_size*len(encoders) hidden_sizeto hidden_size
        self.fc_2 = nn.Linear(hidden_size, output_size).to('cuda')

    def forward(self, x):
        batch_size, seq_length, _ = x.size()
        hn = torch.zeros(batch_size, self.hidden_size, requires_grad = True).to('cuda')
        cn = torch.zeros(batch_size, self.hidden_size, requires_grad = True).to('cuda')
        out_list = []
        for i in range(seq_length):
            xt = x[:,i,:]
            hn,cn = self.lstm_1(xt, (hn,cn))
            out = self.fc_1(hn)
            out_list.append(out)
        
        query = torch.stack(out_list, dim=1)

        # Multihead Attention
        attention_output, _ = self.attention_2(query, query, query)

        # Decoding
        for i in range(seq_length):
            xt = attention_output[:,i,:]
            hn,cn = self.lstm_2(xt, (hn,cn))
        
        out = self.fc_2(hn)

        return out

##Seq2Seq models with attention
class GRUSeq2Seq(SingleHead2MultiHead):
    def __init__(self, input_size, hidden_size, output_size, num_heads):
        super(GRUSeq2Seq, self).__init__()
        self.hidden_size = hidden_size
        self.gru_1 = nn.GRUCell(input_size, hidden_size).to('cuda')
        self.fc_1 = nn.Linear(hidden_size, input_size).to('cuda')

        ## Attention mechanism 
        self.attention = nn.MultiheadAttention(input_size, num_heads, batch_first = True).to('cuda')
        # Decoder

        self.gru_2 = nn.GRUCell(input_size, hidden_size).to('cuda') #encoders[0].hidden_size*len(encoders) hidden_sizeto hidden_size
        self.fc_2 = nn.Linear(hidden_size, output_size).to('cuda')

    def forward(self, x):
        batch_size, seq_length, _ = x.size()
        hn = torch.zeros(batch_size, self.hidden_size, requires_grad = True).to('cuda')
        out_list = []
        for i in range(seq_length):
            xt = x[:,i,:]
            hn = self.gru_1(xt, hn)
            out = self.fc_1(hn)
            out_list.append(out)
        
        query = torch.stack(out_list, dim=1)

        # Multihead Attention
        attention_output, _ = self.attention(query, query, query)

        # Decoding
        for i in range(seq_length):
            xt = attention_output[:,i,:]
            hn = self.gru_2(xt, hn)
        
        out = self.fc_2(hn)

        return out