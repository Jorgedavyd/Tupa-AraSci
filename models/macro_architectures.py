import torch.nn.functional as F
import torch.nn as nn
import torch

class DefaultBase(nn.Module):
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

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr'] # Seguimiento del learning rate

class NormalArchitecture(DefaultBase):
    def __init__(self, encoder, dst, kp):
        super(NormalArchitecture, self).__init__()
        self.encoder = encoder
        self.fc_dst = dst #multiheaded neural network ##regression
        self.fc_kp = kp   #multiclass
    def forward(self, x):
        out = self.encoder(x)
        dst_out = self.fc_dst(out)
        kp_out = self.fc_kp(out)
        return dst_out, kp_out
    def training_step(self, batch):
        loss = 0 #initialize loss
        feature, dst, kp = batch #decompose batch
        dst_out, kp_out = self(feature)        
        #dst index cost-1st head
        loss += F.mse_loss(dst_out, dst)
        #kp index cost - 2nd head
        loss += F.mse_loss(kp_out, kp)
        return loss
    def validation_step(self, batch):
        loss = 0 #initialize loss
        feature, dst, kp = batch #decompose batch
        dst_out, kp_out = self(feature)        
        #dst index cost-1st head
        loss += F.mse_loss(dst_out, dst)
        #kp index cost - 2nd head
        loss += F.mse_loss(kp_out, kp)
        return {'val_loss': loss.detach()}
class RefinedArchitecture(nn.Module):
    def __init__(self, encoder, dst, kp):
        super(RefinedArchitecture, self).__init__()
        self.encoder = encoder
        self.fc_dst = dst #multiheaded neural network  ##regression
        self.fc_kp = kp   #multiclass
    def forward(self, x):
        out = self.encoder(x)
        dst_out = self.fc_dst(out)
        kp_out = self.fc_kp(out)
        return dst_out, kp_out
    def training_step(self, batch, weigths = [0.01,0.01,1]):
        h_weigth, o_weigth, main = weigths
        #initialize loss
        loss = 0
        #from batch
        l1_sample, l2_sample, dst, kp = batch
        #encoder loss
        h_t = self.encoder(l1_sample)
        h_t_hat = self.encoder(l2_sample)
        encoder_loss = F.mse_loss(h_t, h_t_hat)
        ##add to overall
        loss+=h_weigth*encoder_loss
        #output loss
        ##initialize output loss
        output_loss = 0
        #inference l1
        dst_out = self.fc_dst(h_t)
        kp_out = self.fc_kp(h_t)
        #inference l2
        dst_out_hat = self.fc_dst(h_t_hat)
        kp_out_hat = self.fc_kp(h_t_hat)
        ##dst index cost-1st head with both outputs
        output_loss += F.mse_loss(dst_out, dst_out_hat)
        ##kp index cost - 2nd head with both outputs
        output_loss += F.mse_loss(kp_out, kp_out_hat)
        ##add to overall
        loss+=o_weigth*output_loss
        #main loss 
        main_loss = 0
        ##dst index cost-1st head
        main_loss += F.mse_loss(dst_out, dst)
        ##kp index cost - 2nd head
        main_loss += F.mse_loss(kp_out, kp)
        ##add to overall
        loss+=main*main_loss
        return loss, encoder_loss, output_loss, main_loss

    def validation_step(self, batch, weigths = [0.01,0.01,1]):
        h_weigth, o_weigth, main = weigths
        #initialize loss
        loss = 0
        #from batch
        l1_sample, l2_sample, dst, kp = batch
        #encoder loss
        h_t = self.encoder(l1_sample)
        h_t_hat = self.encoder(l2_sample)
        encoder_loss = F.mse_loss(h_t, h_t_hat)
        ##add to overall
        loss+=h_weigth*encoder_loss
        #output loss
        ##initialize output loss
        output_loss = 0
        #inference l1
        dst_out = self.fc_dst(h_t)
        kp_out = self.fc_kp(h_t)
        #inference l2
        dst_out_hat = self.fc_dst(h_t_hat)
        kp_out_hat = self.fc_kp(h_t_hat)
        ##dst index cost-1st head with both outputs
        output_loss += F.mse_loss(dst_out, dst_out_hat)
        ##kp index cost - 2nd head with both outputs
        output_loss += F.mse_loss(kp_out, kp_out_hat)
        ##add to overall
        loss+=o_weigth*output_loss
        #main loss 
        main_loss = 0
        ##dst index cost-1st head
        main_loss += F.mse_loss(dst_out, dst)
        ##kp index cost - 2nd head
        main_loss += F.mse_loss(kp_out, kp)
        ##add to overall
        loss+=main*main_loss
        return {'overall_loss': loss.detach(), 'main_loss': main_loss.detach(), 'output_loss': output_loss.detach(), 'encoder_loss': encoder_loss.detach()}

    def validation_epoch_end(self, outputs):
        batch_overall = [x['overall_loss'] for x in outputs]
        epoch_overall = torch.stack(batch_overall).mean()   
        batch_main = [x['main_loss'] for x in outputs]
        epoch_main = torch.stack(batch_main).mean()   
        batch_output = [x['output_loss'] for x in outputs]
        epoch_output = torch.stack(batch_output).mean()  
        batch_encoder = [x['encoder_loss'] for x in outputs]
        epoch_encoder = torch.stack(batch_encoder).mean()   
        return {'val_overall_loss': epoch_overall.item(), 'val_main_loss': epoch_main.item(), 'val_output_loss': epoch_output.item(), 'val_encoder_loss': epoch_encoder.item()}

    def epoch_end(self, epoch, result): # Seguimiento del entrenamiento
        print("Epoch [{}]\n\tlast_lr: {:.5f}\n\ttrain_overall_loss: {:.4f}\n\ttrain_main_loss: {:.4f}\n\ttrain_output_loss: {:.4f}\n\ttrain_encoder_loss: {:.4f}\n\tval_overall_loss: {:.4f}\n\tval_main_loss: {:.4f}\n\tval_output_loss: {:.4f}\n\tval_encoder_loss: {:.4f}".format(
            epoch, result['lrs'][-1], result['train_overall_loss'], result['train_main_loss'], result['train_output_loss'], result['train_encoder_loss'], result['val_overall_loss'], result['val_main_loss'], result['val_output_loss'], result['val_encoder_loss']))
    
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
            encoder_losses = []
            output_losses = []
            main_losses = []
            lrs = [] # Seguimiento
            for batch in train_loader:
                # Calcular el costo
                loss, encoder_loss, output_loss, main_loss = self.training_step(batch)
                #Seguimiento
                train_losses.append(loss)
                encoder_losses.append(encoder_loss)
                output_losses.append(output_loss)
                main_losses.append(main_loss)
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
            result['train_overall_loss'] = torch.stack(train_losses).mean().item() 
            result['train_main_loss'] = torch.stack(main_losses).mean().item() 
            result['train_output_loss'] = torch.stack(output_losses).mean().item()
            result['train_encoder_loss'] = torch.stack(encoder_losses).mean().item()
            result['lrs'] = lrs 
            self.epoch_end(epoch, result)
            history.append(result)
        return history

class RNNtoSingleOUT(DefaultBase):
    def __init__(self, encoder, fc):
        super(RNNtoSingleOUT, self).__init__()
        self.encoder = encoder
        self.fc = fc 
    def forward(self, x):
        out = self.encoder(x)
        out = self.fc(out)
        return out
    def training_step(self, batch):
        feature, target = batch #decompose batch
        pred = self(feature)        
        #cost
        loss = F.mse_loss(pred, target)
        return loss

    def validation_step(self, batch):
        feature, target = batch #decompose batch
        pred = self(feature)        
        #cost
        loss = F.mse_loss(pred, target)
        return {'val_loss': loss.detach()}

class MultiHead2SingleOUT(DefaultBase):
    def __init__(self, encoder_fc, encoder_mg, fc):
        super(MultiHead2SingleOUT, self).__init__()
        self.encoder_fc = encoder_fc
        self.encoder_mg = encoder_mg
        self.fc = fc #sum of both hiddens at input_size
    def forward(self, fc, mg):
        out_1 = self.encoder_fc(fc)
        out_2 = self.encoder_mg(mg)
        hidden = torch.cat((out_1, out_2), dim = 1)
        out = self.fc(hidden)
        return out
    def training_step(self, batch):
        fc, mg, target = batch #decompose batch
        pred = self(fc, mg)        
        #cost
        loss = F.mse_loss(pred, target)
        return loss

    def validation_step(self, batch):
        fc, mg, target = batch #decompose batch
        pred = self(fc, mg)        
        #cost
        loss = F.mse_loss(pred, target)
        return {'val_loss': loss.detach()}
