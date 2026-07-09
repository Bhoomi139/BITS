import numpy as np
import torch
import torch.nn.functional as F
from torch.autograd import Variable
import time
import sys
import os

class Trainer():
    def __init__(self, model, model_type, loss_fn, optimizer, lr_schedule, log_batchs, is_use_cuda, train_data_loader, \
                valid_data_loader=None, metric=None, start_epoch=0, num_epochs=25, is_debug=False, logger=None, writer=None):
        self.model = model
        self.model_type = model_type
        self.loss_fn  = loss_fn
        self.optimizer = optimizer
        self.lr_schedule = lr_schedule
        self.log_batchs = log_batchs
        self.is_use_cuda = is_use_cuda
        self.train_data_loader = train_data_loader
        self.valid_data_loader = valid_data_loader
        self.start_epoch = start_epoch
        self.num_epochs = num_epochs
        self.is_debug = is_debug
        
        self.cur_epoch = start_epoch
        self.best_mae = sys.float_info.max
        self.logger = logger
        self.writer = writer

    def fit(self):
        for epoch in range(self.start_epoch, self.num_epochs):
            self.logger.append('Epoch {}/{}'.format(epoch, self.num_epochs - 1))
            self.logger.append('-' * 60)
            self.cur_epoch = epoch
            self.lr_schedule.step()
            if self.is_debug:
                self._dump_infos()
            train_loss, train_mae=self._train()
            val_loss, val_mae=self._valid()
            self.logger.append(
                f"Epoch {epoch+1} Summary | "
                f"Train Loss: {train_loss:.4f} | "
                f"Train MAE: {train_mae:.4f} | "
                f"Val Loss: {val_loss:.4f} | "
                f"Val MAE: {val_mae:.4f}"
)
            self._save_best_model(val_mae)
            print()

    def _dump_infos(self):
        self.logger.append('---------------------Current Parameters---------------------')
        self.logger.append('is use GPU: ' + ('True' if self.is_use_cuda else 'False'))
        self.logger.append('lr: %f' % (self.lr_schedule.get_lr()[0]))
        self.logger.append('model_type: %s' % (self.model_type))
        self.logger.append('current epoch: %d' % (self.cur_epoch))
        self.logger.append('best MAE: %f' % (self.best_mae))
        self.logger.append('------------------------------------------------------------')

    def _train(self):
        self.model.train()  # Set model to training mode
        losses = []
        maes=[]

        for i, (inputs, labels) in enumerate(self.train_data_loader):              # Notice
            if self.is_use_cuda:
                inputs, labels = inputs.cuda(), labels.cuda()
                labels = labels.squeeze()
            else:
                labels = labels.squeeze()

            self.optimizer.zero_grad()
            outputs = self.model(inputs)   
            outputs = outputs.squeeze(1)         # Notice 
            loss = self.loss_fn[0](outputs, labels)
            mae = torch.mean(torch.abs(outputs - labels))
            loss.backward()
            self.optimizer.step()

            losses.append(loss.item())
            maes.append(mae.item())       # Notice
            if 0 == i % self.log_batchs or (i == len(self.train_data_loader) - 1):
                local_time_str = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))
                batch_mean_loss  = np.mean(losses)
                batch_mean_maes=np.mean(maes)
                print_str = '[%s]\tTraining Batch[%d/%d]\t Loss: %.4f\t MAE: %.4f\t'           \
                            % (local_time_str, i, len(self.train_data_loader) - 1, batch_mean_loss, np.mean(maes))
                self.logger.append(print_str)  
                
        if self.writer:
            self.writer.add_scalar(
                'train/loss',
                 batch_mean_loss,
                self.cur_epoch
            )
            self.writer.add_scalar(
                'train/mae',
                 batch_mean_maes,
                self.cur_epoch
            )
        return batch_mean_loss, batch_mean_maes  


    def _valid(self):
        self.model.eval()
        losses = []
        maes=[]
        acc_rate = 0.

        with torch.no_grad():              # Notice
            for i, (inputs, labels) in enumerate(self.valid_data_loader):
                if self.is_use_cuda:
                    inputs, labels = inputs.cuda(), labels.cuda()
                    labels = labels.squeeze()
                else:
                    labels = labels.squeeze()

                outputs = self.model(inputs)
                outputs=outputs.squeeze(1)           # Notice 
                loss = self.loss_fn[0](outputs, labels)
                mae = torch.mean(torch.abs(outputs-labels))
                losses.append(loss.item())
                maes.append(mae.item())
            
        local_time_str = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))
        #self.logger.append(losses)
        batch_mean_loss = np.mean(losses)
        batch_mean_maes =np.mean(maes)
    
        print_str = '[%s]\tValidation:\t Loss: %.4f\t MAE: %.4f\t' \
            % (local_time_str, batch_mean_loss, batch_mean_maes)

        self.logger.append(print_str)
        if self.writer:
            self.writer.add_scalar(
                'val/loss',
                 batch_mean_loss,
                self.cur_epoch
            )
            self.writer.add_scalar(
                'val/mae',
                 batch_mean_maes,
                self.cur_epoch
            )

        return batch_mean_loss, batch_mean_maes 


    def _save_best_model(self, val_mae):
        if val_mae < self.best_mae:
            self.best_mae = val_mae

            self.logger.append(
                'Saving Best Model...'
            )

            state = {
                'state_dict': self.model.state_dict(),
                'best_mae': self.best_mae,
                'cur_epoch': self.cur_epoch,
                'num_epochs': self.num_epochs
            }

            save_dir = './checkpoint/' + self.model_type

            if not os.path.isdir(save_dir):
                os.makedirs(save_dir)

            torch.save(
                state,
                save_dir + '/best_model.ckpt'
            )
        else:
            self.logger.append(
                'Validation MAE did not improve. Model not saved.'
            )
