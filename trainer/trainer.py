import numpy as np
import torch
import time
import sys
import os


class Trainer():
    def __init__(self, model, model_type, loss_fn, optimizer, lr_schedule, log_batchs, is_use_cuda, train_data_loader,
                 valid_data_loader=None, metric=None, start_epoch=0, num_epochs=25, is_debug=False, logger=None,
                 writer=None):
        self.model = model
        self.model_type = model_type
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.lr_schedule = lr_schedule
        self.log_batchs = log_batchs
        self.is_use_cuda = is_use_cuda
        self.train_data_loader = train_data_loader
        self.valid_data_loader = valid_data_loader
        self.start_epoch = start_epoch
        self.num_epochs = num_epochs
        self.is_debug = is_debug
        self.patience = 10      # Stop after 10 epochs without improvement
        self.counter = 0
        self.cur_epoch = start_epoch
        self.best_mae = sys.float_info.max
        self.logger = logger
        self.writer = writer

    def fit(self):
        for epoch in range(self.start_epoch, self.num_epochs):
            self.cur_epoch = epoch

            print(f"\nEpoch [{epoch + 1}/{self.num_epochs}]")
            print("-" * 60)

            if self.is_debug:
                self._dump_infos()

            train_loss, train_mae = self._train()
            val_loss, val_mae = self._valid()
            
            if val_mae < self.best_mae:
             self.counter = 0
             self._save_best_model(val_mae)
           else:
            self.counter += 1
             self.logger.append(
               f"No improvement for {self.counter}/{self.patience} epochs.")

            if self.counter >= self.patience:
              self.logger.append("Early stopping triggered!")
              break

            self.logger.append(
                f"Epoch {epoch + 1} Summary | "
                f"Train Loss: {train_loss:.4f} | "
                f"Train MAE: {train_mae:.4f} | "
                f"Val Loss: {val_loss:.4f} | "
                f"Val MAE: {val_mae:.4f}"
            )
            self._save_best_model(val_mae)

            self.lr_schedule.step()

    def _dump_infos(self):
        self.logger.append('---------------------Current Parameters---------------------')
        self.logger.append('is use GPU: ' + ('True' if self.is_use_cuda else 'False'))
        self.logger.append('lr: %f' % (self.lr_schedule.get_last_lr()[0]))
        self.logger.append('model_type: %s' % (self.model_type))
        self.logger.append('current epoch: %d' % (self.cur_epoch))
        self.logger.append('best MAE: %f' % (self.best_mae))
        self.logger.append('------------------------------------------------------------')

    def _train(self):
        self.model.train()

        running_loss = 0.0
        running_mae = 0.0
        total_samples = 0

        for inputs, labels in self.train_data_loader:

            if self.is_use_cuda:
                inputs = inputs.cuda()
                labels = labels.cuda()

            labels = labels.squeeze()

            self.optimizer.zero_grad()

            outputs = self.model(inputs)
            outputs = outputs.squeeze(1)

            loss = self.loss_fn[0](outputs, labels)
            mae = torch.mean(torch.abs(outputs - labels))

            loss.backward()
            self.optimizer.step()

            batch_size = inputs.size(0)

            running_loss += loss.item() * batch_size
            running_mae += mae.item() * batch_size
            total_samples += batch_size

        epoch_loss = running_loss / total_samples
        epoch_mae = running_mae / total_samples

        if self.writer:
            self.writer.add_scalar(
                'train/loss',
                epoch_loss,
                self.cur_epoch
            )
            self.writer.add_scalar(
                'train/mae',
                epoch_mae,
                self.cur_epoch
            )

        return epoch_loss, epoch_mae

    def _valid(self):
        self.model.eval()

        running_loss = 0.0
        running_mae = 0.0
        total_samples = 0

        with torch.no_grad():

            for inputs, labels in self.valid_data_loader:

                if self.is_use_cuda:
                    inputs = inputs.cuda()
                    labels = labels.cuda()

                labels = labels.squeeze()

                outputs = self.model(inputs)
                outputs = outputs.squeeze(1)

                loss = self.loss_fn[0](outputs, labels)
                mae = torch.mean(torch.abs(outputs - labels))

                batch_size = inputs.size(0)

                running_loss += loss.item() * batch_size
                running_mae += mae.item() * batch_size
                total_samples += batch_size

        epoch_loss = running_loss / total_samples
        epoch_mae = running_mae / total_samples

        if self.writer:
            self.writer.add_scalar(
                'val/loss',
                epoch_loss,
                self.cur_epoch
            )
            self.writer.add_scalar(
                'val/mae',
                epoch_mae,
                self.cur_epoch
            )

        return epoch_loss, epoch_mae

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
