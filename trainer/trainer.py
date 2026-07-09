import os
import csv
import time
import sys
import torch


class Trainer:

    def __init__(
        self,
        model,
        model_type,
        loss_fn,
        optimizer,
        lr_schedule,
        log_batchs,
        is_use_cuda,
        train_data_loader,
        valid_data_loader=None,
        metric=None,
        start_epoch=0,
        num_epochs=90,
        is_debug=False,
        logger=None,
        writer=None
    ):

        self.model = model
        self.model_type = model_type
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.lr_schedule = lr_schedule

        self.log_batchs = log_batchs

        self.is_use_cuda = is_use_cuda

        self.train_data_loader = train_data_loader
        self.valid_data_loader = valid_data_loader

        self.metric = metric

        self.start_epoch = start_epoch
        self.num_epochs = num_epochs

        self.is_debug = is_debug

        self.logger = logger
        self.writer = writer

        self.cur_epoch = start_epoch

        self.best_mae = float("inf")

        self.patience = 10
        self.counter = 0

        os.makedirs("logs", exist_ok=True)

        self.history_path = os.path.join("logs", "history.csv")

        with open(self.history_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "Epoch",
                "Train Loss",
                "Train MAE",
                "Val Loss",
                "Val MAE",
                "Learning Rate"
            ])

    def fit(self):

        for epoch in range(self.start_epoch, self.num_epochs):

            self.cur_epoch = epoch

            print("\n" + "=" * 70)
            print(f"Epoch {epoch + 1}/{self.num_epochs}")
            print("=" * 70)

            train_loss, train_mae = self._train()

            val_loss, val_mae = self._valid()

            lr = self.optimizer.param_groups[0]["lr"]

            if self.writer:

                self.writer.add_scalar("Loss/Train", train_loss, epoch)
                self.writer.add_scalar("Loss/Validation", val_loss, epoch)

                self.writer.add_scalar("MAE/Train", train_mae, epoch)
                self.writer.add_scalar("MAE/Validation", val_mae, epoch)

                self.writer.add_scalar("Learning Rate", lr, epoch)

            with open(self.history_path, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([
                    epoch + 1,
                    train_loss,
                    train_mae,
                    val_loss,
                    val_mae,
                    lr
                ])

            print(
                f"Epoch {epoch+1:03d} | "
                f"Train Loss: {train_loss:.4f} | "
                f"Train MAE: {train_mae:.4f} | "
                f"Val Loss: {val_loss:.4f} | "
                f"Val MAE: {val_mae:.4f}"
            )

            if self.logger:

                self.logger.append(
                    f"Epoch {epoch+1} | "
                    f"Train Loss={train_loss:.4f} | "
                    f"Train MAE={train_mae:.4f} | "
                    f"Val Loss={val_loss:.4f} | "
                    f"Val MAE={val_mae:.4f}"
                )

            if val_mae < self.best_mae:

                self.best_mae = val_mae
                self.counter = 0

                self._save_best_model()

            else:

                self.counter += 1

                print(
                    f"No improvement "
                    f"({self.counter}/{self.patience})"
                )

            self.lr_schedule.step()

            if self.counter >= self.patience:

                print("\nEarly stopping triggered.")

                if self.logger:
                    self.logger.append("Early stopping triggered.")

                break

    def _train(self):

        self.model.train()

        running_loss = 0.0
        running_mae = 0.0
        total = 0

        for inputs, labels in self.train_data_loader:

            if self.is_use_cuda:

                inputs = inputs.cuda(non_blocking=True)
                labels = labels.cuda(non_blocking=True)

            labels = labels.squeeze()

            self.optimizer.zero_grad()

            outputs = self.model(inputs).squeeze(1)

            loss = self.loss_fn[0](outputs, labels)

            mae = torch.mean(torch.abs(outputs - labels))

            loss.backward()

            self.optimizer.step()

            batch = inputs.size(0)

            running_loss += loss.item() * batch
            running_mae += mae.item() * batch
            total += batch

        epoch_loss = running_loss / total
        epoch_mae = running_mae / total

        return epoch_loss, epoch_mae

    def _valid(self):

        self.model.eval()

        running_loss = 0.0
        running_mae = 0.0
        total = 0

        with torch.no_grad():

            for inputs, labels in self.valid_data_loader:

                if self.is_use_cuda:

                    inputs = inputs.cuda(non_blocking=True)
                    labels = labels.cuda(non_blocking=True)

                labels = labels.squeeze()

                outputs = self.model(inputs).squeeze(1)

                loss = self.loss_fn[0](outputs, labels)

                mae = torch.mean(torch.abs(outputs - labels))

                batch = inputs.size(0)

                running_loss += loss.item() * batch
                running_mae += mae.item() * batch
                total += batch

        epoch_loss = running_loss / total
        epoch_mae = running_mae / total

        return epoch_loss, epoch_mae

    def _save_best_model(self):

        save_dir = os.path.join("checkpoint", self.model_type)

        os.makedirs(save_dir, exist_ok=True)

        checkpoint = {

            "state_dict": self.model.state_dict(),
            "best_mae": self.best_mae,
            "epoch": self.cur_epoch + 1

        }

        torch.save(
            checkpoint,
            os.path.join(save_dir, "best_model.ckpt")
        )

        print(
            f"Best model saved "
            f"(Validation MAE = {self.best_mae:.4f})"
        )

        if self.logger:

            self.logger.append(
                f"Saved best model with MAE={self.best_mae:.4f}"
            )
