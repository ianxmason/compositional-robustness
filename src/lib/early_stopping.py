"""
Extended from https://github.com/Bjarten/early-stopping-pytorch
"""
import numpy as np
import torch
import os


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=7, verbose=False, delta=0, path='checkpoint.pt', trace_func=print,
                 save_full_state=False):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
            trace_func (function): trace print function.
                            Default: print
            save_full_state (bool): If True, saves the full state of the model, optimizer, scheduler etc.
                            Default: False
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path
        self.trace_func = trace_func
        self.save_full_state = save_full_state

    def __call__(self, val_loss, model, optimizer=None, scheduler=None, epoch=None):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, optimizer, scheduler, epoch)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, optimizer, scheduler, epoch)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, optimizer=None, scheduler=None, epoch=None):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            self.trace_func(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        if self.save_full_state:
            if optimizer is None or scheduler is None or epoch is None:
                raise ValueError("If save_full_state is True, optimizer, scheduler and epoch must be provided")
            torch.save({
                'epoch': epoch,
                'loss': val_loss,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
            }, self.path)
        else:
            torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss

    def load_from_checkpoint(self, model, optimizer=None, scheduler=None, device=None):
        if self.save_full_state:
            if optimizer is None or scheduler is None or device is None:
                raise ValueError("If save_full_state is True, optimizer, scheduler and device must be provided")
            ckpt = torch.load(self.path, map_location=device)
            last_epoch = ckpt['epoch']
            last_loss = ckpt['loss']
            model.load_state_dict(ckpt['model_state_dict'])
            optimizer.load_state_dict(ckpt['optimizer_state_dict'])
            scheduler.load_state_dict(ckpt['scheduler_state_dict'])
            self.val_loss_min = last_loss
            self.best_score = -last_loss
            return last_epoch
        else:
            model.load_state_dict(torch.load(self.path))

    def delete_checkpoint(self):
        os.remove(self.path)
