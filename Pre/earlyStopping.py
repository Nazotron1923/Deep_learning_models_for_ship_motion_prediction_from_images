import numpy as np
import torch

class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=10, verbose=False):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 10
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf

    def __call__(self, val_loss, model, best_model_weight_path, cuda):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, best_model_weight_path, cuda)
        elif score < self.best_score:
            self.counter += 1
            if self.verbose:
                print(f'\nEarlyStopping counter: {self.counter} out of {self.patience}\n')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, best_model_weight_path, cuda)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, best_model_weight_path, cuda):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            print(f'\nValidation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...\n')
        if cuda:
            model.cpu()
        torch.save(model.state_dict(), best_model_weight_path)
        if cuda:
            model.cuda()
        self.val_loss_min = val_loss
