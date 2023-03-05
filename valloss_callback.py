import copy

import numpy as np


class EarlyStopping:
    # Early stops the training if validation loss doesn't improve after a given patience.

    def __init__(self, patience=7, verbose=False, delta=0, trace_func=print):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            trace_func (function): trace print function.
                            Default: print
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.trace_func = trace_func
        self.best_model_wts = None

    def __call__(self, val_loss, model):

        # Set the score to negative validation loss
        current_loss = -val_loss

        # If there is no best score yet, set the best score to be the current one and save a checkpoint
        if self.best_loss is None:
            self.best_loss = current_loss
            self.save_checkpoint(val_loss, model)
        # If the current score if worse that the best score, increase the counter
        elif current_loss > self.best_loss + self.delta:
            self.counter += 1
            self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        # If the current score is better than the best score, reset the counter and make a checkpoint
        else:
            self.val_loss_min = -self.best_loss
            self.best_loss = current_loss
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        # Saves model when validation loss decrease
        if self.verbose:
            self.trace_func(
                f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')

        # Create a copy of the model weights
        self.best_model_wts = copy.deepcopy(model.state_dict())
