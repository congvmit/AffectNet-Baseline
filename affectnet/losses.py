import torch.nn as nn


def get_loss_fn(loss_name, args=None):
    if loss_name == 'bce':
        return nn.BCEWithLogitsLoss()
    if loss_name == 'cce':
        return nn.CrossEntropyLoss()
    elif loss_name == 'mse':
        return nn.MSELoss()
    elif loss_name == 'smooth_mae':
        return nn.SmoothL1Loss()
    elif loss_name == 'mae':
        return nn.L1Loss()
    else:
        raise ValueError('Invalid `disc_loss_type`')
