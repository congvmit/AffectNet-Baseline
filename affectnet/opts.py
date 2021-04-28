import torch.nn as nn
import torch.optim as optim


def get_optimizer(opt_name, model, args=None):
    if opt_name == 'sgd':
        opt = optim.SGD(model.parameters(),
                        lr=args.lr,
                        momentum=0.9,
                        nesterov=True,
                        weight_decay=args.weight_decay)

    elif opt_name == 'adam':
        opt = optim.Adam(model.parameters(),
                         lr=args.lr,
                         betas=(0.9, 0.999),
                         weight_decay=args.weight_decay)

    elif opt_name == 'adamw':
        opt = optim.AdamW(model.parameters(),
                          lr=args.lr,
                          betas=(0.9, 0.999),
                          weight_decay=args.weight_decay)
    return opt
