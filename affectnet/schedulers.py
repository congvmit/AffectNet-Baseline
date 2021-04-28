import torch.optim as optim


def get_scheduler(scheduler_name, optimizer, args=None):
    if scheduler_name == 'multistep':
        milestones = list(
            map(lambda x: int(x), args.lr_multistep_milestones.split(',')))
        lr_scheduler = {
            'scheduler': optim.lr_scheduler.MultiStepLR(optimizer,
                                                        milestones=milestones,
                                                        gamma=args.lr_gamma,
                                                        verbose=True),
            'interval': 'epoch',
            'name': f'LearningRate_MultiStepLR'
        }
    elif scheduler_name is None:
        lr_scheduler = None
    else:
        raise ValueError('Invalid learning rate scheduler in arguments')
    return lr_scheduler
