import torch
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import torchvision
import torchvision.transforms as T
from typing import Tuple


_STATS = {
    'cifar10':  dict(mean=(0.4914, 0.4822, 0.4465),
                     std =(0.2023, 0.1994, 0.2010),
                     cls =torchvision.datasets.CIFAR10),
    'cifar100': dict(mean=(0.5071, 0.4867, 0.4408),
                     std =(0.2675, 0.2565, 0.2761),
                     cls =torchvision.datasets.CIFAR100),
}


def get_dataloaders(
    config,
    rank:       int,
    world_size: int,
) -> Tuple[DataLoader, DataLoader]:

    if config.dataset not in _STATS:
        raise ValueError(f"Unknown dataset '{config.dataset}'. "
                         f"Choose from {list(_STATS.keys())}")

    info        = _STATS[config.dataset]
    mean, std   = info['mean'], info['std']
    DatasetCls  = info['cls']

    train_tf = T.Compose([
        T.RandomCrop(32, padding=4),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize(mean, std),
    ])
    test_tf = T.Compose([
        T.ToTensor(),
        T.Normalize(mean, std),
    ])

    train_ds = DatasetCls(config.data_root, train=True,
                          download=True, transform=train_tf)
    test_ds  = DatasetCls(config.data_root, train=False,
                          download=True, transform=test_tf)

    if world_size > 1:
        train_sampler = DistributedSampler(train_ds,
                                            num_replicas=world_size,
                                            rank=rank,
                                            shuffle=True)
        test_sampler  = DistributedSampler(test_ds,
                                            num_replicas=world_size,
                                            rank=rank,
                                            shuffle=False)
    else:
        train_sampler = None    # DataLoader uses default random sampler
        test_sampler  = None

    loader_kwargs = dict(
        num_workers=config.num_workers,
        pin_memory=True,
        persistent_workers=(config.num_workers > 0),
    )

    train_loader = DataLoader(train_ds,
                               batch_size=config.batch_size,
                               sampler=train_sampler,
                               shuffle=(train_sampler is None),
                               **loader_kwargs)
    test_loader  = DataLoader(test_ds,
                               batch_size=config.batch_size,
                               sampler=test_sampler,
                               shuffle=False,
                               **loader_kwargs)

    return train_loader, test_loader
