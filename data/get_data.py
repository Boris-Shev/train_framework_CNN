import os

from torch.utils.data import DataLoader
import torchvision.datasets
from torchvision import transforms as tfs
from torch.utils.data import random_split
from functools import partial


def get_dataloader( dataset: str = 'MNIST',
                    root: str = os.path.join('data', 'datasets'),
                    batch_size: int = 32,
                    num_workers: int = 0,
                    val_size: float = 0.2,
                    test_size: float = 0e-10,
                    img_height: int = 28,
                    img_width: int = 28
                    ) -> DataLoader:
    avaliable_datasets = ['MNIST', 'CIFAR10', 'EuroSAT']
    if dataset in avaliable_datasets:
        torch_dataset = getattr(torchvision.datasets, dataset)
    else:
        raise Exception(f'No such dataset: {dataset}. Only {avaliable_datasets} avaliable')


    data_tfs = tfs.Compose([
        tfs.ToTensor(),
        tfs.Normalize((0.5), (0.5)),
        tfs.Resize((img_height, img_width))
    ])

    import ssl 
    ssl._create_default_https_context = ssl._create_unverified_context
    train_size = 1 - val_size - test_size
    train_dataset = torch_dataset(root, transform=data_tfs, download=True)
    train_dataset, val_dataset, test_dataset = random_split(train_dataset, [train_size, val_size, test_size])

    train_dataloader =  DataLoader(train_dataset, 
                                batch_size=batch_size, 
                                num_workers=num_workers)
    val_dataloader =  DataLoader(val_dataset, 
                                batch_size=batch_size, 
                                num_workers=num_workers)                               
    test_dataloader =  DataLoader(test_dataset, 
                                batch_size=batch_size, 
                                num_workers=num_workers)
    return train_dataloader, val_dataloader, test_dataloader