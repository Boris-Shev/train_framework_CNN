from torch.utils.data import DataLoader
import torchvision.datasets
from torchvision import transforms as tfs
from torch.utils.data import random_split
from functools import partial



def get_dataloader( dataset: str = 'MNIST',
                    root: str = 'data',
                    batch_size: int = 32,
                    num_workers: int = 0,
                    val_size: float = 0.2,
                    img_height: int = 28,
                    img_width: int = 28
                    ) -> DataLoader:
    avaliable_datasets = ['MNIST', 'CIFAR10']
    if dataset in avaliable_datasets:
        torch_dataset = getattr(torchvision.datasets, dataset)
    else:
        raise Exception(f'No such dataset: {dataset}. Only {avaliable_datasets} avaliable')
    if  img_height < 28 or img_width < 28:
        raise Exception('Image size must be at least 28x28')

    crop = partial(torchvision.transforms.functional.crop, top=0, left=0, height=img_height, width=img_width)
    data_tfs = tfs.Compose([
        tfs.ToTensor(),
        tfs.Normalize((0.5), (0.5)),
        crop
    ])

    train_dataset = torch_dataset(root, train=True,  transform=data_tfs, download=True)
    train_dataset, val_dataset = random_split(train_dataset, [1 - val_size, val_size])
    test_dataset  = torch_dataset(root, train=False, transform=data_tfs, download=True)

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
    