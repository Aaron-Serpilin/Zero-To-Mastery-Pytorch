"""
Creates PyTorch DataLoaders for image classification data.
"""
import os

from torchvision import datasets, transforms
from torch.utils.data import DataLoader

NUM_WORKERS = os.cpu_count()

class_names = ''

def create_dataloaders(
    train_dir: str, # training directory path
    test_dir: str, # testing directory path
    transform: transforms.Compose, # transforms to perform on training and testing
    batch_size: int, # number of samples per batch in each DataLoader
    num_workers: int=NUM_WORKERS # number of workers per DataLoader
):

  """

  Creates the training and testing DataLoaders.

  It takes in a training directory and testing directory path and turns them 
  into PyTorch Datasets and then into PyTorch DataLoaders.

  It returns a tuple of (train_dataloader, test_dataloader, class_names) where
  class_names is a list of the target classes. 
  
  """

  # Use ImageFolder to create dataset(s)
  train_data = datasets.ImageFolder(train_dir, transform=transform)
  test_data = datasets.ImageFolder(test_dir, transform=transform)

  class_names = train_data.classes

  train_dataloader = DataLoader(
      train_data, 
      batch_size=batch_size,
      shuffle=True,
      num_workers=num_workers,
      pin_memory=True
  )

  test_dataloader = DataLoader(
      test_data,
      batch_size=batch_size,
      shuffle=False,
      num_workers=num_workers,
      pin_memory=True
  )

  return train_dataloader, test_dataloader, class_names
