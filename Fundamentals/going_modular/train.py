""" 
Trains a PyTorch image classification model using device-agnostic code.
"""

import os
import argparse
import torch
import torchvision
import data_setup, engine, model_builder, utils
from torchvision import transforms

# Parser
parser = argparse.ArgumentParser(description="Enter hyperparameters.")

parser.add_argument("--num_epochs", default=5, type=int, help="The number of epochs to train for")
parser.add_argument("--batch_size", default=32, type=int, help="The number of samples per batch")
parser.add_argument("--hidden_units", default=32, type=int, help="The number of hidden units in hidden layers")
parser.add_argument("--learning_rate", default=0.001, type=float, help="The learning rate to use for the model")
parser.add_argument("--train_dir", default="../data/pizza_steak_sushi/train", type=str, help="The directory file path to training data in standard image classification format")
parser.add_argument("--test_dir", default="../data/pizza_steak_sushi/test", type=str, help="The directory file path to testing data in standard image classification format")

args = parser.parse_args()

# Hyper parameters
NUM_EPOCHS = args.num_epochs
BATCH_SIZE = args.batch_size
HIDDEN_UNITS = args.hidden_units
LEARNING_RATE = args.learning_rate

# Directories
train_dir = args.train_dir
test_dir = args.test_dir
# print(f"[INFO]\nEpochs:{NUM_EPOCHS}\nBatch size:{BATCH_SIZE}\nHidden units:{HIDDEN_UNITS}\nLearning rate:{LEARNING_RATE}\nTrain directory:{train_dir}\nTest directory:{test_dir}\n")

device = "cuda" if torch.cuda.is_available() else "cpu"

# Transforms
data_transform = transforms.Compose([
  transforms.Resize((64, 64)),
  transforms.ToTensor()
])

if __name__ == '__main__': # When num_workers > 0, the multiprocessing module can encounter issues by spawning worker processes. Wrapping it in this guard provides a clear entry point to avoid re-importing the module and leading to errors. 
  # DataLoaders with help from data_setup.py
  train_dataloader, test_dataloader, class_names = data_setup.create_dataloaders(
      train_dir=train_dir,
      test_dir=test_dir,
      transform=data_transform,
      batch_size=BATCH_SIZE
  )

  # Model with help from model_builder.py
  model = model_builder.TinyVGG(
      input_shape=3,
      hidden_units=HIDDEN_UNITS,
      output_shape=len(class_names)
  ).to(device)

  # Loss and optimizer
  loss_fn = torch.nn.CrossEntropyLoss()
  optimizer = torch.optim.Adam(model.parameters(),
                              lr=LEARNING_RATE)

  # Training with help from engine.py
  engine.train(model=model,
              train_dataloader=train_dataloader,
              test_dataloader=test_dataloader,
              loss_fn=loss_fn,
              optimizer=optimizer,
              epochs=NUM_EPOCHS,
              device=device)

  # Save the model with help from utils.py
  utils.save_model(model=model,
                  target_dir="models",
                  model_name="05_going_modular_script_mode_tinyvgg_model.pth")
