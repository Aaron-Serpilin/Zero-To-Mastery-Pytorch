
""" 
Contains functions for training and testing a PyTorch model.
"""

import torch
from tqdm.auto import tqdm
from typing import Dict, List, Tuple

def train_step(
    model: torch.nn.Module, # model to be trained
    dataloader: torch.utils.data.DataLoader, # DataLoader instance for the model to be trained on
    loss_fn: torch.nn.Module, # loss function to minimize
    optimizer: torch.optim.Optimizer, # optimizer to help minimize the loss function
    device: torch.device
    ) -> Tuple[float, float]:

    """ 

    Trains a PyTorch model for a single epoch

    Turns a target PyTorch model to training mode and then runs through all of the training steps:
        Forward Pass
        Loss Calculation
        Optimizer Step

    It returns a tuple of training loss and training accuracy metrics

    """

    model.train()

    train_loss, train_acc = 0, 0

    for batch, (X, y) in enumerate(dataloader):

        X, y = X.to(device), y.to(device)

        # Forward pass
        y_pred = model(X)

        # Calculate and accumulate loss
        loss = loss_fn(y_pred, y)
        train_loss += loss.item()

        # Optimizer zero grad
        optimizer.zero_grad()

        # Loss backward
        loss.backward()

        # Optimizer step
        optimizer.step()

        # Accuracy metric across all batches
        y_pred_class = torch.argmax(torch.softmax(y_pred, dim=1), dim=1)
        train_acc += (y_pred_class == y).sum().item()/len(y_pred)

    train_loss /= len(dataloader)
    train_acc /= len(dataloader)
    return train_loss, train_acc

def test_step(
    model: torch.nn.Module, # model to be tested
    dataloader: torch.utils.data.DataLoader, # DataLoader instance for the model to be tested on
    loss_fn: torch.nn.Module, # loss function to calculate loss on the test data
    device: torch.device 
    ) -> Tuple[float, float]:

    """ 

    Test a PyTorch model for a single epoch

    Turns a target PyTorch model to "eval" mode and then performs a forward pass on a testing dataset

    It returns a tuple of testing loss and testing accuracy metrics

    """

    model.eval()

    test_loss, test_acc = 0, 0

    with torch.inference_mode():

        for batch, (X, y) in enumerate(dataloader):
            X, y = X.to(device), y.to(device)

            # Forward pass
            test_pred_logits = model(X)

            # Calculate and accumulate loss
            loss = loss_fn(test_pred_logits, y)
            test_loss = loss.item()

            test_pred_labels = test_pred_logits.argmax(dim=1)
            test_acc += ((test_pred_labels == y).sum().item()/len(test_pred_labels))

    test_loss /= len(dataloader)
    test_acc /= len(dataloader)
    return test_loss, test_acc

def train(
    model: torch.nn.Module, # model to be trained and tested
    train_dataloader: torch.utils.data.DataLoader, # DataLoader instance for the model to be trained on
    test_dataloader: torch.utils.data.DataLoader, # DataLoader instance for the model to be tested on
    optimizer: torch.optim.Optimizer, # optimizer to help minimize the loss function
    loss_fn: torch.nn.Module, # loss function to calculate loss on both datasets
    epochs: int,
    device: torch.device
    ) -> Dict[str, List]:

    """ 
    
    Trains and tests a PyTorch model

    Passes a target PyTorch model through the train_step() and test_step() functions for a number of epochs,
    training and testing the model in the same epoch loop

    It calculates, prints and stores evaluation metrics throughout

    It returns a dictionary of training and testing loss as well as training and testing accuracy metrics.
    Each metric has a value in a list for each epoch

    """

    results = {
        "train_loss": [],
        "train_acc": [], 
        "test_loss": [],
        "test_acc": []
    }

    for epoch in tqdm(range(epochs)):
        
        train_loss, train_acc = train_step(
            model=model, 
            dataloader=train_dataloader,
            loss_fn=loss_fn,
            optimizer=optimizer,
            device=device
        )

        test_loss, test_acc = test_step(
            model=model,
            dataloader=test_dataloader,
            loss_fn=loss_fn,
            device=device
        )

        print(f"Epoch: {epoch+1} | train_loss: {train_loss:.4f} | train_acc: {train_acc:.4f} | test_loss: {test_loss:.4f} | test_acc: {test_acc:.4f}")

        results["train_loss"].append(train_loss)
        results["train_acc"].append(train_acc)
        results["test_loss"].append(test_loss)
        results["test_acc"].append(test_acc)

    return results
