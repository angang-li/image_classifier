# Dependencies
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
from torchvision import datasets, transforms, models
from torch import nn, optim
import torch.nn.functional as F
from collections import OrderedDict
import time
from PIL import Image
import matplotlib

from workspace_utils import active_session


# Build model
def build_model(arch, hidden_units):

    # Load in a pre-trained model, DenseNet default
    if arch.lower() == "vgg13":
        model = models.vgg13(pretrained=True)
    else:
        model = models.densenet121(pretrained=True)

    # Define a new, untrained feed-forward network as a classifier, using ReLU activations and dropout
    for param in model.parameters():
        param.requires_grad = False # Freeze parameters so we don't backprop through them

    if arch.lower() == "vgg13":
        classifier = nn.Sequential(OrderedDict([
                            ('dropout1', nn.Dropout(0.1)),
                            ('fc1', nn.Linear(25088, hidden_units)), # 25088 must match
                            ('relu1', nn.ReLU()),
                            ('dropout2', nn.Dropout(0.1)),
                            ('fc2', nn.Linear(hidden_units, 102)),
                            ('output', nn.LogSoftmax(dim=1))
                            ]))
    else:
        classifier = nn.Sequential(OrderedDict([
                            ('dropout1', nn.Dropout(0.1)),
                            ('fc1', nn.Linear(1024, hidden_units)), # 1024 must match
                            ('relu1', nn.ReLU()),
                            ('dropout2', nn.Dropout(0.1)),
                            ('fc2', nn.Linear(hidden_units, 102)),
                            ('output', nn.LogSoftmax(dim=1))
                            ]))

    model.classifier = classifier
    print(f"Model built from {arch} and {hidden_units} hidden units.")

    return model


# Measure the validation loss and accuracy
def validation(model, dataloader, criterion, device):
    loss = 0
    accuracy = 0
    with torch.no_grad():
        for images, labels in iter(dataloader):
            
            images, labels = images.to(device), labels.to(device) # Move input and label tensors to the GPU
            
            output = model.forward(images)
            loss += criterion(output, labels).item()

            ps = torch.exp(output) # get the class probabilities from log-softmax
            equality = (labels.data == ps.max(dim=1)[1])
            accuracy += equality.type(torch.FloatTensor).mean()
    
    return loss, accuracy


# Train model
def train_model(model, train_loader, valid_loader, learning_rate, epochs, gpu):

    # Criterion and optimizer
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=learning_rate)

    # Train the classifier layers using backpropagation using the pre-trained network to get the features
    device = torch.device("cuda:0" if gpu else "cpu")
    print(type(model))
    model.to(device)
    print_every = 40
    steps = 0
    running_loss = 0
    train_accuracy = 0

    print(f'Training with {learning_rate} learning rate, {epochs} epochs, and {(gpu)*"cuda" + (not gpu)*"cpu"} computing.')

    with active_session():
        for e in range(epochs):

            model.train() # Dropout is turned on for training

            for images, labels in iter(train_loader):

                images, labels = images.to(device), labels.to(device) # Move input and label tensors to the GPU

                steps += 1
                optimizer.zero_grad()
                output = model.forward(images)
                loss = criterion(output, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()

                # get the class probabilities from log-softmax
                ps = torch.exp(output) 
                equality = (labels.data == ps.max(dim=1)[1])
                train_accuracy += equality.type(torch.FloatTensor).mean()

                if steps % print_every == 0:

                    model.eval() # Make sure network is in eval mode for inference

                    with torch.no_grad():
                        valid_loss, valid_accuracy = validation(model, valid_loader, criterion, device)

                    print("Epoch: {}/{}.. ".format(e+1, epochs),
                        "Training Loss: {:.3f}.. ".format(running_loss/print_every),
                        "Training Accuracy: {:.3f}".format(train_accuracy/print_every),
                        "Validation Loss: {:.3f}.. ".format(valid_loss/len(valid_loader)),
                        "Validation Accuracy: {:.3f}".format(valid_accuracy/len(valid_loader)))

                    running_loss = 0
                    train_accuracy = 0
                    model.train() # Make sure training is back on
                    
        print("\nTraining completed!")
    
    return model, optimizer, criterion