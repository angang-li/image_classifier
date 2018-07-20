# Train a new network on a dataset and save the model as a checkpoint

'''
Basic usage: python train.py data_directory
Prints out training loss, validation loss, and validation accuracy as the network trains

Options:
- Set directory to save checkpoints: python train.py data_dir --save_dir save_directory
- Choose architecture: python train.py data_dir --arch "vgg13"
- Set hyperparameters: python train.py data_dir --learning_rate 0.01 --hidden_units 512 --epochs 20
- Use GPU for training: python train.py data_dir --gpu

Example usage:
python train.py flowers --gpu --save_dir assets
'''

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
import json

from workspace_utils import active_session
from train_preprocessing import preproc
from train_model import build_model, train_model

# Get the command line input into the scripts
parser = argparse.ArgumentParser()

# Basic usage: python train.py data_directory
parser.add_argument('data_directory', action='store',
                    default = 'flowers',
                    help='Set directory to load training data, e.g., "flowers"')

# Set directory to save checkpoints: python train.py data_dir --save_dir save_directory
parser.add_argument('--save_dir', action='store',
                    default = '.',
                    dest='save_dir',
                    help='Set directory to save checkpoints, e.g., "assets"')

# Choose architecture: python train.py data_dir --arch "vgg13"
parser.add_argument('--arch', action='store',
                    default = 'densenet121',
                    dest='arch',
                    help='Choose architecture, e.g., "vgg13"')

# Set hyperparameters: python train.py data_dir --learning_rate 0.01 --hidden_units 512 --epochs 20
parser.add_argument('--learning_rate', action='store',
                    default = 0.001,
                    dest='learning_rate',
                    help='Choose architecture learning rate, e.g., 0.01')

parser.add_argument('--hidden_units', action='store',
                    default = 512,
                    dest='hidden_units',
                    help='Choose architecture hidden units, e.g., 512')

parser.add_argument('--epochs', action='store',
                    default = 4,
                    dest='epochs',
                    help='Choose architecture number of epochs, e.g., 20')

# Use GPU for training: python train.py data_dir --gpu
parser.add_argument('--gpu', action='store_true',
                    default=False,
                    dest='gpu',
                    help='Use GPU for training, set a switch to true')

parse_results = parser.parse_args()

# print('data_directory     = {!r}'.format(parse_results.data_directory))
# print('save_dir     = {!r}'.format(parse_results.save_dir))
# print('arch     = {!r}'.format(parse_results.arch))
# print('learning_rate     = {!r}'.format(parse_results.learning_rate))
# print('hidden_units     = {!r}'.format(parse_results.hidden_units))
# print('epochs     = {!r}'.format(parse_results.epochs))
# print('gpu     = {!r}'.format(parse_results.gpu))

data_dir = parse_results.data_directory
save_dir = parse_results.save_dir
arch = parse_results.arch
learning_rate = float(parse_results.learning_rate)
hidden_units = int(parse_results.hidden_units)
epochs = int(parse_results.epochs)
gpu = parse_results.gpu

# Load and preprocess data
image_datasets, train_loader, valid_loader, test_loader = preproc(data_dir)

# Building and training the classifier
model_init = build_model(arch, hidden_units)
model, optimizer, criterion = train_model(model_init, train_loader, valid_loader, learning_rate, epochs, gpu)

# Save the checkpoint 
model.to('cpu')
model.class_to_idx = image_datasets['train'].class_to_idx
checkpoint = {'model': model,
              'state_dict': model.state_dict(),
              'optimizer_state_dict': optimizer.state_dict,
              'criterion': criterion,
              'epochs': epochs,
              'class_to_idx': model.class_to_idx}

torch.save(checkpoint, save_dir + '/checkpoint.pth')

if save_dir == ".":
    save_dir_name = "current folder"
else:
    save_dir_name = save_dir + " folder"

print(f'Checkpoint saved to {save_dir_name}.')