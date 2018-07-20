# Use a trained network to predict the flower name from an input image along with the probability of that name

'''
Basic usage: python predict.py /path/to/image checkpoint

Options:
- Return top KK most likely classes: python predict.py input checkpoint --top_k 3
- Use a mapping of categories to real names: python predict.py input checkpoint --category_names cat_to_name.json
- Use GPU for inference: python predict.py input checkpoint --gpu

Example usage: 
python predict.py flowers/test/1/image_06743.jpg assets
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
from predict_preprocessing import process_image, imshow
from predict_model import predict

# Get the command line input into the scripts
parser = argparse.ArgumentParser()

# Basic usage: python predict.py /path/to/image checkpoint
parser.add_argument('image_path', action='store',
                    default = 'flowers/test/1/image_06743.jpg',
                    help='Path to image, e.g., "flowers/test/1/image_06743.jpg"')

parser.add_argument('checkpoint', action='store',
                    default = '.',
                    help='Directory of saved checkpoints, e.g., "assets"')

# Return top KK most likely classes: python predict.py input checkpoint --top_k 3
parser.add_argument('--top_k', action='store',
                    default = 5,
                    dest='top_k',
                    help='Return top KK most likely classes, e.g., 5')

# Use a mapping of categories to real names: python predict.py input checkpoint --category_names cat_to_name.json
parser.add_argument('--category_names', action='store',
                    default = 'cat_to_name.json',
                    dest='category_names',
                    help='File name of the mapping of flower categories to real names, e.g., "cat_to_name.json"')

# Use GPU for inference: python predict.py input checkpoint --gpu
parser.add_argument('--gpu', action='store_true',
                    default=False,
                    dest='gpu',
                    help='Use GPU for inference, set a switch to true')

parse_results = parser.parse_args()

# print('image_path     = {!r}'.format(parse_results.image_path))
# print('checkpoint     = {!r}'.format(parse_results.checkpoint))
# print('top_k     = {!r}'.format(parse_results.top_k))
# print('category_names     = {!r}'.format(parse_results.category_names))
# print('gpu     = {!r}'.format(parse_results.gpu))

image_path = parse_results.image_path
checkpoint = parse_results.checkpoint
top_k = int(parse_results.top_k)
category_names = parse_results.category_names
gpu = parse_results.gpu

# Label mapping
with open(category_names, 'r') as f:
    cat_to_name = json.load(f)

# Load the checkpoint
filepath = checkpoint + '/checkpoint.pth'
checkpoint = torch.load(filepath, map_location='cpu')
model = checkpoint["model"]
model.load_state_dict(checkpoint['state_dict'])

# Image preprocessing
np_image = process_image(image_path)
# imshow(np_image)

# Predict class and probabilities
print(f"Predicting top {top_k} most likely flower names from image {image_path}.")

probs, classes = predict(np_image, model, top_k, gpu)
classes_name = [cat_to_name[class_i] for class_i in classes]

# print("Flower names: ", classes_name)
# print("Probabilities: ", [round(prob, 3) for prob in probs]) 

print("\nFlower name (probability): ")
print("---")
for i in range(len(probs)):
    print(f"{classes_name[i]} ({round(probs[i], 3)})")
print("")