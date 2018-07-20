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


# Predict class and probabilities
def predict(np_image, model, topk, gpu):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    
    # Implement the code to predict the class from an image file
    device = torch.device("cuda:0" if gpu else "cpu")

    model.to(device)
    model.eval()
    
    with torch.no_grad():
        images = torch.from_numpy(np_image)
        images = images.unsqueeze(0)
        images = images.type(torch.FloatTensor)
        images = images.to(device) # Move input tensors to the GPU/CPU

        output = model.forward(images)
        ps = torch.exp(output) # get the class probabilities from log-softmax

        probs, indices = torch.topk(ps, topk)
        probs = [float(prob) for prob in probs[0]]
        inv_map = {v: k for k, v in model.class_to_idx.items()}
        classes = [inv_map[int(index)] for index in indices[0]]
        
    return probs, classes