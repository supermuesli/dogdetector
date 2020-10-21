""" image classifier. 
helpful links:
- https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html

"""

import os, sys

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from PIL import Image

from ml import CNN


def main():

	# customize your datasource here
	dogs = '/home/kashim/Downloads/dogsncats/dogs'

	# customize your CNN here
	model_path = '/home/kashim/Documents/github/supermuesli/dogdetector/model.pth'

	# create a CNN
	net = CNN('cpu', 115)
	# load an existing model if possible
	
	net.load(model_path)

	sample = transforms.Compose([transforms.ToTensor()])(Image.open(dogs + '/12472.jpg').convert('L'))
	net(sample)

if __name__ == '__main__':
	main()
