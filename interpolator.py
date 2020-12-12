""" image classifier client. """

import os, sys, time

import matplotlib.pyplot as plt
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from PIL import Image
from bhtsne import tsne

from classifier import DogDetector
from ml import GrayVAE

def main():


	# create a GrayVAE
	vae = GrayVAE()
	
	# load an existing model if possible
	vae.load('autoencoder.pth')
	vae.eval() # inference mode

	frames = vae.transform([('/home/kashim/Downloads/dogsncats/frames/' + ('%d' % i) + '.png' ) for i in range(1, 6)])

	e = vae.encode(frames)
	b = torch.tensor([])

	plt.ion()

	n = 8            # n interpolation points between two frames
	cur_frame = 0
	
	with torch.no_grad():
		while True:
			for i in range(n):
				f = vae.decode((i/n * e[cur_frame] + (1-i/n) * e[cur_frame+1]).unsqueeze(0))
				grid = torchvision.utils.make_grid(f, nrow=1)
			
				plt.imshow(vae.untransform(grid))
				#plt.imshow(vae.untransform(f.squeeze(0)), cmap='gray')
				
				plt.show()
				plt.pause(0.16)
				print(cur_frame)
				
			cur_frame += 1
			if cur_frame >= len(frames):
				cur_frame = 0

		
if __name__ == '__main__':
	main()
