""" image classifier. 
helpful links:
- https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html

"""

import os, logging

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from PIL import Image

class ImageGrayScale(Dataset):
	""" Load any dataset of images, but only their grayscale values. """

	def __init__(self, root_dir, transform=transforms.Compose([transforms.Resize(255), transforms.ToTensor()])):
		"""
		Args:
			root_dir (string)             : Directory with all the images of one specific class.
			transform (callable, optional): Transform to be applied on a sample.
		"""

		self.transform = transform
		self.root_dir = root_dir

		# get all image paths in root_dir. since you are expected to use one
		# directory for one class of data (for instance dogs) there is no need
		# to label each sample from a specific dataset.
		self.image_paths = []
		for entry in os.scandir(root_dir):
			if (entry.path.endswith(".jpeg") or entry.path.endswith(".jpg") or entry.path.endswith(".png")) and entry.is_file():
				self.image_paths += [entry.path]

	def __len__(self):
		return len(self.image_paths)

	def __getitem__(self, idx):
		# slicing
		if isinstance(idx, slice):
			start, stop, step = idx.indices(len(self))
			return [self[i] for i in range(start, stop, step)]
		elif isinstance(idx, int):
			# load image and convert to grayscale
			sample = Image.open(self.image_paths[idx]).convert('L')
			if self.transform:
				sample = self.transform(sample)

			return sample
		else:
			logging.critical('CNN.__getitem__ is not implemented for idx of type %s ' % type(idx))

		return None

class CNN(nn.Module):
	""" Convolutional Neural Network for classification of grayscale images. """

	def __init__(self, lr=0.01):
		super(CNN, self).__init__()
		
		# network topology
		self.conv1 = nn.Conv2d(1, 6, 5)
		self.pool = nn.MaxPool2d(2, 2)
		self.conv2 = nn.Conv2d(6, 16, 5)
		self.fc1 = nn.Linear(16 * 5 * 5, 120)
		self.fc2 = nn.Linear(120, 84)
		self.fc3 = nn.Linear(84, 1)

		self.criterion = nn.CrossEntropyLoss()
		self.optimizer = optim.SGD(self.parameters(), lr=lr)
		self.scheduler = ReduceLROnPlateau(optimizer, 'min')

	def forward(self, x):
		x = self.pool(F.relu(self.conv1(x)))
		x = self.pool(F.relu(self.conv2(x)))
		x = x.view(-1, 16 * 5 * 5)
		x = F.relu(self.fc1(x))
		x = F.relu(self.fc2(x))
		x = self.fc3(x)
		return x

	def save(self, path):
		torch.save(self.state_dict(), path)

	def load(self, path):
		self.load_state_dict(torch.load(path))
		self.eval()

	def train(self, cycles, training_loader):
		""" Train the CNN for the given number of cycles on the given training dataset. 
			
		Args:
			cycles (int)								  : Number of training cycles.
			training_loader (torch.utils.data.DataLoader) : DataLoader containing the dataset.

		"""

		# all of our images are of one class, so of course all of them will have a 100% probability
		# of being a hit. don't forget to consider the batch_size of the training_loader !
		one_tensor = torch.tensor([[1] for i in range(training_loader.batch_size)]).float()

		for cycle in range(cycles):
			for batch in training_loader:
				self.optimizer.zero_grad() # don't forget to zero the gradient buffers per batch !
				self.loss = self.criterion(self(batch), one_tensor)
				
				# debugging loss
				if cycle % 10 == 0:
					logging.info('loss: %f' % self.loss)
				
				self.loss.backward()  # backward propagate loss
				self.optimizer.step() # update the parameters
				self.scheduler.step(self.loss, cycle) # adaptive learning rate: if loss doesn't decrease significantly
				                                      # then the learning rate will be decreased
				                                      # read here:
				                                      # https://github.com/pytorch/pytorch/pull/1370/commits/9f48a2cccb238aea13e2f170e60d48430e2b2aee

	def test(self, testing_loader):
		""" Test the CNN for the given testing dataset. 
			
		Args:
			testing_loader (torch.utils.data.DataLoader) : DataLoader containing the dataset.

		"""
		pass

def main():
	# customize your datasource here
	dogs = '/home/muesli/Downloads/dogscats/dogs'
	batch_size = 16
	data_total = ImageGrayScale(dogs)
	train_test_ratio = 0.9 # this would result in a 90:10 training:testing split

	# split data into training:testing datasets
	training_data = data_total[:int(train_test_ratio*len(data_total))]
	testing_data = data_total[int(train_test_ratio*len(data_total)):]

	# data loaders (sexy iterators)
	training_loader = torch.utils.data.DataLoader(training_data, batch_size=batch_size, shuffle=True, num_workers=2)
	testing_loader = torch.utils.data.DataLoader(testing_data, batch_size=batch_size, shuffle=True, num_workers=2)

	# customize your CNN here
	model_path = 'model'
	training_cycles = 1000
	learning_rate = 0.01

	# create a CNN
	net = CNN(lr=learning_rate)

	# load an existing model if possible
	try:
		net.load(model_path)
	except:
		logging.warning("could not load model %s, initializing empty model instead" % model_path)

	# train the model
	net.train(training_cycles, training_loader)

	# test the model accuracy
	net.test(testing_loader)

	# save/dump model
	net.save("model")

if __name__ == '__main__':
	main()
