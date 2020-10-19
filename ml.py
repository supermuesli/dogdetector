""" image classifier. 
helpful links:
- https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html

"""

import os, logging, random

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from PIL import Image

class ImageGrayScale(Dataset):
	""" Load any dataset of images, but only their grayscale values. """

	def __init__(self, root_dir, im_size=255, transform=transforms.Compose([transforms.ToTensor()])):
		"""
		Args:
			root_dir (string)             : Directory with all the images of one specific class.
			im_size (int, optional)       : Images will be padded (black) into a square of length im_size, defaults to 255.
			transform (callable, optional): Transform to be applied on a sample, defaults to transforms.ToTensor().
		"""

		self.im_size = im_size
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

			# apply square padding (black) if aspect ratio is not 1:1
			if sample.size[0] != self.im_size or sample.size[1] != self.im_size:
				sample.thumbnail((self.im_size, self.im_size), Image.ANTIALIAS)
				
				# edge case where thumbnail cannot guarantee exact dimensions
				if sample.size[0] != self.im_size or sample.size[1] != self.im_size:
					sample = sample.resize((self.im_size, self.im_size), Image.ANTIALIAS)
				else:	
					padded_sample = Image.new("L", (self.im_size, self.im_size))
					padded_sample.paste(sample, ((self.im_size-sample.size[0])//2, (self.im_size-sample.size[1])//2))
					sample = padded_sample

			if self.transform:
				sample = self.transform(sample)

			return sample
		else:
			logging.critical('CNN.__getitem__ is not implemented for idx of type %s ' % type(idx))

		return None

class CNN(nn.Module):
	""" Convolutional Neural Network for classification of grayscale images. """

	def __init__(self, im_size=100, lr=0.01):
		super(CNN, self).__init__()
		
		self.im_size = im_size

		# network topology
		self.conv1 = nn.Conv2d(1, 6, 5)
		self.pool = nn.MaxPool2d(2, 2)
		self.conv2 = nn.Conv2d(6, 16, 5)
		self.fc1 = nn.Linear(16 * 60*60, 120)
		self.fc2 = nn.Linear(120, 84)
		self.fc3 = nn.Linear(84, 1)

		self.criterion = nn.MSELoss()
		self.optimizer = optim.SGD(self.parameters(), lr=lr)
		self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, 'min')

	def forward(self, x):
		x = self.pool(F.relu(self.conv1(x)))
		#x.register_hook(lambda t: logging.info(t.size())) # tensor debugging	
		x = self.pool(F.relu(self.conv2(x)))
		x = x.view(-1, 16 * 60*60) # flatten the self.conv2 convolution layer. i have no idea why 60*60, though
		x = F.relu(self.fc1(x))
		x = F.relu(self.fc2(x))
		x = F.relu(self.fc3(x)) 
		x = x.squeeze(-1)          # squeeze output into batch_dim

		# squash value to interval [0, 1]

		return x

	def save(self, path):
		torch.save(self.state_dict(), path)

	def load(self, path):
		self.load_state_dict(torch.load(path))
		self.eval()

	def train(self, cycles, training_loader):
		""" Train the CNN for the given number of cycles on the given training dataset. 
			
		Args:
			cycles (int)                                  : Number of training cycles.
			training_loader (torch.utils.data.DataLoader) : DataLoader containing the dataset.

		"""
		
		# alternataive inputs
		black_image = torch.tensor([[[[0 for x in range(self.im_size)] for y in range(self.im_size)]] for b in range(training_loader.batch_size)]).float()
		noise_image = torch.tensor([[[[random.randint(0, 255) for x in range(self.im_size)] for y in range(self.im_size)]] for b in range(training_loader.batch_size)]).float()

		for cycle in range(cycles):
			for batch in training_loader:
				self.optimizer.zero_grad() # don't forget to zero the gradient buffers per batch !

				choice = 'against batch tensor'
				if random.random() < 0.6:
					self.loss = self.criterion(self(batch), torch.ones(batch.size()[0]))
				# randomly throw in black- and noise-tensors with torch.zeros output
				else:
					if random.random() < 0.5:
						self.loss = self.criterion(self(black_image), torch.zeros(training_loader.batch_size))
						choice = 'against black tensor'	
					else:
						# also update noise tensor from time to time
						if random.random() < 0.3:
							noise_image = torch.tensor([[[[random.randint(0, 255) for x in range(self.im_size)] for y in range(self.im_size)]] for b in range(training_loader.batch_size)]).float()
	
						self.loss = self.criterion(self(noise_image), torch.zeros(training_loader.batch_size))
						choice = 'against noise tensor'

				# debugging loss
				if cycle % 10 == 9:
					logging.info('batch loss: %f\t%s\tcycle: %d' % (self.loss, choice, cycle))
				
				self.loss.backward()  # backward propagate loss
				self.optimizer.step() # update the parameters
				self.scheduler.step(self.loss)        # adaptive learning rate: if loss doesn't decrease significantly
				                                      # then the learning rate will be decreased
				                                      # read here:
				                                      # https://github.com/pytorch/pytorch/pull/1370/commits/9f48a2cccb238aea13e2f170e60d48430e2b2aee


	def test(self, testing_loader):
		""" Test the CNN for the given testing dataset. 
			
		Args:
			testing_loader (torch.utils.data.DataLoader) : DataLoader containing the dataset.

		"""

		correct = 0
		total = 0
		with torch.no_grad():
			for data in testing_loader:
				output = self(data)
				_, predicted = torch.max(outputs.data, 1)
				total += labels.size(0)
				correct += (predicted == labels).sum().item()

			logging.info('accuracy on testing-data: %f %%' % (100*correct/total))

def main():
	# customize logging
	log_level = logging.INFO
	logging.basicConfig(level=log_level)

	# customize your datasource here
	dogs = '/home/kashim/Downloads/dogsncats/dogs'
	image_size = 255       # resize and (black-border)-pad images to shape 255x255
	data_ratio = 0.1       # only use the first 1% of the dataset
	train_test_ratio = 0.6 # this would result in a 90:10 training:testing split
	batch_size = 32        # for batch gradient descent set batch_size = int(len(data_total)*train_test_ratio*data_ratio)
	data_total = ImageGrayScale(dogs, image_size)

	# split data into training:testing datasets
	training_data = data_total[:int(data_ratio*train_test_ratio*len(data_total))]
	#training_data = data_total[:13]
	#testing_data = data_total[int(data_ratio*train_test_ratio*len(data_total)):]
	#testing-data = data_total[10:20]

	# data loaders (sexy iterators)
	training_loader = torch.utils.data.DataLoader(training_data, batch_size=batch_size, shuffle=True, num_workers=2)
	#testing_loader = torch.utils.data.DataLoader(testing_data, batch_size=1, shuffle=True, num_workers=2)


	# customize your CNN here
	model_path = 'model'
	training_cycles = 1000
	learning_rate = 0.000001


	# create a CNN
	net = CNN(image_size, lr=learning_rate)

	# load an existing model if possible
	try:
		net.load(model_path)
	except:
		logging.warning("could not load model %s, initializing empty model instead" % model_path)

	# train the model
	net.train(training_cycles, training_loader)

	# test the model accuracy
	#net.test(testing_loader)

	# save/dump model
	net.save("model")

if __name__ == '__main__':
	main()
