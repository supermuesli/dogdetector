""" universal grayscale image classifier. see https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html """

import os, logging, random, sys

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

class DynamicBatchDataLoader(torch.utils.data.DataLoader):
	def __init__(self, training_data, batch_size, shuffle):
		super(DynamicBatchDataLoader, self).__init__(training_data, batch_size=batch_size, shuffle=shuffle, num_workers=1)
		self.training_data_ = training_data
		self.batch_size_ = batch_size
		self.shuffle_ = shuffle

	def step(self):
		""" Increase batch_size, for instance per epoch. """
		print("bs:",self.batch_size_)
		print(self.__initialized)
		self.__initialized = False
		self.__init__(self.training_data_, 2*self.batch_size_, self.shuffle_)

class CNN(nn.Module):
	""" Convolutional Neural Network for classification of grayscale images. """

	def __init__(self, device='cpu', im_size=100, lr=0.01):
		super(CNN, self).__init__()
		
		self.im_size = im_size

		# network topology
		kernel_size = 5	
		pool_size = 2
		out_dim = 6
		self.conv1 = nn.Conv2d(1, out_dim, kernel_size)
		self.pool = nn.MaxPool2d(pool_size, pool_size)
		
		in_dim = out_dim
		out_dim = 16
		self.conv2 = nn.Conv2d(in_dim, out_dim, kernel_size)
		self.fc1 = nn.Linear(out_dim * ((((im_size - kernel_size) // pool_size) - kernel_size) // pool_size)**2, 1200) # consider self.forward as to why this input dimension was chosen, also read here:
		                                                                                                               # https://stackoverflow.com/questions/53784998/how-are-the-pytorch-dimensions-for-linear-layers-calculated
		
		self.fc2 = nn.Linear(1200, 84)
		self.fc3 = nn.Linear(84, 1)

		self.criterion = nn.MSELoss()

		# read https://openreview.net/pdf?id=B1Yy1BxCZ
		self.optimizer = optim.SGD(self.parameters(), lr=lr, momentum=0.9)
		self.scheduler = optim.lr_scheduler.CyclicLR(self.optimizer, base_lr=lr/100, max_lr=lr*100)

		self.device = device
		#self.to(self.device)

	def forward(self, x):
		x = self.pool(F.relu(self.conv1(x)))
		x = self.pool(F.relu(self.conv2(x)))
		x = x.view(-1, self.fc1.in_features) # flatten the self.conv2 convolution layer. 
		x = self.fc1(x)
		x = self.fc2(x)
		x = self.fc3(x) 
		x = x.squeeze(-1)                    # squeeze output into batch_dim

		# squash value to interval [0, 1].
		min_val = abs(x.min())
		max_val = x.max() + min_val
		if max_val == 0:
			max_val = 1
		x = (x + min_val)/ max_val

		return x

	def save(self, path):
		""" Save the CNN model as an altered state dict (.asd). """

		# manually add self.im_size to state_dict so we can restore it when loading the model again
		sd = self.state_dict()
		sd['im_size'] = self.im_size
	
		torch.save(sd, path + '.asd')
		
	def load(self, path):
		sd = torch.load(path)
		
		# load non-conventional attributes from altered state dict (if they exist)
		try:
			self.im_size = sd['im_size']
			del sd['im_size']
		
			self.__init__(im_size=self.im_size)
		except KeyError as e:
			logging.critical(e)
		# load conventional state dict
		
		self.load_state_dict(sd)
		
		self.eval()

	def im_transform(self, path):
		""" Resize and pad image to self.im_size. """

		sample = Image.open(path).convert('L')

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

		return sample

	def transform(self, path):
		""" Transform a sample input so it fits through the network topology. """

		# a list of paths was given
		if type(path) == list:
			batch = transforms.ToTensor()(self.im_transform(path[0])).unsqueeze(0)
			for i, p in enumerate(path):
				if i > 0:
					# see https://discuss.pytorch.org/t/concatenate-torch-tensor-along-given-dimension/2304
					batch = torch.cat((batch, transforms.ToTensor()(self.im_transform(p)).unsqueeze(0)), 0)
		# a single path was given
		else:
			# add batch_dim (1)
		 	# see https://stackoverflow.com/questions/57237352/what-does-unsqueeze-do-in-pytorch
			return transforms.ToTensor()(self.im_transform(path)).unsqueeze(0)  

		return batch

	def fit(self, cycles, training_loader, save_per_cycle=False):
		""" Train the CNN for the given number of cycles on the given training dataset. 
			
		Args:
			cycles (int)                                  : Number of training cycles.
			training_loader (torch.utils.data.DataLoader) : DataLoader containing the dataset.

		"""
	
		# the training loop utilizes mini-batch gradient descent

		cycle = 0		
		continue_training = True
		while continue_training:
			for batch in training_loader:
				self.optimizer.zero_grad() # don't forget to zero the gradient buffers per batch !

				# outputs
				target = torch.ones(batch.size()[0])
				
				for i in range(len(batch)):
					if random.random() < 0.7:
						for color_channel in range(len(batch[i])):
							
							for row in range(len(batch[i][color_channel])):
						
								if random.random() < 0.5:
									# black 
									batch[i][color_channel][row] = torch.tensor([0 for i in range(self.im_size)])
								else:
									# noise
									batch[i][color_channel][row] = torch.tensor([random.random() for i in range(self.im_size)])

								# row changes, output changes to 0
								target[i] = torch.tensor([0])
				
				choice = 'against batch/black/noise'
				self.loss = self.criterion(self(batch), target)

				# debugging loss
				if cycle % 10 == 9:
					logging.info('batch loss: %f\t%s\tcycle: %d' % (self.loss, choice, cycle))
				
				self.loss.backward()   # backward propagate loss
				self.optimizer.step()  # update the parameters
				self.scheduler.step()  # dynamic learning rate
				training_loader.step() # dynamic batch size
				
				# cycle is finished at this point
				if save_per_cycle:
					self.save('model')

				cycle += 1
				if cycle == cycles:
					continue_training = False
					break

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

	# if CUDA available, use it
	if torch.cuda.is_available():  
		dev = 'cuda:0' 
	else:  
		dev = 'cpu'  
	device = torch.device(dev) 

	# customize your datasource here
	dogs = '/home/kashim/Downloads/dogsncats/dogs'
	image_size = 115       # resize and (black-border)-pad images to image_size x image_size
	data_ratio = 0.1       # only use the first data_ratio*100% of the dataset
	train_test_ratio = 0.6 # this would result in a train_test_ratio*100%:(100-train_test_ratio*100)% training:testing split
	batch_size = 1         # for batch gradient descent set batch_size = int(len(data_total)*train_test_ratio*data_ratio)
	data_total = ImageGrayScale(dogs, image_size)
	#batch_size = int(len(data_total)*train_test_ratio*data_ratio)

	# split data into training:testing datasets
	training_data = data_total[:int(data_ratio*train_test_ratio*len(data_total))]
	#training_data = data_total[:13]
	#testing_data = data_total[int(data_ratio*train_test_ratio*len(data_total)):]
	#testing-data = data_total[10:20]

	# data loaders (sexy iterators)
	training_loader = DynamicBatchDataLoader(training_data, batch_size=batch_size, shuffle=True)
	#testing_loader = torch.utils.data.DataLoader(testing_data, batch_size=1, shuffle=True)


	# customize your CNN here
	model_path = 'model.asd'
	cycles = 100
	learning_rate = 1
	save_per_cycle = True

	# create a CNN
	net = CNN(im_size=image_size, lr=learning_rate)

	# load an existing model if possible
	#net.load(model_path)

	# train the model
	net.fit(cycles, training_loader, save_per_cycle=save_per_cycle)

	# test the model accuracy
	#net.test(testing_loader)

	# save/dump model
	net.save("model")

if __name__ == '__main__':
	main()
