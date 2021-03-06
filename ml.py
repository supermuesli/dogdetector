""" universal grayscale image classifier. see https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html """

import os, logging, random, sys

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from PIL import Image

class ImageGrayScale():
	""" Load any collection of images, but only their grayscale values. """

	def __init__(self, root_dir, im_size=255, transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])):
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

					# debugging
					# sample.show()

			if self.transform:
				sample = self.transform(sample)

			return sample
		else:
			logging.critical('CNN.__getitem__ is not implemented for idx of type %s ' % type(idx))

		return None

def binary_likelihood_loss(pred_probs, true_labels):
	""" """

	# read here: https://stackoverflow.com/questions/60922782/how-can-i-count-the-number-of-1s-and-0s-in-the-second-dimension-of-a-pytorch-t

	ones =  ((true_labels == 0).sum(dim=0)).float() # number of positive samples in training batch
	zeros = (true_labels.shape[0] - ones).float()   # number of negative samples in training batch

	# note that pred_prob is the probability of a sample being a positive
	# and 1 - preb_prob is the probability of a sample being negative

	loss = (((ones - pred_probs*pred_probs.shape[0])**2 + (zeros - (1 - pred_probs)*pred_probs.shape[0])**2) ).mean()

	"""
	ones big, pred_prob big      -> loss small
	zeros big, 1-pred_prob big   -> loss small
	ones small, pred_prob big    -> loss big
	zeros small, 1-pred_prob big -> loss big

	"""
	
	return loss

class DynamicBatchDataLoader():
	def __init__(self, training_data, batch_size=1, bs_multiplier=1.001, shuffle=True):
		
		self.training_data = training_data
		
		self.shuffle = shuffle
		if self.shuffle:
			random.shuffle(self.training_data)

		self.batch_size = batch_size
		self.offset = 0
		self.bs_multiplier = bs_multiplier
		self.bs_value = batch_size

	def __len__(self):
		return len(self.training_data)

	def __getitem__(self, idx):
		return self.training_data[idx]

	def __iter__(self):
		if self.shuffle: 
			random.shuffle(self.training_data)

		for b in range(self.batch_size):
			x = torch.tensor([])
			
			upper_limit = self.offset + self.batch_size
			
			if upper_limit > len(self):
				self.offset = 0
				upper_limit = self.batch_size

			#logging.debug('dynamic batch offset/upper_limit: %d/%d' % (self.offset, upper_limit))

			for i in range(self.offset, upper_limit, 1):	
				x = torch.cat((x, self.training_data[i].unsqueeze(0)), 0)

				self.offset += 1
				if self.offset >= len(self):
					self.offset = 0
			
			yield x

	def step(self):
		""" Increase batch_size, for instance per epoch. """
		if self.bs_value * self.bs_multiplier < len(self):
			self.bs_value *= self.bs_multiplier
			self.batch_size = int(self.bs_value)

		
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
		self.fc1 = nn.Linear(out_dim * ((((im_size - kernel_size) // pool_size) - kernel_size) // pool_size)**2, 64) # consider self.forward as to why this input dimension was chosen, also read here:
		                                                                                                             # https://stackoverflow.com/questions/53784998/how-are-the-pytorch-dimensions-for-linear-layers-calculated
		
		self.fc2 = nn.Linear(64, 32)
		self.fc3 = nn.Linear(32, 1)

		self.criterion = nn.BCELoss()
		#self.criterion = binary_likelihood_loss

		# read https://openreview.net/pdf?id=B1Yy1BxCZ
		self.optimizer = optim.SGD(self.parameters(), lr=lr, weight_decay=0.01, momentum=0.9)
		self.scheduler = optim.lr_scheduler.CyclicLR(self.optimizer, base_lr=lr, max_lr=lr*100)

		# gpu computation if possible, else cpu
		self.device = device
		self.to(self.device)

		# gradient clipping in order to prevent nan values for loss
		for p in self.parameters():
			p.register_hook(lambda grad: torch.clamp(grad, -100, 100))

	def forward(self, x):
		x = self.pool(torch.sigmoid(self.conv1(x)))
		x = self.pool(torch.sigmoid(self.conv2(x)))
		x = x.view(x.size(0), -1)                   # flatten the self.conv2 convolution layer while keeping batch_dim. 
		x = self.fc1(x)
		x = self.fc2(x)
		x = self.fc3(x)
		
		x = x.squeeze(-1)                    # squeeze output into batch_dim


		#x = x - x.min()                      # normalize tensor to range [0, 1]
		#x = x / x.max() if x.max() != 0 else x
		x = torch.sigmoid(x)
		
		return x

	def save(self, path):
		""" Save the CNN model as an altered state dict (.asd). """

		# manually add self.im_size to state_dict so we can restore it when loading the model again
		sd = self.state_dict()
		sd['im_size'] = self.im_size
	
		logging.warning('writing model into %s.asd, do not kill this process or %s.asd will be corrupted!' % (path, path))
		torch.save(sd, path + '.asd')
		logging.warning('done writing.')
		
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

		transf = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

		# a list of paths was given
		if type(path) == list:
			batch = transf(self.im_transform(path[0])).unsqueeze(0)
			for i, p in enumerate(path):
				if i > 0:
					# see https://discuss.pytorch.org/t/concatenate-torch-tensor-along-given-dimension/2304
					batch = torch.cat((batch, transf(self.im_transform(p)).unsqueeze(0)), 0)
		# a single path was given
		else:
			# add batch_dim (1)
		 	# see https://stackoverflow.com/questions/57237352/what-does-unsqueeze-do-in-pytorch
			return transf(self.im_transform(path)).unsqueeze(0)  

		return batch

	def fit(self, cycles, training_loader, save_per_cycle=1):
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

				# outputs if the batch is left as is (batch contains only positive samples at this point)
				target = torch.tensor([[1] for b in range(batch.shape[0])]) # class 0 is a dog
				
				for b in range(len(batch)):

					# with some probability, we want to introduce negative samples, but to prevent
					# predictability based on class distribution in the training dataset, the randomness
					# is supposed to be uniformly random as well
					if random.random() < 0.5:
						if False:
							for color_channel in range(len(batch[b])):
								for row in range(len(batch[b][color_channel])):
									# black 
									batch[b][color_channel][row] = torch.tensor([-1 for j in range(self.im_size)])
						
						else:
							for color_channel in range(len(batch[b])):
								for row in range(len(batch[b][color_channel])):	
									# noise
									batch[b][color_channel][row] = torch.tensor([2*random.random()-1 for j in range(self.im_size)])

						# image changes, so output changes to class 1
						target[b] = torch.tensor([0]) # class 1 is not a dog

					# tensor debugging (what are you really feeding into the neural network?). uncomment the next line if not needed.
					#transforms.ToPILImage()(batch[b]).show()

				self.loss = self.criterion(self(batch).unsqueeze(1), target.float()) # note that self(batch) outputs the probability of the input being a dog, 
				                                                                     # while target holds the actual class of the input
			
				# debugging loss
				if cycle % 10 == 9:
					logging.info('batch loss@size: %f@%d\tcycle: %d' % (self.loss, batch.size()[0], cycle))
				
				self.loss.backward()   # backward propagate loss
				self.optimizer.step()  # update the parameters
				#self.scheduler.step()  # dynamic learning rate
				training_loader.step() # dynamic batch size
				
				# cycle is finished at this point
				if cycle % save_per_cycle == save_per_cycle - 1:
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
	dogs = sys.argv[1]     # TODO: use doc_opt instead of sys.argv
	image_size = 75        # resize and (black-border)-pad images to image_size x image_size
	data_ratio = 1         # only use the first data_ratio*100% of the dataset
	train_test_ratio = 0.6 # this would result in a train_test_ratio*100%:(100-train_test_ratio*100)% training:testing split
	batch_size = 64         # for batch gradient descent set batch_size = int(len(data_total)*train_test_ratio*data_ratio)
	data_total = ImageGrayScale(dogs, image_size)
	#batch_size = int(len(data_total)*train_test_ratio*data_ratio)y

	# split data into training:testing datasets
	training_data = data_total[:int(data_ratio*train_test_ratio*len(data_total))]
	#training_data = data_total[:13]
	#testing_data = data_total[int(data_ratio*train_test_ratio*len(data_total)):]
	#testing-data = data_total[10:20]

	# data loaders (sexy iterators)
	training_loader = DynamicBatchDataLoader(training_data, batch_size=batch_size, bs_multiplier=1.001, shuffle=True)
	#testing_loader = torch.utils.data.DataLoader(testing_data, batch_size=1, shuffle=True)


	# customize your CNN here
	model_path = 'model.asd'
	cycles = 1000000
	learning_rate = 0.001
	save_per_cycle = 100  # save model every 100 cycles

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
