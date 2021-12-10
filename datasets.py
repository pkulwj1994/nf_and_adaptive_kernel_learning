import os
import numpy as np
import pickle
from torch.utils import data

import torch
from torchvision.datasets import MNIST
from torchvision.transforms import Compose, ToTensor
from torch.utils.data import DataLoader, ConcatDataset, Subset
from sklearn.model_selection import train_test_split

from torchvision.datasets import CIFAR10



################################ mnist dataset part

class UnsupervisedMNIST(MNIST):
	def __init__(self,root='./',train=True,transform=None,download=False):
		super(UnsupervisedMNIST,self).__init__(root,train=train,transform=transform,download=download)

	def __getitem__(self,index):

		return super(UnsupervisedMNIST, self).__getitem__(index)[0]

	@property
	def raw_folder(self):
		return os.path.join(self.root, 'MNIST','raw')
	@property
	def processed_folder(self):
		return os.path.join(self.root,'MNIST','processed')

class Flatten():
	def __call__(self,image):
		return image.view(-1)

class Cifar10Flatten():
	def __call__(self,image):
		return image.reshape([3*32*32])

class StaticBinarize():
	def __call__(self,image):
		return image.round().long()

class DynamicBinarize():
	def __call__(self,image):
		return image.bernoulli().long()


class Quantize():
	def __init__(self, num_bits=8):
		self.num_bits = num_bits

	def __call__(self, image):
		image = image*255
		if self.num_bits !=8:
			image = torch.floor(image/2**(8-self.num_bits))
		return image

class Dequantize():
	def __call__(self, image):
		return (image*255 + torch.rand_like(image))/256



class MNIST_Normalize():

	def __call__(self,image):
		return (2 * image.float() - 1)

class MNIST_Normalizev2():

	def __init__(self, num_bits=8):
		self.num_bits = num_bits

	def __call__(self, image):
		img = image*255 + torch.rand_like(image)
		img = img/256	
		return (2*img.float()-1)




class DynamicallyBinarizedMNIST():

	def __init__(self,root='./',download=True,flatten=False):

		self.root = root

		trans = [ToTensor(),DynamicBinarize()]
		if flatten: trans.append(Flatten())

		self.train = UnsupervisedMNIST(root,train=True, transform=Compose(trans), download=download)
		self.test = UnsupervisedMNIST(root,train=False, transform=Compose(trans))

		self.splits = ['train','test']

	@property
	def num_splits(self):
		return len(self.splits)

	def get_data_loader(self,split, batch_size, shuffle=True, pin_memory=True, num_workers=4):
		return DataLoader(getattr(self,split),batch_size=batch_size, shuffle=shuffle,pin_memory=pin_memory,num_workers=num_workers)

	def get_data_loaders(self, batch_size, shuffle=True, pin_memory=True, num_workers=4):
		data_loaders = [self.get_data_loader(split=split,batch_size=batch_size,shuffle=shuffle,
			pin_memory=pin_memory,num_workers=num_workers) for split in self.splits]
		return data_loaders


class QuantizedMNIST():

	def __init__(self,root='./',download=True,flatten=False):

		self.root = root

		trans = [ToTensor(),Quantize()]
		if flatten: trans.append(Flatten())

		self.train = UnsupervisedMNIST(root,train=True, transform=Compose(trans), download=download)
		self.test = UnsupervisedMNIST(root,train=False, transform=Compose(trans))

		self.splits = ['train','test']

	@property
	def num_splits(self):
		return len(self.splits)

	def get_data_loader(self,split, batch_size, shuffle=True, pin_memory=True, num_workers=4):
		return DataLoader(getattr(self,split),batch_size=batch_size, shuffle=shuffle,pin_memory=pin_memory,num_workers=num_workers)

	def get_data_loaders(self, batch_size, shuffle=True, pin_memory=True, num_workers=4):
		data_loaders = [self.get_data_loader(split=split,batch_size=batch_size,shuffle=shuffle,
			pin_memory=pin_memory,num_workers=num_workers) for split in self.splits]
		return data_loaders


class NormalizeMNIST():

	def __init__(self,root='./',download=True,flatten=True):

		self.root = root

		trans = [ToTensor(),MNIST_Normalize()]
		if flatten: trans.append(Flatten())

		self.train = UnsupervisedMNIST(root,train=True, transform=Compose(trans), download=download)
		self.test = UnsupervisedMNIST(root,train=False, transform=Compose(trans))

		self.splits = ['train','test']

	@property
	def num_splits(self):
		return len(self.splits)

	def get_data_loader(self,split, batch_size, shuffle=True, pin_memory=True, num_workers=4):
		return DataLoader(getattr(self,split),batch_size=batch_size, shuffle=shuffle,pin_memory=pin_memory,num_workers=num_workers)

	def get_data_loaders(self, batch_size, shuffle=True, pin_memory=True, num_workers=4):
		data_loaders = [self.get_data_loader(split=split,batch_size=batch_size,shuffle=shuffle,
			pin_memory=pin_memory,num_workers=num_workers) for split in self.splits]
		return data_loaders

class NormalizeMNISTv2():

	def __init__(self,root='./',download=True,flatten=True):

		self.root = root

		trans = [ToTensor(),MNIST_Normalizev2()]
		if flatten: trans.append(Flatten())

		self.train = UnsupervisedMNIST(root,train=True, transform=Compose(trans), download=download)
		self.test = UnsupervisedMNIST(root,train=False, transform=Compose(trans))

		self.splits = ['train','test']

	@property
	def num_splits(self):
		return len(self.splits)

	def get_data_loader(self,split, batch_size, shuffle=True, pin_memory=True, num_workers=4):
		return DataLoader(getattr(self,split),batch_size=batch_size, shuffle=shuffle,pin_memory=pin_memory,num_workers=num_workers)

	def get_data_loaders(self, batch_size, shuffle=True, pin_memory=True, num_workers=4):
		data_loaders = [self.get_data_loader(split=split,batch_size=batch_size,shuffle=shuffle,
			pin_memory=pin_memory,num_workers=num_workers) for split in self.splits]
		return data_loaders

class RawMNIST():

	def __init__(self,root='./',download=True,flatten=True):

		self.root = root

		trans = [ToTensor()]
		if flatten: trans.append(Flatten())

		self.train = UnsupervisedMNIST(root,train=True, transform=Compose(trans), download=download)
		self.test = UnsupervisedMNIST(root,train=False, transform=Compose(trans))

		self.splits = ['train','test']

	@property
	def num_splits(self):
		return len(self.splits)

	def get_data_loader(self,split, batch_size, shuffle=True, pin_memory=True, num_workers=4):
		return DataLoader(getattr(self,split),batch_size=batch_size, shuffle=shuffle,pin_memory=pin_memory,num_workers=num_workers)

	def get_data_loaders(self, batch_size, shuffle=True, pin_memory=True, num_workers=4):
		data_loaders = [self.get_data_loader(split=split,batch_size=batch_size,shuffle=shuffle,
			pin_memory=pin_memory,num_workers=num_workers) for split in self.splits]
		return data_loaders


class DequantizedMNIST():

	def __init__(self,root='./',download=True,flatten=True):

		self.root = root

		trans = [ToTensor(),Dequantize()]
		if flatten: trans.append(Flatten())

		self.train = UnsupervisedMNIST(root,train=True, transform=Compose(trans), download=download)
		self.test = UnsupervisedMNIST(root,train=False, transform=Compose(trans))

		self.splits = ['train','test']

	@property
	def num_splits(self):
		return len(self.splits)

	def get_data_loader(self,split, batch_size, shuffle=True, pin_memory=True, num_workers=4):
		return DataLoader(getattr(self,split),batch_size=batch_size, shuffle=shuffle,pin_memory=pin_memory,num_workers=num_workers)

	def get_data_loaders(self, batch_size, shuffle=True, pin_memory=True, num_workers=4):
		data_loaders = [self.get_data_loader(split=split,batch_size=batch_size,shuffle=shuffle,
			pin_memory=pin_memory,num_workers=num_workers) for split in self.splits]
		return data_loaders




# DataSets = DynamicallyBinarizedMNIST()
# train_loader, test_loader = DataSets.get_data_loaders(128)



################################ cifar10 dataset part


class UnsupervisedCIFAR10(CIFAR10):
	def __init__(self,root='./',train=True,transform=None,download=False):
		super(UnsupervisedCIFAR10,self).__init__(root,train=train,transform=transform,download=download)

	def __getitem__(self,index):

		return super(UnsupervisedCIFAR10,self).__getitem__(index)[0]


class VectorizedCIFAR10():
	def __init__(self,root='./',download=True,num_bits=8,pil_transforms=[]):

		self.root = root
		trans_train = pil_transforms + [ToTensor(),Cifar10Flatten()]
		trans_test = [ToTensor(),Cifar10Flatten()]

		self.train = UnsupervisedCIFAR10(root,train=True, transform=Compose(trans_train),download=download)
		self.test = UnsupervisedCIFAR10(root,train=False, transform=Compose(trans_test))

		self.splits = ['train','test']

	@property
	def num_splits(self):
		return len(self.splits)

	def get_data_loader(self,split, batch_size, shuffle=True, pin_memory=True, num_workers=4):
		return DataLoader(getattr(self,split),batch_size=batch_size, shuffle=shuffle,pin_memory=pin_memory,num_workers=num_workers)

	def get_data_loaders(self, batch_size, shuffle=True, pin_memory=True, num_workers=4):
		data_loaders = [self.get_data_loader(split=split,batch_size=batch_size,shuffle=shuffle,
			pin_memory=pin_memory,num_workers=num_workers) for split in self.splits]
		return data_loaders

class MyCIFAR10():
	def __init__(self,root='./',download=True,num_bits=8,pil_transforms=[]):

		self.root = root
		trans_train = pil_transforms + [ToTensor(),Quantize(num_bits)]
		trans_test = [ToTensor(), Quantize(num_bits)]

		self.train = UnsupervisedCIFAR10(root,train=True, transform=Compose(trans_train),download=download)
		self.test = UnsupervisedCIFAR10(root,train=False, transform=Compose(trans_test))

		self.splits = ['train','test']

	@property
	def num_splits(self):
		return len(self.splits)

	def get_data_loader(self,split, batch_size, shuffle=True, pin_memory=True, num_workers=4):
		return DataLoader(getattr(self,split),batch_size=batch_size, shuffle=shuffle,pin_memory=pin_memory,num_workers=num_workers)

	def get_data_loaders(self, batch_size, shuffle=True, pin_memory=True, num_workers=4):
		data_loaders = [self.get_data_loader(split=split,batch_size=batch_size,shuffle=shuffle,
			pin_memory=pin_memory,num_workers=num_workers) for split in self.splits]
		return data_loaders







# data = MyCIFAR10()
# train_loader, test_loader = data.get_data_loaders(32)
# imgTensor = next(iter(train_loader))




################################### plan dataset 

import numpy as np
import os
import torch

from collections.abc import Iterable
from skimage import color, io, transform
from sklearn.datasets import make_moons
from torch.utils.data import Dataset


class PlaneDataset(Dataset):
	def __init__(self, num_points, flip_axes=False,data_path=None):
		self.num_points = num_points
		self.flip_axes = flip_axes
		self.data = None

		if data_path is not None:
			self.data = torch.load(data_path)
		else:
			self.reset()

	def __getitem__(self, item):
		return self.data[item]

	def __len__(self):
		return self.num_points

	def reset(self):
		self._create_data()
		if self.flip_axes:
			x1 = self.data[:,0]
			x2 = self.data[:,1]
			self.data = torch.stack([x2,x1]).t()

	def _create_data(self):
		raise NotImplementedError


class GaussianDataset(PlaneDataset):

	def _create_data(self):
		x1 = torch.randn(self.num_points)
		x2 = 0.5*torch.randn(self.num_points)
		self.data = torch.stack((x1,x2)).t()


class CrescentDataset(PlaneDataset):

	def _create_data(self):
		x1 = torch.randn(self.num_points)
		x2_mean = 0.5*x1**2 -1
		x2_var = torch.exp(torch.Tensor([-2]))
		x2 = x2_mean + x2_var ** 0.5*torch.randn(self.num_points)
		self.data = torch.stack((x1,x2)).t()

class CrescentCubedDataset(PlaneDataset):
	def _create_data(self):
		x1 = torch.randn(self.num_points)
		x2_mean = 0.2*x1**3
		x2_var = torch.ones(x1.shape)
		x2 = x2_mean + x2_var ** 0.5 * torch.randn(self.num_points)
		self.data = torch.stack((x1,x2)).t()

class SineWaveDataset(PlaneDataset):
	def _create_data(self):
		x1 = torch.randn(self.num_points)
		x2_mean = torch.sin(5*x1)
		x2_var = torch.exp(-2*torch.ones(x1.shape))
		x2 = x2_mean + x2_var ** 0.5*torch.randn(self.num_points)
		self.data = torch.stack((x1,x2)).t()

class AbsDataset(PlaneDataset):
	def _create_data(self):
		x1 = torch.randn(self.num_points)
		x2_mean = torch.abs(x1)-1
		x2_var = torch.exp(-3*torch.ones(x1.shape))
		x2 = x2_mean + x2_var ** 0.5 * torch.randn(self.num_points)
		self.data = torch.stack((x1,x2)).t()

class SignDataset(PlaneDataset):
	def _create_data(self):
		x1 = torch.randn(self.num_points)
		x2_mean = torch.sign(x1) + x1
		x2_var = torch.exp(-3*torch.ones(x1.shape))
		x2 = x2_mean + x2_var**0.5*torch.randn(self.num_points)
		self.data = torch.stack((x1,x2)).t()

class FourCirclesDataset(PlaneDataset):
	def __init__(self,num_points, flip_axes=False):
		if num_points % 4 !=0:
			raise ValueError('Number of data points must be a multiple of four')
		super().__init__(num_points,flip_axes)

	@staticmethod
	def create_circle(num_per_circle, std=0.1):
		u = torch.rand(num_per_circle)
		x1 = torch.cos(2*np.pi*u)
		x2 = torch.sin(2*np.pi*u)
		data = 2*torch.stack((x1,x2)).t()
		data += std * torch.randn(data.shape)
		return data

	def _create_data(self):
		num_per_circle = self.num_points //4
		centers = [
		[-1,-1],
		[-1,1],
		[1,-1],
		[1,1]]

		self.data = torch.cat([self.create_circle(num_per_circle)-torch.Tensor(center) for center in centers])

class DiamondDataset(PlaneDataset):
	def __init__(self,num_points, flip_axes=False, width=20, bound=2.5, std = 0.04):

		self.width = width
		self.bound = bound
		self.std = std
		super().__init__(num_points, flip_axes)

	def _create_data(self, rotate=True):
		means = np.array([
			(x+1e-3 * np.random.rand(),y+1e-3 * np.random.rand())
			for x in np.linspace(-self.bound, self.bound, self.width)
			for y in np.linspace(-self.bound, self.bound, self.width)
			])

		covariance_factor = self.std * np.eye(2)

		index = np.random.choice(range(self.width**2),size=self.num_points, replace=True)
		noise = np.random.randn(self.num_points,2)
		self.data = means[index] + noise@covariance_factor
		if rotate:
			rotation_matrix = np.array([
				[1/np.sqrt(2), -1/np.sqrt(2)],
				[1/np.sqrt(2), 1/np.sqrt(2)]])
			self.data = self.data @ rotation_matrix

		self.data = self.data.astype(np.float32)
		self.data = torch.Tensor(self.data)

class TwoSpiralsDataset(PlaneDataset):
	def _create_data(self):
		n = torch.sqrt(torch.rand(self.num_points//2))* 540*(2*np.pi)/360
		d1x = -torch.cos(n)*n + torch.rand(self.num_points//2)*0.5
		d1y = torch.sin(n)*n + torch.rand(self.num_points//2)*0.5
		x = torch.cat([torch.stack([d1x,d1y]).t(),torch.stack([-d1x,-d1y]).t()])
		self.data = x/3 + torch.randn_like(x)*0.1

class TestGridDataset(PlaneDataset):
	def __init__(self, num_points_per_axis, bounds):
		self.num_points_per_axis = num_points_per_axis
		self.bounds = bounds
		self.shape = [num_points_per_axis] * 2
		self.X = None
		self.Y = None
		super().__init__(num_points = num_points_per_axis**2)

	def _create_data(self):
		x = np.linspace(self.bounds[0][0], self.bounds[0][1], self.num_points_per_axis)
		y = np.linspace(self.bounds[1][0], self.bounds[1][1], self.num_points_per_axis)
		self.X, self.Y = np.meshgrid(x,y)
		data_ = np.vstack([self.X.flatten(),self.Y.flatten()]).T
		self.data = torch.tensor(data_).float()

class CheckerboardDataset(PlaneDataset):
	def _create_data(self):
		x1 = torch.rand(self.num_points)*4-2
		x2_ = torch.rand(self.num_points) - torch.randint(0,2,[self.num_points]).float()*2
		x2 = x2_ + torch.floor(x1)%2
		self.data = torch.stack([x1,x2]).t() * 2

class TwoMoonsDataset(PlaneDataset):

	def _create_data(self):
		data = make_moons(n_samples=self.num_points, noise=0.1, random_state=0)[0]
		data = data.astype('float32')
		data = data*2 + np.array([-1, -0.2])
		self.data = torch.from_numpy(data).float()

class FaceDataset(PlaneDataset):

	def __init__(self, num_points, name='einstein', resize=[512,512], flip_axes = False):
		self.name = name
		self.image = None
		self.resize = resize if isinstance(resize, Iterable) else [resize, resize]
		super().__init__(num_points, flip_axes)

	def _create_data(self):
		root = './'
		path = os.path.join(root,'faces', self.name + '.jpg')
		try:
			image = io.imread(path)
		except FileNotFoundError:
			raise RuntimeError('Unknown face name: {}'.format(self.name))
		image = color.rgb2gray(image)
		self.image = transform.resize(image,self.resize)

		grid = np.array([
			(x,y) for x in range(self.image.shape[0]) for y in range(self.image.shape[1])])

		rotation_matrix = np.array([
			[0,-1],
			[1,0]])

		p = self.image.reshape(-1)/sum(self.image.reshape(-1))
		ix = np.random.choice(range(len(grid)), size=self.num_points, replace=True,p=p)
		points = grid[ix].astype(np.float32)
		points += np.random.rand(self.num_points, 2)
		points /= (self.image.shape[0])

		self.data = torch.tensor(points @ rotation_matrix).float()
		self.data[:,1] += 1

class Gaussian():

	def __init__(self,train_samples=100,test_samples=100):

		self.train = GaussianDataset(num_points=train_samples)
		self.test = GaussianDataset(num_points=test_samples)
		self.splits = ['train','test']

	@property
	def num_splits(self):
		return len(self.splits)

	def get_data_loader(self,split, batch_size, shuffle=True, pin_memory=True, num_workers=4):
		return DataLoader(getattr(self,split),batch_size=batch_size, shuffle=shuffle,pin_memory=pin_memory,num_workers=num_workers)

	def get_data_loaders(self, batch_size, shuffle=True, pin_memory=True, num_workers=4):
		data_loaders = [self.get_data_loader(split=split,batch_size=batch_size,shuffle=shuffle,
			pin_memory=pin_memory,num_workers=num_workers) for split in self.splits]
		return data_loaders

class Crescent():

	def __init__(self,train_samples=100,test_samples=100,train_data_path=None, test_data_path=None):

		self.train = CrescentDataset(num_points=train_samples,flip_axes=False,data_path=train_data_path)
		self.test = CrescentDataset(num_points=test_samples,flip_axes=False,data_path=test_data_path)
		self.splits = ['train','test']

	@property
	def num_splits(self):
		return len(self.splits)

	def get_data_loader(self,split, batch_size, shuffle=True, pin_memory=True, num_workers=4):
		return DataLoader(getattr(self,split),batch_size=batch_size, shuffle=shuffle,pin_memory=pin_memory,num_workers=num_workers)

	def get_data_loaders(self, batch_size, shuffle=True, pin_memory=True, num_workers=4):
		data_loaders = [self.get_data_loader(split=split,batch_size=batch_size,shuffle=shuffle,
			pin_memory=pin_memory,num_workers=num_workers) for split in self.splits]
		return data_loaders

class CrescentCubed():

	def __init__(self,train_samples=100,test_samples=100):

		self.train = CrescentCubedDataset(num_points=train_samples)
		self.test = CrescentCubedDataset(num_points=test_samples)
		self.splits = ['train','test']

	@property
	def num_splits(self):
		return len(self.splits)

	def get_data_loader(self,split, batch_size, shuffle=True, pin_memory=True, num_workers=4):
		return DataLoader(getattr(self,split),batch_size=batch_size, shuffle=shuffle,pin_memory=pin_memory,num_workers=num_workers)

	def get_data_loaders(self, batch_size, shuffle=True, pin_memory=True, num_workers=4):
		data_loaders = [self.get_data_loader(split=split,batch_size=batch_size,shuffle=shuffle,
			pin_memory=pin_memory,num_workers=num_workers) for split in self.splits]
		return data_loaders

class SineWave():

	def __init__(self,train_samples=100,test_samples=100):

		self.train = SineWaveDataset(num_points=train_samples)
		self.test = SineWaveDataset(num_points=test_samples)
		self.splits = ['train','test']

	@property
	def num_splits(self):
		return len(self.splits)

	def get_data_loader(self,split, batch_size, shuffle=True, pin_memory=True, num_workers=4):
		return DataLoader(getattr(self,split),batch_size=batch_size, shuffle=shuffle,pin_memory=pin_memory,num_workers=num_workers)

	def get_data_loaders(self, batch_size, shuffle=True, pin_memory=True, num_workers=4):
		data_loaders = [self.get_data_loader(split=split,batch_size=batch_size,shuffle=shuffle,
			pin_memory=pin_memory,num_workers=num_workers) for split in self.splits]
		return data_loaders

class Abs():

	def __init__(self,train_samples=100,test_samples=100):

		self.train = AbsDataset(num_points=train_samples)
		self.test = AbsDataset(num_points=test_samples)
		self.splits = ['train','test']

	@property
	def num_splits(self):
		return len(self.splits)

	def get_data_loader(self,split, batch_size, shuffle=True, pin_memory=True, num_workers=4):
		return DataLoader(getattr(self,split),batch_size=batch_size, shuffle=shuffle,pin_memory=pin_memory,num_workers=num_workers)

	def get_data_loaders(self, batch_size, shuffle=True, pin_memory=True, num_workers=4):
		data_loaders = [self.get_data_loader(split=split,batch_size=batch_size,shuffle=shuffle,
			pin_memory=pin_memory,num_workers=num_workers) for split in self.splits]
		return data_loaders

class Sign():

	def __init__(self,train_samples=100,test_samples=100):

		self.train = SignDataset(num_points=train_samples)
		self.test = SignDataset(num_points=test_samples)
		self.splits = ['train','test']

	@property
	def num_splits(self):
		return len(self.splits)

	def get_data_loader(self,split, batch_size, shuffle=True, pin_memory=True, num_workers=4):
		return DataLoader(getattr(self,split),batch_size=batch_size, shuffle=shuffle,pin_memory=pin_memory,num_workers=num_workers)

	def get_data_loaders(self, batch_size, shuffle=True, pin_memory=True, num_workers=4):
		data_loaders = [self.get_data_loader(split=split,batch_size=batch_size,shuffle=shuffle,
			pin_memory=pin_memory,num_workers=num_workers) for split in self.splits]
		return data_loaders

class FourCircles():

	def __init__(self,train_samples=100,test_samples=100):

		self.train = FourCirclesDataset(num_points=train_samples)
		self.test = FourCirclesDataset(num_points=test_samples)
		self.splits = ['train','test']

	@property
	def num_splits(self):
		return len(self.splits)

	def get_data_loader(self,split, batch_size, shuffle=True, pin_memory=True, num_workers=4):
		return DataLoader(getattr(self,split),batch_size=batch_size, shuffle=shuffle,pin_memory=pin_memory,num_workers=num_workers)

	def get_data_loaders(self, batch_size, shuffle=True, pin_memory=True, num_workers=4):
		data_loaders = [self.get_data_loader(split=split,batch_size=batch_size,shuffle=shuffle,
			pin_memory=pin_memory,num_workers=num_workers) for split in self.splits]
		return data_loaders

class Diamond():

	def __init__(self,train_samples=100,test_samples=100):

		self.train = DiamondDataset(num_points=train_samples)
		self.test = DiamondDataset(num_points=test_samples)
		self.splits = ['train','test']

	@property
	def num_splits(self):
		return len(self.splits)

	def get_data_loader(self,split, batch_size, shuffle=True, pin_memory=True, num_workers=4):
		return DataLoader(getattr(self,split),batch_size=batch_size, shuffle=shuffle,pin_memory=pin_memory,num_workers=num_workers)

	def get_data_loaders(self, batch_size, shuffle=True, pin_memory=True, num_workers=4):
		data_loaders = [self.get_data_loader(split=split,batch_size=batch_size,shuffle=shuffle,
			pin_memory=pin_memory,num_workers=num_workers) for split in self.splits]
		return data_loaders

class TwoSpirals():

	def __init__(self,train_samples=100,test_samples=100):

		self.train = TwoSpiralsDataset(num_points=train_samples)
		self.test = TwoSpiralsDataset(num_points=test_samples)
		self.splits = ['train','test']

	@property
	def num_splits(self):
		return len(self.splits)

	def get_data_loader(self,split, batch_size, shuffle=True, pin_memory=True, num_workers=4):
		return DataLoader(getattr(self,split),batch_size=batch_size, shuffle=shuffle,pin_memory=pin_memory,num_workers=num_workers)

	def get_data_loaders(self, batch_size, shuffle=True, pin_memory=True, num_workers=4):
		data_loaders = [self.get_data_loader(split=split,batch_size=batch_size,shuffle=shuffle,
			pin_memory=pin_memory,num_workers=num_workers) for split in self.splits]
		return data_loaders

class TwoMoons():

	def __init__(self,train_samples=100,test_samples=100):

		self.train = TwoMoonsDataset(num_points=train_samples)
		self.test = TwoMoonsDataset(num_points=test_samples)
		self.splits = ['train','test']

	@property
	def num_splits(self):
		return len(self.splits)

	def get_data_loader(self,split, batch_size, shuffle=True, pin_memory=True, num_workers=4):
		return DataLoader(getattr(self,split),batch_size=batch_size, shuffle=shuffle,pin_memory=pin_memory,num_workers=num_workers)

	def get_data_loaders(self, batch_size, shuffle=True, pin_memory=True, num_workers=4):
		data_loaders = [self.get_data_loader(split=split,batch_size=batch_size,shuffle=shuffle,
			pin_memory=pin_memory,num_workers=num_workers) for split in self.splits]
		return data_loaders

class Checkerboard():

	def __init__(self,train_samples=100,test_samples=100):

		self.train = CheckerboardDataset(num_points=train_samples)
		self.test = CheckerboardDataset(num_points=test_samples)
		self.splits = ['train','test']

	@property
	def num_splits(self):
		return len(self.splits)

	def get_data_loader(self,split, batch_size, shuffle=True, pin_memory=True, num_workers=4):
		return DataLoader(getattr(self,split),batch_size=batch_size, shuffle=shuffle,pin_memory=pin_memory,num_workers=num_workers)

	def get_data_loaders(self, batch_size, shuffle=True, pin_memory=True, num_workers=4):
		data_loaders = [self.get_data_loader(split=split,batch_size=batch_size,shuffle=shuffle,
			pin_memory=pin_memory,num_workers=num_workers) for split in self.splits]
		return data_loaders

class Face():

	def __init__(self,train_samples=100,test_samples=100):

		self.train = FaceDataset(num_points=train_samples)
		self.test = FaceDataset(num_points=test_samples)
		self.splits = ['train','test']

	@property
	def num_splits(self):
		return len(self.splits)

	def get_data_loader(self,split, batch_size, shuffle=True, pin_memory=True, num_workers=4):
		return DataLoader(getattr(self,split),batch_size=batch_size, shuffle=shuffle,pin_memory=pin_memory,num_workers=num_workers)

	def get_data_loaders(self, batch_size, shuffle=True, pin_memory=True, num_workers=4):
		data_loaders = [self.get_data_loader(split=split,batch_size=batch_size,shuffle=shuffle,
			pin_memory=pin_memory,num_workers=num_workers) for split in self.splits]
		return data_loaders





# DataSets = Face(train_samples=5000, test_samples=5000)
# train_loader, test_loader = DataSets.get_data_loaders(128)


# imgTensor = next(iter(train_loader))
# x = imgTensor


# plt.scatter(DataSets.train.data[:,0],DataSets.train.data[:,1]);plt.show()



### spatial mnist data


class SpatialMNISTDataset(data.Dataset):

	def __init__(self, data_dir = './spatial_mnist', split='train'):
		
		splits = {
		'train':slice(0,50000),
		'valid':slice(50000,60000),
		'test':slice(60000,70000)
		}

		spatial_path = os.path.join(data_dir, 'spatial.pkl')
		with open(spatial_path,'rb') as file:
			spatial = pickle.load(file)

		labels_path = os.path.join(data_dir, 'labels.pkl')
		with open(labels_path, 'rb') as file:
			labels = pickle.load(file)

		self._spatial = np.array(spatial[splits[split]]).astype(np.float32)
		self._labels = np.array(labels[splits[split]])

		assert len(self._spatial) == len(self._labels)
		self._n = len(self._spatial)

	def __getitem__(self, item):
		return self._spatial[item]

	def __len__(self):
		return self._n



class DataContainer():
	def __init__(self, train, valid, test):
		self.train = train
		self.valid = valid 
		self.test = test


dataset_choices = {'spatial_mnist'}

def get_data(args):
	assert args.dataset in dataset_choices


	if args.dataset == 'spatial_mnist':
		dataset = DataContainer(SpatialMNISTDataset(os.path.join('./','spatial_mnist'),split='train'),
			SpatialMNISTDataset(os.path.join('./','spatial_mnist'),split='valid'),
			SpatialMNISTDataset(os.path.join('./','spatial_mnist'),split='test'))

	train_loader = DataLoader(dataset.train, batch_size=args.batch_size, shuffle=True)
	valid_loader = DataLoader(dataset.valid, batch_size=args.batch_size, shuffle=False)
	test_loader = DataLoader(dataset.test, batch_size=args.batch_size, shuffle=False)

	return train_loader, valid_loader, test_loader


from prettytable import PrettyTable


def get_args_table(args_dict):
	table = PrettyTable(['Arg','Value'])
	for arg, val in args_dict.item():
		table.add_row([arg,val])

	return table
