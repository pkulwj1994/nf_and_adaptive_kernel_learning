from activations import *
from datasets import *
from transforms import *
from utils import *

import torch
from torchvision.transforms import Compose, ToTensor
import torch.nn as nn
import torch.nn.functional as F
from collections.abc import Iterable
from functools import reduce
from operator import mul

from torch.distributions import Normal
from torch.distributions import Bernoulli




from torch.optim import Adam

import seaborn as sns
import time
import gc


device = 'cuda' if torch.cuda.is_available() else 'cpu'


from os import mkdir
from torch.utils import tensorboard
tb_dir = os.path.join('./', "tensorboard")
asset_dir = os.path.join('./','assets')
if os.path.exists(tb_dir):
	pass
else:
	mkdir(tb_dir)

if os.path.exists(asset_dir):
	pass
else:
	mkdir(asset_dir)



def compute_flow_s2(x):
	grad2, _ = compute_flow_score_and_hess(x)
	return torch.sum(grad2**2,-1)


flow_model = make_flow()
flow_model.load_state_dict(torch.load(os.path.join(asset_dir,'pretrained_banana_flow.pth')))

DataSets = Crescent(train_samples=100000, test_samples=50000,train_data_path='./crescent_train.pt', test_data_path='./crescent_test.pt')
train_loader, test_loader = DataSets.get_data_loaders(5000)



s2_const = 0.0
for i,x in enumerate(train_loader):
	s2_const += compute_flow_s2(x).sum().detach().cpu()

print(s2_const/100000)









