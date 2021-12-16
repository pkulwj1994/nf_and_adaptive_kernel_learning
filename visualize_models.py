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




TRAIN_NAME = 'ISM_000_SELF_DIAGSST'
# tries = [10,11,12,13,14,15,16,17,18,19]
tries = [0]
ALPHA = torch.tensor([0.00]).to(device)
S2_CONST = torch.tensor([15.8844]).to(device)
# ft_e_model_path = './assets/ISM_000_SELF_DIAGSST_banana_energy_model_try_0_epoch_350.pth'
# ft_e_model_path = None

# DataSets = Crescent(train_samples=100000, test_samples=50000,train_data_path='./crescent_train.pt', test_data_path='./crescent_test.pt')
# train_loader, test_loader = DataSets.get_data_loaders(5000)



e_model = energy_net(2).to(device)
for epoch in 50*np.arange(29,33):
	e_model.load_state_dict(torch.load('./assets/ISM_000_SELF_DIAGSST_banana_energy_model_try_0_epoch_{}.pth'.format(epoch)))

	mc_model_z = estimate_norm_constant_sampling_accurate(lambda x: e_model(x) - e_model(torch.tensor([[0.0,-1.0]]).to(device)),est_rounds=100)
	mc_flow_z = torch.tensor([1.0000]).to(device)
	mc_true_z = torch.tensor([2.3114562546904662]).to(device)

	banana_energy_plot(TRAIN_NAME,epoch,mc_model_z.cpu(),mc_true_z.cpu())
	banana_prob_plot(TRAIN_NAME,epoch,mc_model_z.cpu(),mc_true_z.cpu())
	banana_score_norm_3D_plot(TRAIN_NAME,epoch)
	banana_score_plot(TRAIN_NAME,epoch)

	gc.collect()

