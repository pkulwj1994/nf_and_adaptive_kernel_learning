import numpy as np 
import banana_utils
from banana_utils import *


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


def calc_energy_score(x):
	x = x.to(device)
	x.requires_grad_(True)
	logp = e_model(x).sum()
	grad1 = grad(logp, x, create_graph=True)[0]
	return grad1

def one_hmc_iter(x,epsilon,M):
	r = torch.randn(x.shape).to(device)*torch.sqrt(M)

	r_ = r + 0.5*epsilon*calc_energy_score(x)
	xx = x + epsilon*M*r_
	rr = r_ + 0.5*epsilon*calc_energy_score(xx)

	u = torch.rand(x.shape[0]).to(device)

	jump = (1+ torch.sign(torch.exp(-e_model(x).squeeze() + 0.5*M*torch.sum(r*r,-1)+e_model(xx).squeeze() - 0.5*M*torch.sum(rr*rr,-1))-u))/2
	return jump.unsqueeze(-1)*xx + (1-jump.unsqueeze(-1))*x


def _one_hmc_iter(x,epsilon,M):
	r = torch.randn(x.shape).to(device)*torch.sqrt(M)

	r_ = r + 0.5*epsilon*(-x)
	xx = x + epsilon*M*r_
	rr = r_ + 0.5*epsilon*(-x)

	u = torch.rand(x.shape[0]).to(device)

	jump = (1+ torch.sign(torch.exp(-e_model(x).squeeze() + 0.5*M*torch.sum(r*r,-1)+e_model(xx).squeeze() - 0.5*M*torch.sum(rr*rr,-1))-u))/2
	return jump.unsqueeze(-1)*xx + (1-jump.unsqueeze(-1))*x


def l_hmc_iter(x,epsilon,M, L):
	xx = x
	for i in range(L):
		xx = one_hmc_iter(x,epsilon,M)
	return xx

def e_model_hmc_sample(batch_size):
	epsilon = torch.tensor([0.1]).to(device)
	M = torch.tensor([1.0]).to(device)
	L = 10

	x = torch.randn((batch_size,2)).to(device)
	for i in range(100):
		x = l_hmc_iter(x,epsilon,M,L)

	return x


def _e_model(x):
	x = x.to(device)
	return -0.5*torch.sum(x*x,-1)




DataSets = Crescent(train_samples=10000, test_samples=5000)
train_loader, test_loader = DataSets.get_data_loaders(128)

flow_model = make_flow()
flow_model.load_state_dict(torch.load(os.path.join(asset_dir,'pretrained_banana_flow.pth')))
# flow_model.eval()

e_model = energy_net(2).to(device)
e_model.load_state_dict(torch.load(os.path.join(asset_dir,'{}_banana_energy_model_try_{}_epoch_{}.pth'.format('ISM_TRAIN',0,799))))




import gc


TRAIN_NAME = 'ISM_TRAIN'


vecs = next(iter(test_loader))[:128]
flow_samples = flow_model.sample(128).cpu()
energy_samples = e_model_hmc_sample(1000).detach().cpu()


fig = plt.figure()
plt.scatter(vecs[:,0],vecs[:,1],s=1,label='data')
plt.legend()
plt.scatter(flow_samples[:,0],flow_samples[:,1],s=1,label='flow model')
plt.legend()
plt.scatter(energy_samples[:,0],energy_samples[:,1],s=1,label='energy model')
plt.legend()
scatter_fig = fig.get_figure()
scatter_fig.savefig(os.path.join(asset_dir,'./banana_data_flow_and_energy.png'), dpi = 400)




fig = plt.figure()
plt.scatter(energy_samples[:,0],energy_samples[:,1],s=1,label='energy model')
plt.legend()
scatter_fig = fig.get_figure()
scatter_fig.savefig(os.path.join(asset_dir,'./banana_data_flow_and_energy.png'), dpi = 400)
