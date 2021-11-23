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


device = 'cuda' if torch.cuda.is_available() else 'cpu'




def coupling_net(input_dim):
	return MLP(int(input_dim/2), input_dim,hidden_units=[128,64,32,16],
	                            activation='elu',
	                            in_lambda=None)

def make_flow():
	return Flow(base_dist=StandardNormal((2,)),
			transforms=[
			AffineCouplingBijection1d(coupling_net(2)),
			SwitchBijection1d(),AffineCouplingBijection1d(coupling_net(2)),
			SwitchBijection1d(),AffineCouplingBijection1d(coupling_net(2)),
			SwitchBijection1d(),AffineCouplingBijection1d(coupling_net(2)),
			SwitchBijection1d(),AffineCouplingBijection1d(coupling_net(2))]).to(device)



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

### 
###      Train Flow
### 

# DataSets = Crescent(train_samples=10000, test_samples=5000)
# train_loader, test_loader = DataSets.get_data_loaders(128)
# writer = tensorboard.SummaryWriter(tb_dir)

# epoch = 0

# model = make_flow()

# optimizer = Adam(model.parameters(), lr=1e-6)

# for epoch in range(epoch,200):
# 	l = 0.0
# 	for i,x in enumerate(train_loader):
# 		optimizer.zero_grad()
# 		loss = -model.log_prob(x.to(device)).mean()
# 		loss.backward()
# 		optimizer.step()
# 		l += loss.detach().cpu().item()
# 		print('Epoch: {}/{}, Iter: {}/{}, Nats: {:.3f}'.format(epoch+1,20,i+1,len(train_loader),l/(i+1)),end='\r')
# 	writer.add_scalar("flow_avg_nll", l/(i+1), epoch)
# 	torch.save(model.state_dict(),os.path.join(asset_dir,'pretrained_banana_flow.pth'))
# 	print('')


# vecs = next(iter(test_loader))[:128]
# samples = model.sample(128).cpu()

# fig = plt.figure()
# sns.kdeplot(vecs[:,0],vecs[:,1],shade=True,color='red')
# scatter_fig = fig.get_figure()
# scatter_fig.savefig(os.path.join(asset_dir,'./banana_data_density.png'), dpi = 400)

# fig = plt.figure()
# sns.kdeplot(samples[:,0],samples[:,1],shade=True)
# scatter_fig = fig.get_figure()
# scatter_fig.savefig(os.path.join(asset_dir,'./banana_model_density.png'), dpi = 400)

# fig = plt.figure()
# plt.scatter(vecs[:,0],vecs[:,1],s=1,label='data')
# plt.legend()
# plt.scatter(samples[:,0],samples[:,1],s=1,label='model')
# plt.legend()
# scatter_fig = fig.get_figure()
# scatter_fig.savefig(os.path.join(asset_dir,'./banana_data_and_model.png'), dpi = 400)



### score matching
from torch.autograd.functional import jacobian as jcb, hessian as hess
from torch.autograd import grad 
from torch.autograd import Variable


def compute_implicit_score_diff(x):
	x = x.to(device)
	x.requires_grad_(True)
	logp = e_model(x).sum()
	grad1 = grad(logp, x, create_graph=True)[0]

	hess1 = torch.stack([grad(grad1[:, i].sum(),x, create_graph=True, retain_graph=True)[0] for i in range(x.shape[-1])] ,-1)
	return 0.5*torch.einsum('bi,bj->bij',grad1,grad1) + hess1

def calc_ism_loss(x):

	# D = compute_batch_D(x)
	imp_mat_diff = compute_implicit_score_diff(x)

	return (imp_mat_diff).sum()/x.shape[0]

# def calc_idsm_loss_old(x):

# 	def logprob_func(x):
# 		with torch.no_grad():
# 			return model.log_prob(x.to(device))

# 	def jcb2_hess(x):
# 		return 1/2*jcb(logprob_func, xi.unsqueeze(0)).squeeze(-2).T@jcb(logprob_func, xi.unsqueeze(0)).squeeze(-2) - hess(logprob_func, xi.unsqueeze(0)).squeeze(-2)

# 	D = compute_batch_D(x)
# 	imp_mat_diff = compute_implicit_score_diff(x)

# 	return (imp_mat_diff*D).sum()/x.shape[0]


def calc_idsm_loss(x):

	x = x.to(device)
	x.requires_grad_(True)
	logp = flow_model.log_prob(x).sum()
	grad1 = grad(logp, x, create_graph=True)[0]
	hess1 = torch.stack([grad(grad1[:, i].sum(),x, create_graph=True, retain_graph=True)[0] for i in range(x.shape[-1])] ,-1)

	D = 0.5*torch.einsum('bi,bj->bij',grad1,grad1) + hess1
	imp_mat_diff = compute_implicit_score_diff(x)
	return (imp_mat_diff*D).sum()/x.shape[0]

def calc_true_idsm_loss(x):
	grad1 = torch.stack([-0.5*torch_e()*x[:,0]**3 + torch_e()**2*x[:,1]*x[:,0] + (torch_e()**2 - 1)*x[:,0],
		-torch_e()**2*x[:,1] + 0.5*torch_e()**2*x[:,0]**2 - torch_e()**2],-1)

	hess11 = -1.5*torch_e()**2*x[:,0]**2 + (torch_e()**2+1)*x[:,1]+torch_e()**2 -1
	hess12 = torch_e()**2*x[:,0]
	hess21 = torch_e()**2*x[:,0]
	hess22 = -torch_e()**2 + 0.0*x[:,0]

	hess1 = torch.stack([hess11,hess12,hess21,hess22],-1).reshape(x.shape[0],2,2)

	D = (0.5*torch.einsum('bi,bj->bij',grad1,grad1) + hess1).to(device)
	imp_mat_diff = compute_implicit_score_diff(x)
	return (imp_mat_diff*D).sum()/x.shape[0]


def calc_flow_score(x):
	x = x.to(device)
	x.requires_grad_(True)
	logp = flow_model.log_prob(x).sum()
	grad1 = grad(logp, x, create_graph=True)[0]
	return grad1

def calc_true_score(x):
	grad1 = torch.stack([-0.5*torch_e()*x[:,0]**3 + torch_e()**2*x[:,1]*x[:,0] + (torch_e()**2 - 1)*x[:,0],
		-torch_e()**2*x[:,1] + 0.5*torch_e()**2*x[:,0]**2 - torch_e()**2],-1)
	return grad1



def torch_e():
	return torch.exp(torch.tensor([1.]))



def true_energy_func(x):
	return -0.5*x[:,0]**2 - 0.5*torch_e()**2*(x[:,1]-0.5*x[:,0]**2 + 1)**2



from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

import numpy as np
from matplotlib import cm


def banana_energy_plot(epoch=0):

	L_BOX = -5
	R_BOX = 5
	KNOTS = 250

	fig=plt.figure()
	ax=fig.add_subplot(111,projection='3d')


	u=np.linspace(L_BOX,R_BOX,KNOTS)
	xx,yy=np.meshgrid(u,u)

	x_in =  (torch.stack([torch.tensor(xx).flatten(), torch.tensor(yy).flatten()],-1) + torch.tensor([0,-1]))
	with torch.no_grad():
		zz = true_energy_func(x_in).reshape(KNOTS,KNOTS)

		x_in = x_in.to(device).to(torch.float32)

		z= e_model(x_in).reshape(KNOTS,KNOTS)

		ax.plot_surface(x_in[:,0].reshape(KNOTS,KNOTS).to('cpu').detach().numpy(),x_in[:,1].reshape(KNOTS,KNOTS).to('cpu').detach().numpy(),z.cpu().detach().numpy(),rstride=4,cstride=4,cmap=cm.YlGnBu_r)


		ax.plot_surface(x_in[:,0].reshape(KNOTS,KNOTS).to('cpu').detach().numpy(),x_in[:,1].reshape(KNOTS,KNOTS).to('cpu').detach().numpy(),zz.numpy(),rstride=4,cstride=4,cmap=cm.coolwarm)

	d3_fig = fig.get_figure()
	d3_fig.savefig(os.path.join(asset_dir,'./banna_energy_3d_epoch_{}.png'.format(epoch)), dpi = 400)
	plt.close()

	return None

def banana_score_plot(epoch=0):

	L_BOX = -10
	R_BOX = 10
	KNOTS = 100

	fig=plt.figure()
	ax=fig.add_subplot(111)


	u=np.linspace(L_BOX,R_BOX,KNOTS)
	xx,yy=np.meshgrid(u,u)

	x_in =  (torch.stack([torch.tensor(xx).flatten(), torch.tensor(yy).flatten()],-1) + torch.tensor([0,-1]))
	zz1,zz2 = calc_true_score(x_in).chunk(2,-1)

	zz1,zz2 = zz1.reshape(KNOTS,KNOTS),zz2.reshape(KNOTS,KNOTS)

	x_in = x_in.to(device).to(torch.float32)

	z1,z2 = calc_flow_score(x_in).chunk(2,-1)
	z1,z2 = z1.reshape(KNOTS,KNOTS),z2.reshape(KNOTS,KNOTS)

	ax.quiver(x_in[:,0].reshape(KNOTS,KNOTS).to('cpu').detach().numpy(),x_in[:,1].reshape(KNOTS,KNOTS).to('cpu').detach().numpy(),z1.cpu().detach().numpy()-zz1.numpy(),z2.cpu().detach().numpy()-zz2.numpy())

	d3_fig = fig.get_figure()
	d3_fig.savefig(os.path.join(asset_dir,'./banna_score_epoch_{}.png'.format(epoch)), dpi = 400)
	plt.close()

	return None




def energy_net(input_dim):
	return MLP(int(input_dim), 1,hidden_units=[512,256,128],
                                activation='elu',
                                in_lambda=None)


flow_model = make_flow()
flow_model.load_state_dict(torch.load(os.path.join(asset_dir,'pretrained_banana_flow.pth')))
# flow_model.eval()

DataSets = Crescent(train_samples=10000, test_samples=5000)
train_loader, test_loader = DataSets.get_data_loaders(128)




current_tb_dir = os.path.join(tb_dir,time.strftime('%Y-%m-%d_%H:%M:%S',time.localtime(time.time())))
mkdir(current_tb_dir)

writer = tensorboard.SummaryWriter(current_tb_dir)



TRAIN_NAME = 'ISM_TRAIN'


ism_losses = []
idsm_losses = []
tidsm_losses = []

ism_eval_losses = []
idsm_eval_losses = []
tidsm_eval_losses = []


for tryy in range(1):

	ism_loss = []
	idsm_loss = []
	tidsm_loss = []

	ism_eval_loss = []
	idsm_eval_loss = []
	tidsm_eval_loss = []

	e_model = energy_net(2).to(device)

	epoch = 0
	optimizer = Adam(e_model.parameters(), lr=1e-5)

	for epoch in range(epoch,800):
		l = 0.0
		ll = 0.0
		lll = 0.0
		for i,x in enumerate(train_loader):
			l += calc_ism_loss(x).detach().cpu().item()
			ll += calc_idsm_loss(x).detach().cpu().item()
			lll += calc_true_idsm_loss(x).detach().cpu().item()			

			optimizer.zero_grad()

			loss = calc_ism_loss(x)

			loss.backward()

			optimizer.step()
			optimizer.zero_grad()

			print('ism Epoch: {}/{}, Iter: {}/{}, AvgLoss: {:.3f}'.format(epoch+1,20,i+1,len(train_loader),l/(i+1)),end='\r')
		ism_loss.append(l/(i+1))
		idsm_loss.append(ll/(i+1))
		tidsm_loss.append(lll/(i+1))

		if tryy == 0:
			writer.add_scalar("ism_loss_{}".format(TRAIN_NAME), l/(i+1), epoch)
			writer.add_scalar("idsm_loss_{}".format(TRAIN_NAME), ll/(i+1), epoch)
			writer.add_scalar("tidsm_loss_{}".format(TRAIN_NAME), lll/(i+1), epoch)

		print('')

		if epoch %10 ==0:

			l = 0.0
			ll = 0.0
			lll = 0.0
			for i,x in enumerate(test_loader):
				l += calc_ism_loss(x).detach().cpu().item()
				ll += calc_idsm_loss(x).detach().cpu().item()
				lll+= calc_true_idsm_loss(x).detach().cpu().item()
			ism_eval_loss.append(l/(i+1))
			idsm_eval_loss.append(ll/(i+1))
			tidsm_eval_loss.append(lll/(i+1))
			if tryy == 0:
				writer.add_scalar("ism_eval_loss_{}".format(TRAIN_NAME), l/(i+1), epoch)
				writer.add_scalar("idsm_eval_loss_{}".format(TRAIN_NAME), ll/(i+1), epoch)
				writer.add_scalar("tidsm_eval_loss_{}".format(TRAIN_NAME), lll/(i+1), epoch)
			banana_energy_plot(epoch)
			banana_score_plot(epoch)

	torch.save(e_model.state_dict(),os.path.join(asset_dir,'{}_banana_energy_model_try_{}_epoch_{}.pth'.format(TRAIN_NAME,tryy,epoch)))


	ism_losses.append(np.array(ism_loss))
	idsm_losses.append(np.array(idsm_loss))
	tidsm_losses.append(np.array(tidsm_losses))

	ism_eval_losses.append(np.array(ism_eval_loss))
	idsm_eval_losses.append(np.array(idsm_eval_loss))
	tidsm_eval_losses.append(np.array(tidsm_eval_loss))

	np.save(os.path.join(asset_dir,'{}_ism_losses.npy'.format(TRAIN_NAME)),np.array(ism_losses))
	np.save(os.path.join(asset_dir,'{}_idsm_losses.npy'.format(TRAIN_NAME)),np.array(idsm_loss))
	np.save(os.path.join(asset_dir,'{}_tidsm_losses.npy'.format(TRAIN_NAME)),np.array(tidsm_losses))

	np.save(os.path.join(asset_dir,'{}_ism_eval_losses_per10_epochs.npy'.format(TRAIN_NAME)),np.array(ism_eval_loss))
	np.save(os.path.join(asset_dir,'{}_idsm_eval_losses_per10_epochs.npy'.format(TRAIN_NAME)),np.array(idsm_eval_loss))
	np.save(os.path.join(asset_dir,'{}_tidsm_eval_losses_per10_epochs.npy'.format(TRAIN_NAME)),np.array(tidsm_eval_loss))

























