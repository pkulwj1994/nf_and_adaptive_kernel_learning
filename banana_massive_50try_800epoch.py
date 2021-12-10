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

def compute_model_score_and_hess(x):
	x = x.to(device)
	x.requires_grad_(True)
	logp = e_model(x).sum()
	grad1 = grad(logp, x, create_graph=True)[0]

	hess1 = torch.stack([grad(grad1[:, i].sum(),x, create_graph=True, retain_graph=True)[0] for i in range(x.shape[-1])] ,-1)
	return grad1, hess1

def compute_flow_score_and_hess(x):
	x = x.to(device)
	x.requires_grad_(True)
	logp = flow_model.log_prob(x).sum()
	grad1 = grad(logp, x, create_graph=True)[0]
	hess1 = torch.stack([grad(grad1[:, i].sum(),x, create_graph=True, retain_graph=True)[0] for i in range(x.shape[-1])] ,-1)
	return grad1, hess1

def compute_true_score_and_hess(x):
	x = x.to(device)
	grad1 = torch.stack([-0.5*torch_e()**2*x[:,0]**3 + torch_e()**2*x[:,1]*x[:,0] + (torch_e()**2 - 1)*x[:,0],
		-torch_e()**2*x[:,1] + 0.5*torch_e()**2*x[:,0]**2 - torch_e()**2],-1)

	hess11 = -1.5*torch_e()**2*x[:,0]**2 + (torch_e()**2+1)*x[:,1]+torch_e()**2 -1
	hess12 = torch_e()**2*x[:,0]
	hess21 = torch_e()**2*x[:,0]
	hess22 = -torch_e()**2 + 0.0*x[:,0]

	hess1 = torch.stack([hess11,hess12,hess21,hess22],-1).reshape(x.shape[0],2,2).detach()

	return grad1.detach(),hess1.detach()


def calc_ism_loss_old(x):

	# D = compute_batch_D(x)
	imp_mat_diff = compute_implicit_score_diff(x)

	return (imp_mat_diff*torch.eye(2).to(device)).sum()/x.shape[0]

def calc_ism_loss(x):
	grad1,hess1 = compute_model_score_and_hess(x)

	return (0.5*(grad1*grad1).sum() + torch.einsum('bii->b',hess1).sum())/x.shape[0]

	# torch.diagonal(hess1, dim1=-2,dim2=-1).sum(-1)

# def calc_idsm_loss_old(x):

# 	def logprob_func(x):
# 		with torch.no_grad():
# 			return model.log_prob(x.to(device))

# 	def jcb2_hess(x):
# 		return 1/2*jcb(logprob_func, xi.unsqueeze(0)).squeeze(-2).T@jcb(logprob_func, xi.unsqueeze(0)).squeeze(-2) - hess(logprob_func, xi.unsqueeze(0)).squeeze(-2)

# 	D = compute_batch_D(x)
# 	imp_mat_diff = compute_implicit_score_diff(x)

# 	return (imp_mat_diff*D).sum()/x.shape[0]


def calc_idsm_loss_old_wrong(x):

	x = x.to(device)
	x.requires_grad_(True)
	logp = flow_model.log_prob(x).sum()
	grad1 = grad(logp, x, create_graph=True)[0]
	hess1 = torch.stack([grad(grad1[:, i].sum(),x, create_graph=True, retain_graph=True)[0] for i in range(x.shape[-1])] ,-1)

	D = 0.5*torch.einsum('bi,bj->bij',grad1,grad1) + hess1
	imp_mat_diff = compute_implicit_score_diff(x)
	return (imp_mat_diff*D).sum()/x.shape[0]

def calc_idsm_loss_diagsst(x):

	grad1, hess1 = compute_model_score_and_hess(x)
	grad2, hess2 = compute_flow_score_and_hess(x)

	D = 0.5*torch.einsum('bi,bj->bij',grad2,grad2)*torch.eye(2).to(device)

	loss1 = ((0.5*torch.einsum('bi,bj->bij',grad1,grad1) + hess1)*D).sum()
	loss2 = (torch.einsum('bi,bj->bij',grad1,grad2)*(hess2*torch.eye(2).to(device))).sum()

	return (loss1+loss2)/x.shape[0]

def calc_idsm_loss_diagsst_diaghess(x):

	grad1, hess1 = compute_model_score_and_hess(x)
	x = x.to(device)
	x.requires_grad_(True)
	logp = flow_model.log_prob(x).sum()
	grad2 = grad(logp, x, create_graph=True)[0]
	hess2 = torch.stack([grad(grad2[:, i].sum(),x, create_graph=True, retain_graph=True)[0] for i in range(x.shape[-1])] ,-1)

	D = (0.5*torch.einsum('bi,bj->bij',grad2,grad2)+ hess2)*torch.eye(2).to(device)

	loss1 = ((0.5*torch.einsum('bi,bj->bij',grad1,grad1) + hess1)*D).sum()
	loss2 = (torch.einsum('bi,bj->bij',grad1,grad2)*(hess2*torch.eye(2).to(device))).sum()

	grad22 = torch.diagonal(hess2, dim1=-1,dim2=-2)
	grad222 = torch.stack([grad(grad22[:, i].sum(),x, create_graph=True, retain_graph=True)[0] for i in range(x.shape[-1])] ,-1)

	loss3 = (grad1*torch.diagonal(grad222, dim1=-1,dim2=-2)).sum()


	return (loss1+loss2+loss3)/x.shape[0]


def calc_idsm_loss_sst(x):

	grad1, hess1 = compute_model_score_and_hess(x)
	grad2, hess2 = compute_flow_score_and_hess(x)

	D = 0.5*torch.einsum('bi,bj->bij',grad2,grad2)

	loss1 = ((0.5*torch.einsum('bi,bj->bij',grad1,grad1) + hess1)*D).sum()
	loss2 = (torch.einsum('bi,bj->bij',grad1,grad2)*(hess2*torch.eye(2).to(device))).sum()
	loss2 = (0.5*(grad1*grad2).sum(-1)*torch.diagonal(hess2, dim1=-2,dim2=-1).sum(-1)).sum()
	loss3 = (0.5*torch.einsum('bi,bj->bij',grad1,grad2)*hess2).sum()

	return (loss1+loss2+loss3)/x.shape[0]


def calc_idsm_loss_sst_hess(x):

	grad1, hess1 = compute_model_score_and_hess(x)
	x = x.to(device)
	x.requires_grad_(True)
	logp = flow_model.log_prob(x).sum()
	grad2 = grad(logp, x, create_graph=True)[0]
	hess2 = torch.stack([grad(grad2[:, i].sum(),x, create_graph=True, retain_graph=True)[0] for i in range(x.shape[-1])] ,-1)

	D = 0.5*torch.einsum('bi,bj->bij',grad2,grad2) + hess2

	loss1 = ((0.5*torch.einsum('bi,bj->bij',grad1,grad1) + hess1)*D).sum()
	loss2 = (torch.einsum('bi,bj->bij',grad1,grad2)*(hess2*torch.eye(2).to(device))).sum()
	loss2 = (0.5*(grad1*grad2).sum(-1)*torch.diagonal(hess2, dim1=-2,dim2=-1).sum(-1)).sum()
	loss3 = (0.5*torch.einsum('bi,bj->bij',grad1,grad2)*hess2).sum()

	grad22 = torch.diagonal(hess2, dim1=-2,dim2=-1).sum(-1).sum()
	grad222 = grad(grad22, x, create_graph=True)[0]

	loss4 = (grad1*grad222).sum()

	return (loss1+loss2+loss3+loss4)/x.shape[0]


def calc_idsm_loss_sstinv(x):

	grad1, hess1 = compute_model_score_and_hess(x)
	grad2, hess2 = compute_flow_score_and_hess(x)

	D = 0.5*torch.einsum('bi,bj-> bij',1/grad2,1/grad2)*torch.eye(2).to(device)

	loss1 = ((0.5*torch.einsum('bi,bj->bij',grad1,grad1) + hess1)*D).sum()
	loss2 = -1*(torch.einsum('bi,bj->bij',grad1,grad2)*D*D*hess2).sum()


	return (loss1+loss2)/x.shape[0]


def calc_true_idsm_loss_old(x):
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


def calc_true_idsm_loss_diagsst(x):

	grad1, hess1 = compute_model_score_and_hess(x)
	grad2, hess2 = compute_true_score_and_hess(x)

	D = 0.5*torch.einsum('bi,bj->bij',grad2,grad2)*torch.eye(2).to(device)

	loss1 = ((0.5*torch.einsum('bi,bj->bij',grad1,grad1) + hess1)*D).sum()
	loss2 = (torch.einsum('bi,bj->bij',grad1,grad2)*(hess2*torch.eye(2).to(device))).sum()

	return (loss1+loss2)/x.shape[0]

def calc_true_idsm_loss_diagsst_diaghess(x):
	x = x.to(device)

	grad1, hess1 = compute_model_score_and_hess(x)
	grad2, hess2 = compute_true_score_and_hess(x)

	D = (0.5*torch.einsum('bi,bj->bij',grad2,grad2)+ hess2)*torch.eye(2).to(device)

	loss1 = ((0.5*torch.einsum('bi,bj->bij',grad1,grad1) + hess1)*D).sum()
	loss2 = (torch.einsum('bi,bj->bij',grad1,grad2)*(hess2*torch.eye(2).to(device))).sum()



	grad222 = torch.stack([-3*torch_e()**2*x[:,0],
		torch.zeros(x.shape[0]).to(device)],-1).detach()

	loss3 = (grad1*grad222).sum()


	return (loss1+loss2+loss3)/x.shape[0]


def calc_true_idsm_loss_sst(x):

	grad1, hess1 = compute_model_score_and_hess(x)
	grad2, hess2 = compute_true_score_and_hess(x)

	D = 0.5*torch.einsum('bi,bj->bij',grad2,grad2)

	loss1 = ((0.5*torch.einsum('bi,bj->bij',grad1,grad1) + hess1)*D).sum()
	loss2 = (torch.einsum('bi,bj->bij',grad1,grad2)*(hess2*torch.eye(2).to(device))).sum()
	loss2 = (0.5*(grad1*grad2).sum(-1)*torch.diagonal(hess2, dim1=-2,dim2=-1).sum(-1)).sum()
	loss3 = (0.5*torch.einsum('bi,bj->bij',grad1,grad2)*hess2).sum()

	return (loss1+loss2+loss3)/x.shape[0]


def calc_true_idsm_loss_sst_hess(x):
	x = x.to(device)

	grad1, hess1 = compute_model_score_and_hess(x)
	grad2, hess2 = compute_true_score_and_hess(x)

	D = 0.5*torch.einsum('bi,bj->bij',grad2,grad2) + hess2

	loss1 = ((0.5*torch.einsum('bi,bj->bij',grad1,grad1) + hess1)*D).sum()
	loss2 = (torch.einsum('bi,bj->bij',grad1,grad2)*(hess2*torch.eye(2).to(device))).sum()
	loss2 = (0.5*(grad1*grad2).sum(-1)*torch.diagonal(hess2, dim1=-2,dim2=-1).sum(-1)).sum()
	loss3 = (0.5*torch.einsum('bi,bj->bij',grad1,grad2)*hess2).sum()

	grad222 = torch.stack([-3*torch_e()**2*x[:,0],torch_e()+torch.zeros(x.shape[0]).to(device)],-1).detach()

	loss4 = (grad1*grad222).sum()

	return (loss1+loss2+loss3+loss4)/x.shape[0]


def calc_true_idsm_loss_sstinv(x):

	grad1, hess1 = compute_model_score_and_hess(x)
	grad2, hess2 = compute_flow_score_and_hess(x)

	D = 0.5*torch.einsum('bi,bj-> bij',1/grad2,1/grad2)*torch.eye(2).to(device)

	loss1 = ((0.5*torch.einsum('bi,bj->bij',grad1,grad1) + hess1)*D).sum()
	loss2 = -1*(torch.einsum('bi,bj->bij',grad1,grad2)*D*D*hess2).sum()


	return (loss1+loss2)/x.shape[0]	



def calc_flow_score(x):
	x = x.to(device)
	x.requires_grad_(True)
	logp = flow_model.log_prob(x).sum()
	grad1 = grad(logp, x, create_graph=True)[0]
	return grad1

def calc_true_score(x):

	cnst_e = torch.exp(torch.tensor([1.]))

	grad1 = torch.stack([-0.5*cnst_e**2*x[:,0]**3 + cnst_e**2*x[:,1]*x[:,0] + (cnst_e**2 - 1)*x[:,0],
		-cnst_e**2*x[:,1] + 0.5*cnst_e**2*x[:,0]**2 - cnst_e**2],-1)
	return grad1



def torch_e():
	return torch.exp(torch.tensor([1.])).to(device)



def true_energy_func(x):
	cnst_e = torch.exp(torch.tensor([1.]))

	return -0.5*x[:,0]**2 - 0.5*cnst_e**2*(x[:,1]-0.5*x[:,0]**2 + 1)**2



from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

import numpy as np
from matplotlib import cm


def banana_energy_plot(train_name, epoch=0):

	L_BOX = -5
	R_BOX = 5
	KNOTS = 250

	fig=plt.figure()
	ax=fig.add_subplot(111,projection='3d')


	u=np.linspace(L_BOX,R_BOX,KNOTS)
	xx,yy=np.meshgrid(u,u)

	x_in =  (torch.stack([torch.tensor(xx).flatten(), torch.tensor(yy).flatten()],-1) + torch.tensor([0,-1]))
	with torch.no_grad():
		zz = (true_energy_func(x_in)-true_energy_func(torch.tensor([[0.0,-1.0]]))).reshape(KNOTS,KNOTS)

		x_in = x_in.to(device).to(torch.float32)

		z= (e_model(x_in)- e_model(torch.tensor([[0.0,-1.0]]).to(device))).reshape(KNOTS,KNOTS)

		ax.plot_surface(x_in[:,0].reshape(KNOTS,KNOTS).to('cpu').detach().numpy(),x_in[:,1].reshape(KNOTS,KNOTS).to('cpu').detach().numpy(),z.cpu().detach().numpy(),rstride=4,cstride=4,cmap=cm.YlGnBu_r)


		ax.plot_surface(x_in[:,0].reshape(KNOTS,KNOTS).to('cpu').detach().numpy(),x_in[:,1].reshape(KNOTS,KNOTS).to('cpu').detach().numpy(),zz.numpy(),rstride=4,cstride=4,cmap=cm.coolwarm)

	d3_fig = fig.get_figure()
	d3_fig.savefig(os.path.join(asset_dir,'./{}_banna_energy_3d_epoch_{}.png'.format(train_name, epoch)), dpi = 400)
	plt.close()

	return None

def banana_score_plot(train_name, epoch=0):

	L_BOX = -5
	R_BOX = 5
	KNOTS = 250

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
	d3_fig.savefig(os.path.join(asset_dir,'./{}_banna_score_epoch_{}.png'.format(train_name, epoch)), dpi = 400)
	plt.close()

	return None




def energy_net(input_dim):
	return MLP(int(input_dim), 1,hidden_units=[512,512,256,256,128,128,64,64,32,32,16,16],
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



TRAIN_NAME = 'IDSM_DIAGSST_TRAIN'


ism_losses = {'ism':[]}
idsm_losses = {'diagsst':[],'diagsst_diaghess':[],'sst':[],'sst_hess':[], 'sstinv':[]}
tidsm_losses = {'diagsst':[],'diagsst_diaghess':[],'sst':[],'sst_hess':[], 'sstinv':[]}

ism_eval_losses = {'ism':[]}
idsm_eval_losses = {'diagsst':[],'diagsst_diaghess':[],'sst':[],'sst_hess':[], 'sstinv':[]}
tidsm_eval_losses = {'diagsst':[],'diagsst_diaghess':[],'sst':[],'sst_hess':[], 'sstinv':[]}


for tryy in range(1):

	ism_loss = {'ism':[]}
	idsm_loss = {'diagsst':[],'diagsst_diaghess':[],'sst':[],'sst_hess':[], 'sstinv':[]}
	tidsm_loss = {'diagsst':[],'diagsst_diaghess':[],'sst':[],'sst_hess':[], 'sstinv':[]}

	ism_eval_loss = {'ism':[]}
	idsm_eval_loss = {'diagsst':[],'diagsst_diaghess':[],'sst':[],'sst_hess':[], 'sstinv':[]}
	tidsm_eval_loss = {'diagsst':[],'diagsst_diaghess':[],'sst':[],'sst_hess':[], 'sstinv':[]}

	e_model = energy_net(2).to(device)

	epoch = 0
	optimizer = Adam(e_model.parameters(), lr=1e-5)

	for epoch in range(epoch,800):
		l = 0.0
		ll_diagsst = 0.0
		ll_diagsst_diaghess = 0.0
		ll_sst = 0.0
		ll_sst_hess = 0.0
		ll_sstinv = 0.0

		lll_diagsst = 0.0
		lll_diagsst_diaghess = 0.0
		lll_sst = 0.0
		lll_sst_hess = 0.0
		lll_sstinv = 0.0		

		for i,x in enumerate(train_loader):
			l += calc_ism_loss(x).detach().cpu().item()
			ll_diagsst += calc_idsm_loss_diagsst(x).detach().cpu().item()
			ll_diagsst_diaghess += calc_idsm_loss_diagsst_diaghess(x).detach().cpu().item()
			ll_sst += calc_idsm_loss_sst(x).detach().cpu().item()
			ll_sst_hess += calc_idsm_loss_sst_hess(x).detach().cpu().item()
			ll_sstinv += calc_idsm_loss_sstinv(x).detach().cpu().item()

			lll_diagsst += calc_true_idsm_loss_diagsst(x).detach().cpu().item()
			lll_diagsst_diaghess += calc_true_idsm_loss_diagsst_diaghess(x).detach().cpu().item()
			lll_sst += calc_true_idsm_loss_sst(x).detach().cpu().item()
			lll_sst_hess += calc_true_idsm_loss_sst_hess(x).detach().cpu().item()
			lll_sstinv += calc_true_idsm_loss_sstinv(x).detach().cpu().item()

			# l += 0
			# ll += 0
			# lll += 0		

			optimizer.zero_grad()

			loss = calc_idsm_loss_diagsst(x)

			loss.backward()

			optimizer.step()
			optimizer.zero_grad()

			# l += loss.detach()

			print('ism Epoch: {}/{}, Iter: {}/{}, AvgLoss: {:.3f}'.format(epoch+1,20,i+1,len(train_loader),l/(i+1)),end='\r')
		ism_loss['ism'].append(l/(i+1))
		idsm_loss['diagsst'].append(ll/(i+1))
		idsm_loss['diagsst_diaghess'].append(ll/(i+1))
		idsm_loss['sst'].append(ll/(i+1))
		idsm_loss['sst_hess'].append(ll/(i+1))
		idsm_loss['sstinv'].append(ll/(i+1))

		tidsm_loss['diagsst'].append(lll_diagsst/(i+1))
		tidsm_loss['diagsst_diaghess'].append(lll_diagsst_diaghess/(i+1))
		tidsm_loss['sst'].append(lll_sst/(i+1))
		tidsm_loss['sst_hess'].append(lll_sst_hess/(i+1))
		tidsm_loss['sstinv'].append(lll_sstinv/(i+1))

		if tryy == 0:
			writer.add_scalar("{}_ism_loss".format(TRAIN_NAME), l/(i+1), epoch)
			writer.add_scalar("{}_idsm_loss_diagsst".format(TRAIN_NAME), ll_diagsst/(i+1), epoch)
			writer.add_scalar("{}_idsm_loss_diagsst_diaghess".format(TRAIN_NAME), ll_diagsst_diaghess/(i+1), epoch)
			writer.add_scalar("{}_idsm_loss_sst".format(TRAIN_NAME), ll_sst/(i+1), epoch)
			writer.add_scalar("{}_idsm_loss_sst_hess".format(TRAIN_NAME), ll_sst_hess/(i+1), epoch)
			writer.add_scalar("{}_idsm_loss_sstinv".format(TRAIN_NAME), ll_sstinv/(i+1), epoch)
			writer.add_scalar("{}_tidsm_loss_diagsst".format(TRAIN_NAME), lll_diagsst/(i+1), epoch)
			writer.add_scalar("{}_tidsm_loss_diagsst_diaghess".format(TRAIN_NAME), lll_diagsst_diaghess/(i+1), epoch)
			writer.add_scalar("{}_tidsm_loss_sst".format(TRAIN_NAME), lll_sst/(i+1), epoch)
			writer.add_scalar("{}_tidsm_loss_sst_hess".format(TRAIN_NAME), lll_sst_hess/(i+1), epoch)
			writer.add_scalar("{}_tidsm_loss_sstinv".format(TRAIN_NAME), lll_sstinv/(i+1), epoch)


		print('')

		if epoch %10 ==0:

			l = 0.0
			ll = 0.0
			lll = 0.0
			for i,x in enumerate(test_loader):
				l += calc_ism_loss(x).detach().cpu().item()
				ll_diagsst += calc_idsm_loss_diagsst(x).detach().cpu().item()
				ll_diagsst_diaghess += calc_idsm_loss_diagsst_diaghess(x).detach().cpu().item()
				ll_sst += calc_idsm_loss_sst(x).detach().cpu().item()
				ll_sst_hess += calc_idsm_loss_sst_hess(x).detach().cpu().item()
				ll_sstinv += calc_idsm_loss_sstinv(x).detach().cpu().item()

				lll_diagsst += calc_true_idsm_loss_diagsst(x).detach().cpu().item()
				lll_diagsst_diaghess += calc_true_idsm_loss_diagsst_diaghess(x).detach().cpu().item()
				lll_sst += calc_true_idsm_loss_sst(x).detach().cpu().item()
				lll_sst_hess += calc_true_idsm_loss_sst_hess(x).detach().cpu().item()
				lll_sstinv += calc_true_idsm_loss_sstinv(x).detach().cpu().item()

				# l += calc_ism_loss(x).detach().cpu().item()
				# ll += 0
				# lll+= 0
			ism_eval_loss['ism'].append(l/(i+1))
			idsm_eval_loss['diagsst'].append(ll/(i+1))
			idsm_eval_loss['diagsst_diaghess'].append(ll/(i+1))
			idsm_eval_loss['sst'].append(ll/(i+1))
			idsm_eval_loss['sst_hess'].append(ll/(i+1))
			idsm_eval_loss['sstinv'].append(ll/(i+1))

			tidsm_eval_loss['diagsst'].append(lll_diagsst/(i+1))
			tidsm_eval_loss['diagsst_diaghess'].append(lll_diagsst_diaghess/(i+1))
			tidsm_eval_loss['sst'].append(lll_sst/(i+1))
			tidsm_eval_loss['sst_hess'].append(lll_sst_hess/(i+1))
			tidsm_eval_loss['sstinv'].append(lll_sstinv/(i+1))
			if tryy == 0:
				writer.add_scalar("{}_eval_ism_loss".format(TRAIN_NAME), l/(i+1), epoch)
				writer.add_scalar("{}_eval_idsm_loss_diagsst".format(TRAIN_NAME), ll_diagsst/(i+1), epoch)
				writer.add_scalar("{}_eval_idsm_loss_diagsst_diaghess".format(TRAIN_NAME), ll_diagsst_diaghess/(i+1), epoch)
				writer.add_scalar("{}_eval_idsm_loss_sst".format(TRAIN_NAME), ll_sst/(i+1), epoch)
				writer.add_scalar("{}_eval_idsm_loss_sst_hess".format(TRAIN_NAME), ll_sst_hess/(i+1), epoch)
				writer.add_scalar("{}_eval_idsm_loss_sstinv".format(TRAIN_NAME), ll_sstinv/(i+1), epoch)
				writer.add_scalar("{}_eval_tidsm_loss_diagsst".format(TRAIN_NAME), lll_diagsst/(i+1), epoch)
				writer.add_scalar("{}_eval_tidsm_loss_diagsst_diaghess".format(TRAIN_NAME), lll_diagsst_diaghess/(i+1), epoch)
				writer.add_scalar("{}_eval_tidsm_loss_sst".format(TRAIN_NAME), lll_sst/(i+1), epoch)
				writer.add_scalar("{}_eval_tidsm_loss_sst_hess".format(TRAIN_NAME), lll_sst_hess/(i+1), epoch)
				writer.add_scalar("{}_eval_tidsm_loss_sstinv".format(TRAIN_NAME), lll_sstinv/(i+1), epoch)
			banana_energy_plot(TRAIN_NAME,epoch)
			banana_score_plot(TRAIN_NAME,epoch)

		if epoch %100 == 0:
			energy_samples = e_model_hmc_sample(1000).detach().cpu()
			fig = plt.figure()
			plt.scatter(energy_samples[:,0],energy_samples[:,1],s=1,label='energy model')
			plt.legend()
			scatter_fig = fig.get_figure()
			scatter_fig.savefig(os.path.join(asset_dir,'./banana_data_flow_and_energy_train_{}_try_{}_epoch_{}.png'.format(TRAIN_NAME,tryy,epoch)), dpi = 400)
			plt.close()

			del energy_samples
			gc.collect()



	torch.save(e_model.state_dict(),os.path.join(asset_dir,'{}_banana_energy_model_try_{}_epoch_{}.pth'.format(TRAIN_NAME,tryy,epoch)))


	ism_losses['ism'].append(np.array(ism_loss['ism']))
	idsm_losses['diagsst'].append(np.array(idsm_loss['diagsst']))
	idsm_losses['diagsst_diaghess'].append(np.array(idsm_loss['diagsst_diaghess']))
	idsm_losses['sst'].append(np.array(idsm_loss['sst']))
	idsm_losses['sst_hess'].append(np.array(idsm_loss['sst_hess']))
	idsm_losses['sstinv'].append(np.array(idsm_loss['sstinv']))

	tidsm_losses['diagsst'].append(np.array(tidsm_losses['diagsst']))
	tidsm_losses['diagsst_diaghess'].append(np.array(tidsm_losses['diagsst_diaghess']))
	tidsm_losses['sst'].append(np.array(tidsm_losses['sst']))
	tidsm_losses['sst_hess'].append(np.array(tidsm_losses['sst_hess']))
	tidsm_losses['sstinv'].append(np.array(tidsm_losses['sstinv']))

	ism_eval_losses['ism'].append(np.array(ism_eval_loss['ism']))
	idsm_eval_losses['diagsst'].append(np.array(idsm_eval_loss['diagsst']))
	idsm_eval_losses['diagsst_diaghess'].append(np.array(idsm_eval_loss['diagsst_diaghess']))
	idsm_eval_losses['sst'].append(np.array(idsm_eval_loss['sst']))
	idsm_eval_losses['sst_hess'].append(np.array(idsm_eval_loss['sst_hess']))
	idsm_eval_losses['sstinv'].append(np.array(idsm_eval_loss['sstinv']))

	tidsm_eval_losses['diagsst'].append(np.array(tidsm_eval_losses['diagsst']))
	tidsm_eval_losses['diagsst_diaghess'].append(np.array(tidsm_eval_losses['diagsst_diaghess']))
	tidsm_eval_losses['sst'].append(np.array(tidsm_eval_losses['sst']))
	tidsm_eval_losses['sst_hess'].append(np.array(tidsm_eval_losses['sst_hess']))
	tidsm_eval_losses['sstinv'].append(np.array(tidsm_eval_losses['sstinv']))


	np.save(os.path.join(asset_dir,'{}_ism_losses_ism.npy'.format(TRAIN_NAME)),np.array(ism_losses['ism']))
	np.save(os.path.join(asset_dir,'{}_idsm_losses_diagsst.npy'.format(TRAIN_NAME)),np.array(idsm_losses['diagsst']))
	np.save(os.path.join(asset_dir,'{}_idsm_losses_diagsst_diaghess.npy'.format(TRAIN_NAME)),np.array(idsm_losses['diagsst_diaghess']))
	np.save(os.path.join(asset_dir,'{}_idsm_losses_sst.npy'.format(TRAIN_NAME)),np.array(idsm_losses['sst']))
	np.save(os.path.join(asset_dir,'{}_idsm_losses_sst_hess.npy'.format(TRAIN_NAME)),np.array(idsm_losses['sst_hess']))
	np.save(os.path.join(asset_dir,'{}_idsm_losses_sstinv.npy'.format(TRAIN_NAME)),np.array(idsm_losses['sstinv']))
	np.save(os.path.join(asset_dir,'{}_tidsm_losses_diagsst.npy'.format(TRAIN_NAME)),np.array(tidsm_losses['diagsst']))
	np.save(os.path.join(asset_dir,'{}_tidsm_losses_diagsst_diaghess.npy'.format(TRAIN_NAME)),np.array(tidsm_losses['diagsst_diaghess']))
	np.save(os.path.join(asset_dir,'{}_tidsm_losses_sst.npy'.format(TRAIN_NAME)),np.array(tidsm_losses['sst']))
	np.save(os.path.join(asset_dir,'{}_tidsm_losses_sst_hess.npy'.format(TRAIN_NAME)),np.array(tidsm_losses['sst_hess']))
	np.save(os.path.join(asset_dir,'{}_tidsm_losses_sstinv.npy'.format(TRAIN_NAME)),np.array(tidsm_losses['sstinv']))

	np.save(os.path.join(asset_dir,'{}_ism_eval_losses_ism.npy'.format(TRAIN_NAME)),np.array(ism_eval_losses['ism']))
	np.save(os.path.join(asset_dir,'{}_idsm_eval_losses_diagsst.npy'.format(TRAIN_NAME)),np.array(idsm_eval_losses['diagsst']))
	np.save(os.path.join(asset_dir,'{}_idsm_eval_losses_diagsst_diaghess.npy'.format(TRAIN_NAME)),np.array(idsm_eval_losses['diagsst_diaghess']))
	np.save(os.path.join(asset_dir,'{}_idsm_eval_losses_sst.npy'.format(TRAIN_NAME)),np.array(idsm_eval_losses['sst']))
	np.save(os.path.join(asset_dir,'{}_idsm_eval_losses_sst_hess.npy'.format(TRAIN_NAME)),np.array(idsm_eval_losses['sst_hess']))
	np.save(os.path.join(asset_dir,'{}_idsm_eval_losses_sstinv.npy'.format(TRAIN_NAME)),np.array(idsm_eval_losses['sstinv']))
	np.save(os.path.join(asset_dir,'{}_tidsm_eval_losses_diagsst.npy'.format(TRAIN_NAME)),np.array(tidsm_eval_losses['diagsst']))
	np.save(os.path.join(asset_dir,'{}_tidsm_eval_losses_diagsst_diaghess.npy'.format(TRAIN_NAME)),np.array(tidsm_eval_losses['diagsst_diaghess']))
	np.save(os.path.join(asset_dir,'{}_tidsm_eval_losses_sst.npy'.format(TRAIN_NAME)),np.array(tidsm_eval_losses['sst']))
	np.save(os.path.join(asset_dir,'{}_tidsm_eval_losses_sst_hess.npy'.format(TRAIN_NAME)),np.array(tidsm_eval_losses['sst_hess']))
	np.save(os.path.join(asset_dir,'{}_tidsm_eval_losses_sstinv.npy'.format(TRAIN_NAME)),np.array(tidsm_eval_losses['sstinv']))


torch.cuda.empty_cache()















