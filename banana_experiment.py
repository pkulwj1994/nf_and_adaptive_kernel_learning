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


device = 'cuda' if torch.cuda.is_available() else 'cpu'






### mnist part
DataSets = Crescent(train_samples=10000, test_samples=5000)
train_loader, test_loader = DataSets.get_data_loaders(128)



# def net(channels):
# 	return nn.Sequential(DenseNet(in_channels=channels//2,
# 		out_channels=channels,
# 		num_blocks=1,
# 		mid_channels=64,
# 		depth=8,
# 		growth=16,
# 		dropout=0.0,
# 		gated_conv=True,
# 		zero_init=True),
# 	ElementwiseParams2d(2))

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

# model = Flow(base_dist=StandardNormal((2,)),
# 			transforms=[
# 			AffineCouplingBijection1d(coupling_net(2)),
# 			SwitchBijection1d(),AffineCouplingBijection1d(coupling_net(2)),
# 			SwitchBijection1d(),AffineCouplingBijection1d(coupling_net(2)),
# 			SwitchBijection1d(),AffineCouplingBijection1d(coupling_net(2)),
# 			SwitchBijection1d(),AffineCouplingBijection1d(coupling_net(2))]).to(device)





# transform = AffineCouplingBijection1d(coupling_net(2))
# z,ldj = transform(x)

# transform = ActNormBijection(784)
# z,ldj = transform(z)

# transform = AffineCouplingBijection1d(coupling_net(784))
# z,ldj = transform(z)

# transform = ActNormBijection(784)
# z,ldj = transform(z)



# optimizer = Adam(model.parameters(), lr=1e-2)

# l = 0.0
# x = next(iter(train_loader))
# optimizer.zero_grad()
# loss = -model.log_prob(x.to(device)).mean()
# loss.backward()
# optimizer.step()
# l += loss.detach().cpu().item()
# print('Epoch: {}/{}, Iter: {}/{}, Nats: {:.3f}'.format(epoch+1,20,i+1,len(train_loader),l/(i+1)),end='\r')
# print('')

# print_params(model)

# torch.save(model.state_dict(),'./saved_model/mnist_flow_single_affinecoupling_layer.pth')


# model = Flow(base_dist=StandardNormal((784,)),
# 			transforms=[
# 			LogisticBijection1d(),
# 			AffineCouplingBijection1d(coupling_net(784))]).to(device)
# model.load_state_dict(torch.load('./saved_model/mnist_flow_single_affinecoupling_layer.pth'))
# model.eval()


# torch.save(model, PATH)
# model = torch.load(PATH)
# model.eval()

# t0 = model.transforms[0]

# t0.split_input(x)

# z = transform0(x.to(device))

# t1 = model.transforms[1]


### pretrain normalizing flow f_\phi

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

writer = tensorboard.SummaryWriter(tb_dir)





epoch = 0

model = make_flow()

optimizer = Adam(model.parameters(), lr=1e-6)

for epoch in range(epoch,200):
	l = 0.0
	for i,x in enumerate(train_loader):
		optimizer.zero_grad()
		loss = -model.log_prob(x.to(device)).mean()
		loss.backward()
		optimizer.step()
		l += loss.detach().cpu().item()
		print('Epoch: {}/{}, Iter: {}/{}, Nats: {:.3f}'.format(epoch+1,20,i+1,len(train_loader),l/(i+1)),end='\r')
	writer.add_scalar("flow_avg_nll", l/(i+1), epoch)
	torch.save(model.state_dict(),os.path.join(asset_dir,'pretrained_banana_flow.pth'))
	print('')

vecs = next(iter(test_loader))[:128]
samples = model.sample(128).cpu()

fig = plt.figure()
sns.kdeplot(vecs[:,0],vecs[:,1],shade=True,color='red')
scatter_fig = fig.get_figure()
scatter_fig.savefig(os.path.join(asset_dir,'./banana_data_density.png'), dpi = 400)

fig = plt.figure()
sns.kdeplot(samples[:,0],samples[:,1],shade=True)
scatter_fig = fig.get_figure()
scatter_fig.savefig(os.path.join(asset_dir,'./banana_model_density.png'), dpi = 400)

fig = plt.figure()
plt.scatter(vecs[:,0],vecs[:,1],s=1,label='data')
plt.legend()
plt.scatter(samples[:,0],samples[:,1],s=1,label='model')
plt.legend()
scatter_fig = fig.get_figure()
scatter_fig.savefig(os.path.join(asset_dir,'./banana_data_and_model.png'), dpi = 400)








### score matching
from torch.autograd.functional import jacobian as jcb, hessian as hess
from torch.autograd import grad 
from torch.autograd import Variable


def compute_implicit_score_diff(x):
	x = x.to(device)
	x.requires_grad_(True)
	logp = -e_model(x).sum()
	grad1 = grad(logp, x, create_graph=True)[0]
	hess1 = torch.stack([grad(grad1[:, i].sum(),x, create_graph=True, retain_graph=True)[0] for i in range(x.shape[-1])] ,-1)
	return torch.cat([(1/2*grad1[i].T@grad1[i] - hess1[i]).unsqueeze(0) for i in range(x.shape[0])],0)



def calc_ism_loss(x):

	# D = compute_batch_D(x)
	imp_mat_diff = compute_implicit_score_diff(x)

	return (imp_mat_diff).sum()/x.shape[0]

def calc_idsm_loss_old(x):

	def logprob_func(x):
		with torch.no_grad():
			return model.log_prob(x.to(device))

	def jcb2_hess(x):
		return 1/2*jcb(logprob_func, xi.unsqueeze(0)).squeeze(-2).T@jcb(logprob_func, xi.unsqueeze(0)).squeeze(-2) - hess(logprob_func, xi.unsqueeze(0)).squeeze(-2)

	D = compute_batch_D(x)
	imp_mat_diff = compute_implicit_score_diff(x)

	return (imp_mat_diff*D).sum()/x.shape[0]


def calc_idsm_loss(x):

	x = x.to(device)
	x.requires_grad_(True)
	logp = flow_model.log_prob(x).sum()
	grad1 = grad(logp, x, create_graph=True)[0]
	hess1 = torch.stack([grad(grad1[:, i].sum(),x, create_graph=True, retain_graph=True)[0] for i in range(x.shape[-1])] ,-1)

	D = torch.cat([(1/2*grad1[i].T@grad1[i] - hess1[i]).unsqueeze(0) for i in range(x.shape[0])],0)
	imp_mat_diff = compute_implicit_score_diff(x)
	return (imp_mat_diff*D).sum()/x.shape[0]



def energy_net(input_dim):
	return MLP(int(input_dim), 1,hidden_units=[512,256,128],
                                activation='elu',
                                in_lambda=None)



flow_model = make_flow()
flow_model.load_state_dict(torch.load(os.path.join(asset_dir,'pretrained_banana_flow.pth')))
flow_model.eval()










ism_train_ism_losses = []
ism_train_idsm_losses = []

ism_eval_ism_losses = []
ism_eval_idsm_losses = []


for tryy in range(1):

	ism_losses = []
	idsm_losses = []

	ism_eval_losses = []
	idsm_eval_losses = []

	e_model = energy_net(2).to(device)

	epoch = 0
	optimizer = Adam(e_model.parameters(), lr=1e-6)

	for epoch in range(epoch,800):
		l = 0.0
		ll = 0.0
		for i,x in enumerate(train_loader):
			optimizer.zero_grad()

			loss_ism = calc_ism_loss(x)

			loss_ism.backward()
			optimizer.step()
			l += loss_ism.detach().cpu().item()
			ll += calc_idsm_loss(x).detach().cpu().item()
			print('ism Epoch: {}/{}, Iter: {}/{}, AvgLoss: {:.3f}'.format(epoch+1,20,i+1,len(train_loader),l/(i+1)),end='\r')
		ism_losses.append(l/(i+1))
		idsm_losses.append(ll/(i+1))
		if tryy == 0:
			writer.add_scalar("avg_training_ism_loss_ism_train", l/(i+1), epoch)
			writer.add_scalar("avg_training_idsm_loss_ism_train", ll/(i+1), epoch)
		print('')

		torch.save(e_model.state_dict(),os.path.join(asset_dir,'ism_train_banana_energy_model_try_{}_epoch_{}.pth'.format(tryy,epoch)))


		if epoch %10 ==0:

			l = 0.0
			ll = 0.0
			for i,x in enumerate(test_loader):
				with torch.no_grad():

					l += calc_ism_loss(x).detach().cpu().item()
					ll += calc_idsm_loss(x).detach().cpu().item()
			ism_eval_losses.append(l/(i+1))
			idsm_eval_losses.append(ll/(i+1))
			if tryy == 0:
				writer.add_scalar("avg_eval_ism_loss_ism_train_1", l/(i+1), epoch)
				writer.add_scalar("avg_eval_idsm_loss_ism_train_1", ll/(i+1), epoch)


	ism_train_ism_losses.append(np.array(ism_losses))
	ism_train_idsm_losses.append(np.array(idsm_losses))

	ism_eval_ism_losses.append(np.array(ism_eval_losses))
	ism_eval_idsm_losses.append(np.array(idsm_eval_losses))

	np.save(os.path.join(asset_dir,'ism_train_ism_losses.npy'),np.array(ism_train_ism_losses))
	np.save(os.path.join(asset_dir,'ism_train_idsm_losses.npy'),np.array(ism_train_idsm_losses))

	np.save(os.path.join(asset_dir,'ism_train_ism_eval_losses_per10_epochs.npy'),np.array(ism_eval_ism_losses))
	np.save(os.path.join(asset_dir,'ism_train_idsm_eval_losses_per10_epochs.npy'),np.array(ism_eval_idsm_losses))





idsm_train_ism_losses = []
idsm_train_idsm_losses = []


idsm_eval_ism_losses = []
idsm_eval_idsm_losses = []



for tryy in range(1):

	ism_losses = []
	idsm_losses = []

	ism_eval_losses = []
	idsm_eval_losses = []

	e_model = energy_net(2).to(device)

	epoch = 0
	optimizer = Adam(e_model.parameters(), lr=1e-6)

	for epoch in range(epoch,800):
		l = 0.0
		ll = 0.0
		for i,x in enumerate(train_loader):
			optimizer.zero_grad()

			loss = calc_idsm_loss(x)

			loss.backward()
			optimizer.step()
			l += loss.detach().cpu().item()
			ll += calc_ism_loss(x).detach().cpu().item()
			print('idsm Epoch: {}/{}, Iter: {}/{}, AvgLoss: {:.3f}'.format(epoch+1,20,i+1,len(train_loader),l/(i+1)),end='\r')
		ism_losses.append(ll/(i+1))
		idsm_losses.append(l/(i+1))

		if tryy == 0:
			writer.add_scalar("avg_train_ism_loss_idsm_train", ll/(i+1), epoch)
			writer.add_scalar("avg_train_idsm_loss_idsm_train", l/(i+1), epoch)
		print('')

		torch.save(e_model.state_dict(),os.path.join(asset_dir,'idsm_train_banana_energy_model_try_{}_epoch_{}.pth'.format(tryy,epoch)))

		if epoch %10 ==0:

			l = 0.0
			ll = 0.0
			for i,x in enumerate(test_loader):
				with torch.no_grad():

					l += calc_ism_loss(x).detach().cpu().item()
					ll += calc_idsm_loss(x).detach().cpu().item()
			ism_eval_losses.append(l/(i+1))
			idsm_eval_losses.append(ll/(i+1))
			if tryy == 0:
				writer.add_scalar("avg_eval_ism_loss_ism_train_1", l/(i+1), epoch)
				writer.add_scalar("avg_eval_idsm_loss_ism_train_1", ll/(i+1), epoch)

	idsm_train_ism_losses.append(np.array(ism_losses))
	idsm_train_idsm_losses.append(np.array(idsm_losses))

	idsm_eval_ism_losses.append(np.array(ism_eval_losses))
	idsm_eval_idsm_losses.append(np.array(idsm_eval_losses))

	np.save(os.path.join(asset_dir,'idsm_train_ism_losses.npy'),np.array(idsm_train_ism_losses))
	np.save(os.path.join(asset_dir,'idsm_train_idsm_losses.npy'),np.array(idsm_train_idsm_losses))

	np.save(os.path.join(asset_dir,'idsm_train_ism_eval_losses_per10_epochs.npy'),np.array(idsm_eval_ism_losses))
	np.save(os.path.join(asset_dir,'idsm_train_idsm_eval_losses_per10_epochs.npy'),np.array(idsm_eval_idsm_losses))









# visualize


ism_train_ism_losses = np.load(os.path.join(asset_dir,'ism_train_ism_losses.npy'))
ism_train_idsm_losses = np.load(os.path.join(asset_dir,'ism_train_idsm_losses.npy'))


idsm_train_ism_losses = np.load(os.path.join(asset_dir,'idsm_train_ism_losses.npy'))
idsm_train_idsm_losses = np.load(os.path.join(asset_dir,'idsm_train_idsm_losses.npy'))







fig = plt.figure()
plt.title('ism losses')
plt.plot(np.arange(400),ism_train_ism_losses[0][:400],color='red',label='ism train')
plt.legend()
plt.plot(np.arange(400),idsm_train_ism_losses[0][:400],color='blue',label='idsm train')
plt.legend()
plot_fig = fig.get_figure()
plot_fig.savefig(os.path.join(asset_dir,'./ism_losses.png'), dpi = 400)



fig = plt.figure()
plt.title('idsm losses')
plt.plot(np.arange(400),ism_train_idsm_losses[0][:400],color='red',label='ism train')
plt.legend()
plt.plot(np.arange(400),idsm_train_idsm_losses[0][:400],color='blue',label='idsm train')
plt.legend()
plot_fig = fig.get_figure()
plot_fig.savefig(os.path.join(asset_dir,'./idsm_losses.png'), dpi = 400)




fig = plt.figure()
plt.title('ism train')
plt.plot(np.arange(400),ism_train_ism_losses[0][:400],color='red',label='ism loss')
plt.legend()
plt.plot(np.arange(400),ism_train_idsm_losses[0][:400],color='blue',label='idsm loss')
plt.legend()
plot_fig = fig.get_figure()
plot_fig.savefig(os.path.join(asset_dir,'./ism_train.png'), dpi = 400)



fig = plt.figure()
plt.title('idsm train')
plt.plot(np.arange(400),idsm_train_ism_losses[0][:400],color='red',label='ism loss')
plt.legend()
plt.plot(np.arange(400),idsm_train_idsm_losses[0][:400],color='blue',label='idsm loss')
plt.legend()
plot_fig = fig.get_figure()
plot_fig.savefig(os.path.join(asset_dir,'./idsm_train.png'), dpi = 400)






def torch_e():
	return torch.exp(torch.tensor([1.]))


def calc_true_banana_score_diff_loss(x):
	x1,x2 = torch.chunk(x,2,-1)
	s1 = -0.5*x1**3 + torch_e()**2*x2*x1**2 + (torch_e()**2 - 1)*x1
	s2 = torch_e()**2*x2 - 0.5*torch_e()**2*x1**2 + torch_e()**2
	true_score = torch.cat([s1,s2],-1).to(device)


	x = x.to(device)
	x.requires_grad_(True)
	logp = -e_model(x).sum()
	model_score = grad(logp, x, create_graph=True)[0].detach()

	return (torch.norm(grad1, dim=-1) ** 2 / 2.).mean()



