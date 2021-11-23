import torch
import matplotlib.pyplot as plt
import torchvision.utils as vutils
import numpy as np
import math





def one_shot_vis(dataloader,nrow):
	assert isinstance(dataloader,DataLoader)
	imgTensor = next(iter(dataloader))

	grid = vutils.make_grid(imgTensor[:nrow*nrow], padding = 4, nrow=nrow)
	grid = grid.permute(1, 2, 0)
	plt.imshow(grid);plt.show()

	return grid

# data = MyCIFAR10()
# train_loader, test_loader = data.get_data_loaders(32)
# imgTensor = next(iter(train_loader))
# one_shot_vis(train_loader,nrow=4)


def sum_except_batch(x,num_dims=1):
	return x.reshape(*x.shape[:num_dims],-1).sum(-1)

def mean_except_batch(x,num_dims=1):
	return x.reshape(*x.shape[:num_dims],-1).mean(-1)

def loglik_nats(model,x):
	return -model.log_prob(x).mean()

def loglik_bpd(model,x):
	return -model.log_prob(x).sum() /(np.log(2)*x.shape.numel())

def elbo_nats(model,x):
	return loglik_nats(model, x)

def elbo_bpd(model,x):
	return loglik_bpd(model,x)

def iwbo(model,x,k):
	x_stack = torch.cat([x for _ in range(k)],dim=0)
	ll_stack = model.log_prob(x_stack)
	ll = torch.stack(torch.chunk(ll_stack, k, dim=0))
	return torch.logsumexp(ll,dim=0) - np.log(k)

def iwbo_batched(model, x, k, kbs):
	assert k % kbs == 0
	num_passes = k // kbs
	ll_batched = []
	for i in range(num_passes):
		x_stack = torch.cat([x for _ in range(kbs)], dim=0)
		ll_stack = model.log_prob(x_stack)
		ll_batched.append(torch.stack(torch.chunk(ll_stack, kbs, dim=0)))
	ll = torch.cat(ll_batched, dim=0)
	return torch.logsumexp(ll, dim=0) - math.log(k)

def iwbo_nats(model,x,k,kbs=None):
	if kbs: return -iwbo_batched(model,x,k,kbs).mean()
	else: return -iwbo(model,x,k).mean()	


def print_params(flow_model):
	for transform in flow_model.transforms:
		for name, param in transform.named_parameters():
			if param.requires_grad:
				print(name,param.data)
	return None



def merge_leading_dims(x, num_dims=2):

	"""Reshapes the tensor `x` such that the first `num_dims` dimensions are merged to one."""

	new_shape = torch.Size([-1]) + x.shape[num_dims:]

	return torch.reshape(x, new_shape)
    


def repeat_rows(x, num_reps):

	shape = x.shape
	x = x.unsqueeze(1)
	x = x.expand(shape[0],num_reps, *shape[1:])
	return merge_leading_dims(x, num_dims=2)