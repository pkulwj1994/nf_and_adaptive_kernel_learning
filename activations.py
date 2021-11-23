import torch
import torch.nn as nn
import torch.nn.functional as F



def gelu(x):
	return x*torch.sigmoid(1.702*x)

def swish(x):
	return x*torch.sigmoid(x)

def concat_relu(x):
	return F.relu(torch.cat([x,-x],dim=1))

def concat_elu(x):
	return F.elu(torch.cat([x,-x],dim=1))

def gated_tanh(x,dim):
	x_tanh, x_sigmoid = torch.chunk(x,2,dim=dim)
	return torch.tanh(x_tanh)*torch.sigmoid(x_sigmoid)

class GELU(nn.Module):

	def forward(self,input):
		return gelu(input)

class Swish(nn.Module):

	def forward(self,input):
		return swish(input)

class ConcatReLU(nn.Module):

	def forward(self,input):
		return concat_relu(input)


class GatedTanhUnit(nn.Module):

	def __init__(self,dim=1):
		super(GatedTanhUnit,self).__init__()
		self.dim = dim

	def forward(self,x):
		return gated_tanh(x,dim=self.dim)


### retrive activation layer
act_strs = {'elu','relu','gelu','swish'}
concat_act_strs = {'concat_elu','concat_relu'}

def act_module(act_str, allow_concat=False):
	if allow_concat: assert act_str in act_strs + concat_act_strs, 'Got invalid activation {}'.format(act_str)
	else: assert act_str in act_strs, 'Got invalid activation {}'.format(act_str)

	if act_str == 'relu': return nn.ReLU()
	elif act_str == 'elu': return nn.ELU()
	elif act_str == 'gelu': return GELU()
	elif act_str == 'swish': return Swish()
	elif act_str == 'concat_relu': return ConcatReLU()
	elif act_str == 'concat_elu': return ConcatELU()




def scale_fn(scale_str):
	assert scale_str in {'exp','softplus', 'sigmoid', 'tanh_exp'}
	if scale_str == 'exp': 	return lambda s: torch.exp(s)
	elif scale_str == 'softplus':	return lambda s: F.softplus(s)
	elif scale_str == 'sigmoid': return lambda s: torch.sigmoid(s+2.) + 1e-3
	elif scale_str == 'tanh_exp': return lambda s: torch.exp(2.*torch.tanh(s/2.))




