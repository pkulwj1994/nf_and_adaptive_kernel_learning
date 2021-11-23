import torch
from torchvision.transforms import Compose, ToTensor
import torch.nn as nn
import torch.nn.functional as F
from collections.abc import Iterable
from functools import reduce
from operator import mul

from torch.distributions import Normal
from torch.distributions import Bernoulli
from torch.utils import checkpoint

import scipy

from utils import *
from activations import *
import copy


########################################## non invertible mappings
# 	build based on nn.Module
#	contains polular neural networks
#	serve as deterministic mapping for stochastic mapping and flow transform
########################################## 

class LambdaLayer(nn.Module):
	def __init__(self, lambd):
		super(LambdaLayer,self).__init__()
		if lambd is None: lambd = lambda x:x
		self.lambd = lambd

	def forward(self,x):
		return self.lambd(x)

class MLP(nn.Sequential):
	def __init__(self,input_size, output_size, hidden_units, activation='relu',in_lambda=None,out_lambda=None):
		self.input_size = input_size
		self.output_size = output_size

		layers = []
		if in_lambda: layers.append(LambdaLayer(in_lambda))
		for in_size, out_size in zip([input_size]+hidden_units[:-1],hidden_units):
			layers.append(nn.Linear(in_size,out_size))
			layers.append(act_module(activation))
		layers.append(nn.Linear(hidden_units[-1], output_size))
		if out_lambda: layers.append(LambdaLayer(out_lambda))

		super(MLP, self).__init__(*layers)
		

class ElementwiseParams(nn.Module):

	def __init__(self, num_params, mode='interleaved'):
		super(ElementwiseParams, self).__init__()
		assert mode in {'interleaved','sequential'}
		self.num_params = num_params
		self.mode = mode
	
	def forward(self,x):
		assert x.dim()==2, 'Expected input of shape (B,D)'
		if self.num_params !=1:
			assert x.shape[1] % self.num_params == 0
			dims = x.shape[1] //self.num_params

			if self.mode == 'interleaved':
				x = x.reshape(x.shape[0:1] + (self.num_params,dims))
				x = x.permute([0,2,1])

			elif self.mode == 'sequential':
				x = x.reshape(x.shape[0:1] + (dims, self.num_params))

		return x

class ElementwiseParams1d(nn.Module):

	def __init__(self, num_params, mode='interleaved'):
		super(ElementwiseParams1d,self).__init__()
		assert mode in {'interleaved', 'sequential'}
		self.num_params = num_params
		self.mode = mode

	def forward(self,x):
		assert x.dim()==3, 'Expected input of shape (B,D,L)'
		if self.num_params != 1:
			assert x.shape[1] % self.num_params ==0
			dims = x.shape[1] //self.num_params

			if self.mode == 'interleaved':
				x = x.reshape(x.shape[0:1] + (self.num_params,dims)+ x.shape[2:])

				x = x.permute([0,2,3,1])

			elif self.mode == 'sequential':
				x = x.reshape(x.shape[0:1] + (dims, self.num_params) + x.shape[2:])
				x = x.permute([0,1,3,2])

		return x

class ElementwiseParams2d(nn.Module):

	def __init__(self, num_params, mode='interleaved'):
		super(ElementwiseParams2d, self).__init__()
		assert mode in {'interleaved', 'sequential'}
		self.num_params = num_params
		self.mode = mode

	def forward(self, x):
		assert x.dim() == 4, 'Expected input of shape (B,C,H,W)'
		if self.num_params != 1:
			assert x.shape[1] % self.num_params == 0
			channels = x.shape[1] // self.num_params
			if self.mode == 'interleaved':
				x = x.reshape(x.shape[0:1] + (self.num_params, channels) + x.shape[2:])
				x = x.permute([0,2,3,4,1])
			elif self.mode == 'sequential':
				x = x.reshape(x.shape[0:1] + (channels, self.num_params) + x.shape[2:])
				x = x.permute([0,1,3,4,2])
		return x


class DensLayer(nn.Module):
	def __init__(self, in_channels, growth, dropout):
		super(DensLayer, self).__init__()

		layers = []

		layers.extend([
			nn.Conv2d(in_channels, in_channels, kernel_size=1,
				stride=1, padding=0, bias=True),
			nn.ReLU(inplace=True),])

		if dropout>0:
			layers.append(nn.Dropout(p=dropout))

		layers.extend([
			nn.Conv2d(in_channels,growth,kernel_size=3,
				stride=1,padding=1,bias=True),
			nn.ReLU(inplace=True)
			])

		self.nn = nn.Sequential(*layers)

	def forward(self,x):
		h = self.nn(x)
		h = torch.cat([x,h],dim=1)

		return h


class GatedConv2d(nn.Module):
	def __init__(self, in_channels, out_channels, kernel_size, padding):
		super(GatedConv2d, self).__init__()
		self.in_channels = in_channels
		self.conv = nn.Conv2d(in_channels, out_channels*3,
			kernel_size=kernel_size,padding=padding)

	def forward(self,x):
		h = self.conv(x)
		a,b,c = torch.chunk(h,chunks=3,dim=1)

		return a + b*torch.sigmoid(c)


class DenseBlock(nn.Sequential):

	def __init__(self, in_channels, out_channels, depth, growth,
		dropout=0., gated_conv=False, zero_init=False):
		layers = [DensLayer(in_channels + i*growth,growth,dropout) for i in range(depth)]

		if gated_conv:
			layers.append(GatedConv2d(in_channels + depth*growth, out_channels, kernel_size=1, padding=0))
		else:
			layers.append(nn.Conv2d(in_channels+depth*growth, out_channels,kernel_size=1,padding=0))

		if zero_init:
			nn.init.zeros_(layers[-1].weight)
			if hasattr(layers[-1],'bias'):
				nn.init.zeros_(layers[-1].bias)

		super(DenseBlock,self).__init__(*layers)


class ResidualDenseBlock(nn.Module):
	def __init__(self, in_channels, out_channels, depth, growth,
		dropout=0., gated_conv=False, zero_init=False):
		super(ResidualDenseBlock,self).__init__()

		self.dense = DenseBlock(in_channels = in_channels,
			out_channels = out_channels,
			depth = depth,
			growth = growth,
			dropout = dropout,
			gated_conv = gated_conv,
			zero_init = zero_init)

	def forward(self, x):
		return x + self.dense(x)	


class DenseNet(nn.Sequential):
	def __init__(self, in_channels, out_channels, num_blocks,
		mid_channels, depth, growth, dropout,
		gated_conv=False, zero_init=False):

		layers = [nn.Conv2d(in_channels,mid_channels, kernel_size=1, padding=0)]+[ResidualDenseBlock(in_channels=mid_channels,
			out_channels=mid_channels,
			depth=depth,
			growth=growth,
			dropout=dropout,
			gated_conv=gated_conv,
			zero_init=False) for _ in range(num_blocks)] + [nn.Conv2d(mid_channels,out_channels,kernel_size=1,padding=0)]
		if zero_init:
			nn.init.zeros_(layers[-1].weight)
			if hasattr(layers[-1],'bias'):
				nn.init.zeros_(layers[-1].bias)

		super(DenseNet,self).__init__(*layers)

class MultiscaleDenseNet(nn.Module):
	def __init__(self, in_channels, out_channels, num_scales, num_blocks, mid_channels,
		depth, growth, dropout, gated_conv=False, zero_init=False):

		super(MultiscaleDenseNet, self).__init__()
		assert num_scales >1
		self.num_scales = num_scales


		def get_densenet(cin, cout, zinit=False):
			return DenseNet(in_channels=cin,
				out_channels=cout,
				num_blocks=num_blocks,
				mid_channels=mid_channels,
				depth=depth,
				growth=growth,
				dropout=dropout,
				gated_conv=gated_conv,
				zero_init=zinit)

		self.down_in = get_densenet(in_channels, mid_channels)

		down = []
		for i in range(num_scales -1):
			down.append(nn.Sequential(nn.Conv2d(mid_channels, mid_channels, kernel_size=2, padding=0,stride=2),
				get_densenet(mid_channels,mid_channels)))

		self.down = nn.ModuleList(down)

		up = []

		for i in range(num_scales -1):
			np.append(nn.Sequential(get_densenet(mid_channels,mid_channels),
				nn.ConvTranspose2d(mid_channels, mid_channels, kernel_size=2, padding=1, stride=2)))

		self.up = nn.ModuleList(up)

		self.up_out = get_densenet(mid_channels, out_channels, zinit=zero_init)


	def forward(self,x):

		d = [self.down_in(x)]

		for down_layer in self.down:
			d.append(down_layer(d[-1]))

		u = [d[-1]]
		for i, up_layer in enumerate(self.up):
			u.append(up_layer(u[-1])+d[self.num_scales -2-i])

		return self.up_out(u[-1])






### sequence network part 

class DenseTransformerBlock(nn.Module):

	def __init__(self, d_model, nhead, dim_feedforward=512, dropout=0.1,activation='gelu',kdim=None, vdim=None, attn_bias=True, checkpoint=False):
		super(DenseTransformerBlock, self).__init__()
		self.self_attn = nn.MultiheadAttention(d_model,nhead, dropout=dropout, kdim=kdim, vdim=vdim,bias=attn_bias)

		self.linear1 = nn.Linear(d_model, dim_feedforward)
		self.linear2 = nn.Linear(dim_feedforward, d_model)

		self.norm1 = nn.LayerNorm(d_model)
		self.norm2 = nn.LayerNorm(d_model)
		self.dropout1 = nn.Dropout(dropout)
		self.dropout2 = nn.Dropout(dropout)

		self.activation = act_module(activation)
		self.checkpoint = checkpoint

		self._reset_parameters()


	def _reset_parameters(self):
		nn.init.normal_(self.linear1.weight, std=0.125/math.sqrt(self.linear1.weight.shape[1]))
		nn.init.normal_(self.linear2.weight, std=0.125/math.sqrt(self.linear2.weight.shape[1]))

		nn.init.zeros_(self.linear1.bias)
		nn.init.zeros_(self.linear2.bias)

		nn.init.normal_(self.self_attn.in_proj_weight, std = 0.1245/math.sqrt(self.self_attn.in_proj_weight.shape[1]))
		if not self.self_attn._qkv_same_embed_dim:
			nn.init.normal_(self.self_attn.q_proj_weight,std=0.125/math.sqrt(self.self_attn.q_proj_weight.shape[1]))
			nn.init.normal_(self.self_attn.k_proj_weight, std=0.125/math.sqrt(self.self_attn.k_proj_weight.shape[1]))
			nn.init.normal_(self.self_attn.v_proj_weight, std=0.125/math.sqrt(self.self_attn.v_proj_weight.shape[1]))

		if self.self_attn.in_proj_bias is not None:
			nn.init.zeros_(self.self_attn.in_proj_bias)

		nn.init.normal_(self.self_attn.out_proj.weight, std=0.125/math.sqrt(self.self_attn.out_proj.weight.shape[1]))

		if self.self_attn.out_proj.bias is not None:
			nn.init.zeros_(self.self_attn.out_proj.bias)


	def _attn_block(self, x, attn_mask=None, key_padding_mask=None):
		x = self.norm1(x)
		x = self.self_attn(x,x,x, attn_mask=attn_mask, key_padding_mask=key_padding_mask)[0]
		x = self.dropout1(x)
		return x

	def _ff_block(self,x):
		x = self.norm2(x)
		x = self.linear2(self.activation(self.linear1(x)))
		x = self.dropout2(x)
		return x

	def _forward(self, x, attn_mask=None, key_padding_mask=None):
		ax = self._attn_block(x, attn_mask=attn_mask,key_padding_mask=key_padding_mask)
		bx = self._ff_block(x+ax)
		return x + ax+bx

	def forward(self, x, attn_mask=None, key_padding_mask=None):
		if not self.checkpoint:
			return self._forward(x,attn_mask, key_padding_mask)
		else:
			x.requires_grad_(True)
			return checkpoint.checkpoint(self._forward, x, attn_mask, key_padding_mask)

def _get_clones(module, N):
	return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

class DenseTransformer(nn.Module):

	def __init__(self, d_model=512, nhead=8,
		num_layers=6, dim_feedforward=512, dropout=0.1,
		activation='gelu',kdim=None, vdim=None,
		attn_bias=True, checkpoint_blocks=False):

		super(DenseTransformer, self).__init__()

		decoder_layer = DenseTransformerBlock(d_model=d_model,nhead=nhead,dim_feedforward=dim_feedforward,
			dropout=dropout,activation=activation,kdim=kdim,vdim=vdim,attn_bias=attn_bias,checkpoint=checkpoint_blocks)

		self.layers = _get_clones(decoder_layer, num_layers)
		self.out_norm = nn.LayerNorm(d_model)

		self.num_layers = num_layers
		self.d_model = d_model
		self.nhead = nhead

		self._reset_parameters()


	def forward(self, x, key_padding_mask=None):
		if x.size(2) != self.d_model:
			raise RuntimeError('the feature number of src and tgt must be equal to d_model')

		attn_mask = self.generate_square_subsequent_mask(x.shape[0]).to(x.device)

		for decoder_layer in self.layers:
			x = decoder_layer(x, attn_mask=attn_mask, key_padding_mask=key_padding_mask)

		return self.out_norm(x)

	def generate_square_subsequent_mask(self, sz):

		mask = (torch.triu(torch.ones(sz,sz))==1).transpose(0,1)
		mask = mask.float().masked_fill(mask ==0, float('-inf')).masked_fill(mask ==1, float(0.0))

		return mask

	def _reset_parameters(self):

		for p in self.parameters():
			if p.dim()>1:
				nn.init.xavier_uniform_(p)

class PositionalEncodeingImage(nn.Module):

	def __init__(self, image_shape, embedding_dim):
		super(PositionalEncodeingImage, self).__init__()
		assert len(image_shape) == 3, 'image shape should have length 3: (C,H,W)'
		self.image_shape = image_shape
		self.embedding_dim = embedding_dim

		c,h,w = image_shape
		self.encode_c = nn.Parameter(torch.Tensor(1,c,1,1,embedding_dim))
		self.encode_h = nn.Parameter(torch.Tensor(1,1,h,1,embedding_dim))
		self.encode_w = nn.Parameter(torch.Tensor(1,1,1,w,embedding_dim))

		self.reset_parameters()

	def reset_parameters(self):

		nn.init.normal_(self.encode_c, std=0.125/math.sqrt(3*self.embedding_dim))
		nn.init.normal_(self.encode_h, std=0.125/math.sqrt(3*self.embedding_dim))
		nn.init.normal_(self.encode_w, std=0.125/math.sqrt(3*self.embedding_dim))

	def forward(self,x):
		return x + self.encode_c + self.encode_h + self.encode_w

class AutoregressiveShift(nn.Module):

	def __init__(self, embed_dim):
		super(AutoregressiveShift, self).__init__()
		self.embed_dim = embed_dim
		self.first_token = nn.Parameter(torch.Tensor(1,1,embed_dim))
		self._reset_parameters()

	def _reset_parameters(self):
		nn.init.xavier_uniform_(self.first_token)

	def forward(self,x):
		first_token = self.first_token.expand(1,x.shape[1],self.embed_dim)
		return torch.cat([first_token, x[:-1]], dim=0)

def _prep_zigzag_cs(channels, height, width):

	diagonals = [[] for i in range(height+width-1)]

	for i in range(height):
		for j in range(width):
			sum = i+j
			if(sum%2==0):
				diagonals[sum].insert(0,(i,j))
			else:
				diagonals[sum].append((i,j))

	idx_list = []
	for d in diagonals:
		for idx in d:
			for c in range(channels):
				idx_list.append((c,)+idx)

	idx0,idx1,idx2 = zip(*idx_list)
	return idx0,idx1,idx2

class Image2Seq(nn.Module):

	def __init__(self, autoregressive_order, image_shape):
		assert autoregressive_order in {'cwh','whc','zigzag_cs'}
		super(Image2Seq, self).__init__()
		self.autoregressive_order = autoregressive_order
		self.channels = image_shape[0]
		self.height = image_shape[1]
		self.width = image_shape[2]
		if autoregressive_order == 'zigzag_cs':
			self.idx0, self.idx1, self.idx2 = _prep_zigzag_cs(self.channels, self.height, self.width)

	def forward(self, x):
		b, dim = x.shape[0], x.shape[-1]
		l = x.shape[1:-1].numel()
		if self.autoregressive_order == 'whc':

			x = x.permute([1,2,3,0,4])

			x = x.reshape(l,b,dim)

		elif self.autoregressive_order == 'cwh':

			x = x.permute([2,3,1,0,4])

			x = x.reshape(l,b,dim)

		elif self.autoregressive_order == 'zigzag_cs':

			x = x[:, self.idx0, self.idx1, self.idx2, :]

			x = x.permute([1,0,2])

		return x

class Seq2Image(nn.Module):

	def __init__(self, autoregressive_order, image_shape):
		assert autoregressive_order in {'cwh','whc','zigzag_cs'}
		super(Seq2Image, self).__init__()
		self.autoregressive_order = autoregressive_order
		self.channels = image_shape[0]
		self.height = image_shape[1]
		self.width = image_shape[2]
		if autoregressive_order == 'zigzag_cs':
			self.idx0, self.idx1, self.idx2 = _prep_zigzag_cs(self.channels, self.height, self.width)

	def forward(self,x):
		b, dim = x.shape[1], x.shape[2]
		if self.autoregressive_order == 'whc':
			x = x.reshape(self.channels, self.height, self.width, b, dim)

			x = x.permute([3,0,1,2,4])

		elif self.autoregressive_order == 'cwh':
			x = x.reshape(self.height, self.width, self.channels, b,dim)
			x = x.permute([3,2,0,1,4])

		elif self.autoregressive_order == 'zigzag_cs':
			x = x.permute([1,0,2])
			y = torch.empty((x.shape[0],self.channels, self.height, self.width, x.shape[-1]),dtype=x.dtype, device=x.device)
			y[:, self.idx0,self.idx1, self.idx2,:] = x

			x = y

		return x

class DenseTransformer2d(nn.Module):

	def __init__(self, image_shape, output_dim, num_bits,
		autoregressive_order='cwh',d_model=512, nhead=8,
		num_layers=6, dim_feedforward=2048, dropout=0.1,
		activation='relu',kdim=None, vdim=None, 
		attn_bias=True, output_bias=True,
		checkpoint_blocks=False,
		in_lambda = lambda x:x,
		out_lambda = lambda x:x):

		super(DenseTransformer2d, self).__init__()
		self.image_shape = torch.Size(image_shape)
		self.autoregressive_order = autoregressive_order
		self.d_model = d_model
		self.num_layers = num_layers


		self.encode = nn.Sequential(LambdaLayer(in_lambda),
			nn.Embedding(2**num_bits,d_model),
			PositionalEncodeingImage(image_shape=image_shape, embedding_dim=d_model))

		self.im2seq = Image2Seq(autoregressive_order,image_shape)
		self.seq2im = Seq2Image(autoregressive_order, image_shape)
		self.ar_shift = AutoregressiveShift(d_model)

		self.transformer = DenseTransformer(d_model=d_model,
			nhead=nhead,
			num_layers=num_layers,
			dim_feedforward=dim_feedforward,
			dropout=dropout,
			activation=activation,
			kdim=kdim,
			vdim=vdim,
			attn_bias=attn_bias,
			checkpoint_blocks=checkpoint_blocks)

		self.out_linear = nn.Linear(d_model, output_dim, bias=output_bias)
		self.out_lambda = LambdaLayer(out_lambda)

		self._reset_parameters()

	def _reset_parameters(self):

		nn.init.zeros_(self.out_linear.weight)
		if self.out_linear.bias is not None:
			nn.init.zeros_(self.out_linear.bias)

		nn.init.normal_(self.encode._modules['1'].weight, std=0.125/math.sqrt(self.d_model))

	def forward(self,x):

		x = self.encode(x.long())
		x = self.im2seq(x)
		x = self.ar_shift(x)
		x = self.transformer(x)
		x = self.out_linear(x)
		x = self.seq2im(x)
		return self.out_lambda(x)


# layer = DenseTransformer2d(image_shape=(3,32,32), output_dim=64, num_bits=8,
# 		autoregressive_order='cwh',d_model=128, nhead=4,
# 		num_layers=1, dim_feedforward=128, dropout=0.1,
# 		activation='relu',kdim=None, vdim=None, 
# 		attn_bias=True, output_bias=True,
# 		checkpoint_blocks=False,
# 		in_lambda = lambda x:x,
# 		out_lambda = lambda x:x).cuda()
# layer(x.cuda())


class PositionalEncoding1d(nn.Module):

	def __init__(self, size, embedding_dim):
		super(PositionalEncoding1d, self).__init__()
		self.size = size
		self.embedding_dim = embedding_dim
		self.encode_l = nn.Parameter(torch.Tensor(size,1,embedding_dim))
		self.reset_parameters()

	def reset_parameters(self):

		nn.init.normal_(self.encode_l, std=0.125/math.sqrt(self.embedding_dim))

	def forward(self,x):
		return x + self.encode_l

class PositionalEncoding1d_no_embedding(nn.Module):

	def __init__(self, size, embedding_dim):
		super(PositionalEncoding1d_no_embedding, self).__init__()
		self.size = size
		self.embedding_dim = embedding_dim
		self.encode_l = nn.Parameter(torch.Tensor(size,embedding_dim))
		self.reset_parameters()

	def reset_parameters(self):

		nn.init.normal_(self.encode_l, std=0.125/math.sqrt(self.embedding_dim))

	def forward(self,x):
		return x + self.encode_l


class PositionalDenseTransformer(nn.Module):
	def __init__(self, l_input=50, d_input=2, d_output=2, d_model=512, nhead=8,
		num_layers=6, dim_feedforward=512, dropout=0.1,
		activation='gelu',kdim=None, vdim=None,
		attn_bias=True, checkpoint_blocks=False,
		in_lambda= lambda x:x,
		out_lambda = lambda x:x):

		super(PositionalDenseTransformer,self).__init__()

		decoder_layer = DenseTransformerBlock(d_model=d_model,
										nhead=nhead,
										dim_feedforward=dim_feedforward,
										dropout=dropout,
										activation=activation,
										attn_bias=attn_bias,
										checkpoint=checkpoint_blocks)

		self.in_lambda = LambdaLayer(in_lambda)
		self.in_linear = nn.Linear(d_input, d_model)
		self.encode = PositionalEncoding1d(l_input, d_model)
		self.layers = _get_clones(decoder_layer, num_layers)
		self.out_norm = nn.LayerNorm(d_model)
		self.out_linear = nn.Linear(d_model, d_output)
		self.out_lambda = LambdaLayer(out_lambda)

		self.num_layers = num_layers
		self.d_model = d_model
		self.nhead = nhead

		self._reset_parameters()

	def forward(self,x):

		x = self.in_lambda(x)
		x = x.permute(2,0,1)

		x = self.in_linear(x)
		x = self.encode(x)

		for decoder_layer in self.layers:
			x = decoder_layer(x, attn_mask=None, key_padding_mask=None)

		x = self.out_norm(x)
		x = self.out_linear(x)

		x = x.permute(1,2,0)
		x = self.out_lambda(x)

		return x

	def _reset_parameters(self):

		for decoder_layer in self.layers:
			decoder_layer.linear2.weight.data /= math.sqrt(2*self.num_layers)
			decoder_layer.self_attn.out_proj.weight.data /= math.sqrt(2*self.num_layers)


		nn.init.zeros_(self.out_linear.weight)
		if self.out_linear.bias is not None:
			nn.init.zeros_(self.out_linear.bias)


class PositionalDenseTransformer_no_embedding(nn.Module):
	def __init__(self, l_input=50, d_input=2, d_output=2, d_model=512, nhead=8,
		num_layers=6, dim_feedforward=512, dropout=0.1,
		activation='gelu',kdim=None, vdim=None,
		attn_bias=True, checkpoint_blocks=False,
		in_lambda= lambda x:x,
		out_lambda = lambda x:x):

		super(PositionalDenseTransformer_no_embedding,self).__init__()

		decoder_layer = DenseTransformerBlock(d_model=d_model,
										nhead=nhead,
										dim_feedforward=dim_feedforward,
										dropout=dropout,
										activation=activation,
										attn_bias=attn_bias,
										checkpoint=checkpoint_blocks)

		self.in_lambda = LambdaLayer(in_lambda)
		self.in_linear = nn.Linear(d_input, d_model)
		self.encode = PositionalEncoding1d_no_embedding(l_input, d_model)
		self.layers = _get_clones(decoder_layer, num_layers)
		self.out_norm = nn.LayerNorm(d_model)
		self.out_linear = nn.Linear(d_model, d_output)
		self.out_lambda = LambdaLayer(out_lambda)

		self.num_layers = num_layers
		self.d_model = d_model
		self.nhead = nhead

		self._reset_parameters()

	def forward(self,x):

		x = self.in_lambda(x)

		x = self.in_linear(x)
		x = self.encode(x)

		for decoder_layer in self.layers:
			x = decoder_layer(x, attn_mask=None, key_padding_mask=None)

		x = self.out_norm(x)
		x = self.out_linear(x)

		x = self.out_lambda(x)

		return x

	def _reset_parameters(self):

		for decoder_layer in self.layers:
			decoder_layer.linear2.weight.data /= math.sqrt(2*self.num_layers)
			decoder_layer.self_attn.out_proj.weight.data /= math.sqrt(2*self.num_layers)


		nn.init.zeros_(self.out_linear.weight)
		if self.out_linear.bias is not None:
			nn.init.zeros_(self.out_linear.bias)








############### pixel CNN part

def mask_conv2d_spatial(mask_type, height, width):

	mask = torch.ones([1,1,height,width])
	mask[:,:, height//2, width//2+(mask_type == 'B'):] = 0
	mask[:, :, height//2+1:] = 0

	return mask



def mask_channels(mask_type, in_channels, out_channels, data_channels=3):

	in_factor = in_channels // data_channels +1
	out_factor = out_channels // data_channels +1

	base_mask = torch.ones([data_channels,data_channels])
	if mask_type =='A':
		base_mask = base_mask.tril(-1)

	else:
		base_mask = base_mask.tril(0)

	mask_p1 = torch.cat([base_mask]*in_factor, dim=1)
	mask_p2 = torch.cat([mask_p1]*out_factor, dim=0)

	mask = mask_p2[0:out_channels,0:in_channels]
	return mask


def mask_conv2d(mask_type, in_channels, out_channels, height, width, data_channels=3):

	mask = torch.ones([out_channels,in_channels,height,width])
	mask[:,:, height//2, width//2] = mask_channels(mask_type,in_channels,out_channels,data_channels)
	mask[:,:, height//2, width//2 +1:] = 0

	mask[:,:,height//2+1:]=0
	return mask

class _MaskedConv2d(nn.Conv2d):

	def register_mask(self, mask):

		self.register_buffer('mask',mask)

	def forward(self, x):
		self.weight.data *= self.mask 
		return super(_MaskedConv2d,self).forward(x)

class SpatialMaskedConv2d(_MaskedConv2d):

	def __init__(self, *args, mask_type, **kwargs):
		super(SpatialMaskedConv2d,self).__init__(*args, **kwargs)
		assert mask_type in {'A','B'}
		_,_, height, width = self.weight.size()
		mask = mask_conv2d_spatial(mask_type,height,width)
		self.register_mask(mask)


class MaskedConv2d(_MaskedConv2d):

	def __init__(self, *args, mask_type, data_channels=3, **kwargs):
		super(MaskedConv2d,self).__init__(*args, **kwargs)
		assert mask_type in {'A','B'}
		out_channels, in_channels, height, width = self.weight.size()

		mask = mask_conv2d(mask_type, in_channels, out_channels, height, width, data_channels)
		self.register_mask(mask)


class MaskedResidualBlock2d(nn.Module):

	def __init__(self, h, kernel_size=3, data_channels=3):
		super(MaskedResidualBlock2d,self).__init__()

		self.conv1 = MaskedConv2d(2*h,h, kernel_size=1, mask_type='B', data_channels=data_channels)
		self.conv2 = MaskedConv2d(h, h, kernel_size=kernel_size,padding=kernel_size//2,mask_type='B',data_channels=data_channels)
		self.conv3 = MaskedConv2d(h,2*h, kernel_size=1, mask_type='B', data_channels=data_channels)

	def forward(self,x):
		identity = x

		x = self.conv1(F.relu(x))
		x = self.conv2(F.relu(x))
		x = self.conv3(F.relu(x))

		return x + identity


class SpatialMaskedResidualBlock2d(nn.Module):
	def __init__(self, h, kernel_size=3):
		super(SpatialMaskedResidualBlock2d,self).__init__()
		self.conv1 = nn.Conv2d(2*h, h, kernel_size=1)
		self.conv2 = SpatialMaskedConv2d(h,h,kernel_size=kernel_size,padding=kernel_size//2,mask_type='B')
		self.conv3 = nn.Conv2d(h, 2*h, kernel_size=1)

	def forward(self,x):
		identity = x

		x = self.conv1(F.relu(x))
		x = self.conv2(F.relu(x))
		x = self.conv3(F.relu(x))

		return x+identity


class PixelCNN(nn.Sequential):

	def __init__(self, in_channels, num_params, filters=128, num_blocks=15, output_filters=1024, kernel_size=3, kernel_size_in=7, init_transforms=lambda x: 2*x-1):

		layers = [LambdaLayer(init_transforms)]+\
			[MaskedConv2d(in_channels, 2*filters, kernel_size=kernel_size_in,padding=kernel_size_in//2, mask_type='A', data_channels=in_channels)]+\
			[MaskedResidualBlock2d(filters, data_channels=in_channels,kernel_size=kernel_size_in) for _ in range(num_blocks)] +\
			[nn.ReLU(True), MaskedConv2d(2*filters, output_filters, kernel_size=1,mask_type='B',data_channels=in_channels)]+\
			[nn.ReLU(True),MaskedConv2d(output_filters, num_params*in_channels, kernel_size=1, mask_type='B',data_channels=in_channels)]+\
			[ElementwiseParams2d(num_params)]

		super(PixelCNN, self).__init__(*layers)


layer = PixelCNN(in_channels=3, num_params=5, filters=5, num_blocks=2, output_filters=15,kernel_size=3, kernel_size_in=3)



########### MADE part

class MaskedLinear(nn.Linear):

	def __init__(self,
		in_degrees,
		out_features,
		data_features,
		random_mask=False,
		random_seed=None,
		is_output=False,
		data_degrees=None,
		bias=True):

		if is_output:
			assert data_degrees is not None
			assert len(data_degrees) == data_features

		super(MaskedLinear, self).__init__(in_features=len(in_degrees),
			out_features=out_features,
			bias=bias)

		self.out_features = out_features
		self.data_features = data_features
		self.is_output = is_output

		mask, out_degrees = self.get_mask_and_degrees(in_degrees=in_degrees,
			data_degrees=data_degrees,
			random_mask=random_mask,
			random_seed=random_seed)

		self.register_buffer('mask',mask)
		self.register_buffer('degrees',out_degrees)

	@staticmethod
	def get_data_degrees(in_features, random_order=False, random_seed=None):
		if random_order:
			rng = np.random.RandomState(random_seed)
			return torch.from_numpy(rng.permutation(in_features)+1)

		else:
			return torch.arange(1,in_features+1)

	def get_mask_and_degrees(self,in_degrees, data_degrees,random_mask, random_seed):
		if self.is_output:
			out_degrees = repeat_rows(data_degrees, self.out_features//self.data_features)
			mask = (out_degrees[...,None]>in_degrees).float()

		else:
			if random_mask:
				min_in_degree = torch.min(in_degrees).item()
				min_in_degree = min(min_in_degree,self.data_features-1)
				rng = np.random.RandomState(random_seed)
				out_degrees = torch.from_numpy(rng.randint(min_in_degree,
					self.data_features,
					size=[self.out_features]))

			else:
				max_ = max(1,self.data_features-1)
				min_ = min(1,self.data_features-1)
				out_degrees = torch.arange(self.out_features)%max_ + min_

			mask = (out_degrees[...,None] >= in_degrees).float()

		return mask, out_degrees

	def update_mask_and_degrees(self,in_degrees,data_degrees,random_mask,random_seed):

		mask, out_degrees = self.get_mask_and_degrees(in_degrees=in_degrees,
			data_degrees=data_degrees,random_mask=random_mask,random_seed=random_seed)

		self.mask.data.copy_(mask)
		self.degrees.data.copy_(out_degrees)

	def forward(self,x):

		return F.linear(x, self.weight*self.mask, self.bias)


class MADE_Old(nn.Sequential):

	def __init__(self, features, num_params, hidden_features, random_order=False, random_mask=False,
		random_seed=None, activation='relu',dropout_prob=0.0,batch_norm=False):

		layers = []

		data_degrees = MaskedLinear.get_data_degrees(features, random_order=random_order,random_seed=random_seed)
		in_degrees = copy.deepcopy(data_degrees)
		for i,out_features in enumerate(hidden_features):
			layers.append(MaskedLinear(in_degrees=in_degrees,out_features=out_features,
				data_features=features,random_mask=random_mask,random_seed=random_seed+i if random_seed else None,
				is_output=False))


			in_degrees = layers[-1].degrees
			if batch_norm:
				layers.append(nn.BatchNorm1d(out_features))
			layers.append(act_module(activation))
			if dropout_prob >0.0:
				layers.append(nn.Dropout(dropout_prob))


		layers.append(MaskedLinear(in_degrees=in_degrees,
			out_features=features*num_params,data_features=features,random_mask=random_mask,
			random_seed=random_seed,is_output=True,data_degrees=data_degrees))

		layers.append(ElementwiseParams(num_params, mode='sequential'))

		super(MADE_Old, self).__init__(*layers)


class MADE(nn.Sequential):

	def __init__(self, features, num_params, hidden_features, random_order=False, random_mask=False,
		random_seed=None, activation='relu',dropout_prob=0.0,batch_norm=False):

		layers = []

		data_degrees = MaskedLinear.get_data_degrees(features, random_order=random_order,random_seed=random_seed)
		in_degrees = copy.deepcopy(data_degrees)
		for i,out_features in enumerate(hidden_features):
			layers.append(MaskedLinear(in_degrees=in_degrees,out_features=out_features,
				data_features=features,random_mask=random_mask,random_seed=random_seed+i if random_seed else None,
				is_output=False))


			in_degrees = layers[-1].degrees
			if batch_norm:
				layers.append(nn.BatchNorm1d(out_features))
			layers.append(act_module(activation))
			if dropout_prob >0.0:
				layers.append(nn.Dropout(dropout_prob))


		layers.append(MaskedLinear(in_degrees=in_degrees,
			out_features=features*num_params,data_features=features,random_mask=random_mask,
			random_seed=random_seed,is_output=True,data_degrees=data_degrees))

		# layers.append(ElementwiseParams(num_params, mode='sequential'))

		super(MADE, self).__init__(*layers)

class AgnosticMADE(MADE):

	def __init__(self, features, num_params, hidden_features, order_agnostic=True,
		connect_agnostic=True, num_masks=16, activation='relu', dropout_prob=0.0, batch_norm=False):

		self.features = features
		self.order_agnostic = order_agnostic
		self.connect_agnostic = connect_agnostic
		self.num_masks = num_masks
		self.current_mask = 0

		super(AgnosticMADE, self).__init__(features=features,num_params=num_params,
			hidden_features=hidden_features,random_order=order_agnostic,random_mask=connect_agnostic,
			random_seed=self.current_mask,activation=activation,dropout_prob=dropout_prob,
			batch_norm=batch_norm)

	def update_masks(self):
		self.current_mask = (self.current_mask+1)%self.num_masks

		data_degrees = MaskedLinear.get_data_degrees(self.features,random_order=self.order_agnostic,
			random_seed=self.current_mask)

		in_degrees = copy.deepcopy(data_degrees)
		for module in self.modules():
			if isinstance(module, MaskedLinear):
				module.update_mask_and_degrees(in_degrees=in_degrees,data_degrees=data_degrees,
					random_mask=self.connect_agnostic,random_seed=self.current_mask)

				in_degrees = module.degrees

	def forward(self,x):
		if self.num_masks>1: self.update_masks()
		return super(AgnosticMADE,self).forward(x)





### pure transformer

class DecoderOnlyTransformerBlock(nn.Module):

	def __init__(self, d_model, nhead, dim_feedforward=2048,dropout=0.1,activation='relu',
		kdim=None,vdim=None,attn_bias=True, checkpoint=False):
		super(DecoderOnlyTransformerBlock,self).__init__()
		self.self_attn = nn.MultiheadAttention(d_model,nhead, dropout=dropout, kdim=kdim,vdim=vdim,bias=attn_bias)

		self.linear1 = nn.Linear(d_model,dim_feedforward)
		self.dropout = nn.Dropout(dropout)
		self.linear2 = nn.Linear(dim_feedforward,d_model)

		self.norm1 = nn.LayerNorm(d_model)
		self.norm2 = nn.LayerNorm(d_model)
		self.dropout1 = nn.Dropout(dropout)
		self.dropout2 = nn.Dropout(dropout)

		self.activation = act_module(activation)
		self.checkpoint = checkpoint

	def _attn_block(self, x, attn_mask=None, key_padding_mask=None):
		x2 = self.self_attn(x,x,x,attn_mask=attn_mask,key_padding_mask=key_padding_mask)[0]
		x = x + self.dropout1(x2)
		x = self.norm1(x)
		return x

	def _ff_block(self,x,attn_mask=None, key_padding_mask=None):
		x2  = self.linear2(self.dropout(self.activation(self.linear1(x))))
		x = x + self.dropout2(x2)
		x = self.norm2(x)
		return x

	def _forward(self, x,attn_mask=None, key_padding_mask=None):
		x = self._attn_block(x, attn_mask=attn_mask, key_padding_mask=key_padding_mask)
		x = self._ff_block(x)

		return x

	def forward(self, x, attn_mask=None, key_padding_mask=None):
		if not self.checkpoint:
			return self._forward(x,attn_mask, key_padding_mask)
		else:
			x.requires_grad_(True)
			return checkpoint.checkpoint(self._forward, x, attn_mask, key_padding_mask)

class DecoderOnlyTransformer(nn.Module):
	def __init__(self, d_model=512, nhead=8,
		num_layers=6, dim_feedforward=2048, dropout=0.1, activation='relu', kdim=None,
		vdim=None,attn_bias=True, checkpoint_blocks=False):

		super(DecoderOnlyTransformer,self).__init__()

		decoder_layer = DecoderOnlyTransformerBlock(d_model=d_model,
			nhead=nhead,dim_feedforward=dim_feedforward,dropout=dropout,activation=activation,
			kdim=kdim,vdim=vdim,attn_bias=attn_bias,checkpoint=checkpoint_blocks)

		self.layers = _get_clones(decoder_layer, num_layers)
		self.out_norm = nn.LayerNorm(d_model)

		self._reset_parameters()

		self.d_model = d_model
		self.nhead = nhead


	def forward(self, x, key_padding_mask=None):
		if x.size(2) != self.d_model:
			raise RuntimeError('the feature number of src and tgt must be equal to d_model')

		attn_mask = self.generate_square_subsequent_mask(x.shape[0]).to(x.device)

		for decoder_layer in self.layers:
			x = decoder_layer(x,attn_mask=attn_mask,key_padding_mask=key_padding_mask)

		return self.out_norm(x)

	def generate_square_subsequent_mask(self, sz):
		mask = (torch.triu(torch.ones(sz,sz))==1).transpose(0,1)
		mask = mask.float().masked_fill(mask==0, float('-inf')).masked_fill(mask==1,float(0.0))
		return mask 

	def _reset_parameters(self):

		for p in self.parameters():
			if p.dim() > 1:
				nn.init.xavier_uniform_(p)















########################################## 2 distributions
# 	build based on nn.Module
#	serve as stochastic mapping in flow transform
########################################## 

class Distribution(nn.Module):

	def log_prob(self, x):

		raise NotImplementError()

	def sample(self, num_samples):

		raise NotImplementError()

	def sample_with_log_prob(self,num_samples):

		samples = self.sample(num_samples)
		log_prob = self.log_prob(samples)
		return samples, log_prob

	def forward(self, *args, mode, **kwargs):

		if mode == 'log_prob':
			return self.log_prob(*args,**kwargs)
		else:
			raise RuntimeError("Mode {} not supported.".format(mode))


class DiagonalNormal(Distribution):

	def __init__(self, shape):
		super(DiagonalNormal, self).__init__()
		self.shape = torch.Size(shape)
		self.loc = nn.Parameter(torch.zeros.shape)
		self.log_scale = nn.Parameter(torch.zeros(shape))

	def log_prob(self,x):
		log_base = -0.5*torch.log(2*np.pi) - self.log_scale
		log_inner = -0.5*torch.exp(-2*self.log_scale)*((x-self.loc)**2)
		return sum_except_batch(log_base + log_inner)

	def sample(self,x):
		eps = torch.randn(num_samples, *self.shape, device=self.loc.device, dtype=self.loc.dtype)
		return self.loc + self.log_scale.exp()*eps

class ConvNormal2d(DiagonalNormal):
	def __init__(self, shape):
		super(DiagonalNormal, self).__init__()
		assert len(shape) ==3
		self.shape = torch.Size(shape)
		self.loc = torch.nn.Parameter(torch.zeros(1,shape[0],1,1))
		self.log_scale = torch.nn.Parameter(torch.zeros(1,shape[0],1,1))

class ConditionalDistribution(Distribution):

	def log_prob(self,x,context):
		raise NotImplementError()
	def sample(self,context):
		raise NotImplementError()
	def sample_with_log_prob(self, context):
		raise NotImplementError()

class ConditionalMeanNormal(ConditionalDistribution):
    """A multivariate Normal with conditional mean and fixed std."""

    def __init__(self, net, scale=1.0):
        super(ConditionalMeanNormal, self).__init__()
        self.net = net
        self.scale = scale

    def cond_dist(self, context):
        mean = self.net(context)
        return Normal(loc=mean, scale=self.scale)

    def log_prob(self, x, context):
        dist = self.cond_dist(context)
        return sum_except_batch(dist.log_prob(x))

    def sample(self, context):
        dist = self.cond_dist(context)
        return dist.rsample()

    def sample_with_log_prob(self, context):
        dist = self.cond_dist(context)
        z = dist.rsample()
        log_prob = dist.log_prob(z)
        log_prob = sum_except_batch(log_prob)
        return z, log_prob

    def mean(self, context):
        return self.cond_dist(context).mean


class ConditionalMeanStdNormal(ConditionalDistribution):
    """A multivariate Normal with conditional mean and learned std."""

    def __init__(self, net, scale_shape):
        super(ConditionalMeanStdNormal, self).__init__()
        self.net = net
        self.log_scale = nn.Parameter(torch.zeros(scale_shape))

    def cond_dist(self, context):
        mean = self.net(context)
        return Normal(loc=mean, scale=self.log_scale.exp())

    def log_prob(self, x, context):
        dist = self.cond_dist(context)
        return sum_except_batch(dist.log_prob(x))

    def sample(self, context):
        dist = self.cond_dist(context)
        return dist.rsample()

    def sample_with_log_prob(self, context):
        dist = self.cond_dist(context)
        z = dist.rsample()
        log_prob = dist.log_prob(z)
        log_prob = sum_except_batch(log_prob)
        return z, log_prob

    def mean(self, context):
        return self.cond_dist(context).mean


class ConditionalNormal(ConditionalDistribution):
    """A multivariate Normal with conditional mean and log_std."""

    def __init__(self, net, split_dim=-1):
        super(ConditionalNormal, self).__init__()
        self.net = net
        self.split_dim = split_dim

    def cond_dist(self, context):
        params = self.net(context)
        mean, log_std = torch.chunk(params, chunks=2, dim=self.split_dim)
        return Normal(loc=mean, scale=log_std.exp())

    def log_prob(self, x, context):
        dist = self.cond_dist(context)
        return sum_except_batch(dist.log_prob(x))

    def sample(self, context):
        dist = self.cond_dist(context)
        return dist.rsample()

    def sample_with_log_prob(self, context):
        dist = self.cond_dist(context)
        z = dist.rsample()
        log_prob = dist.log_prob(z)
        log_prob = sum_except_batch(log_prob)
        return z, log_prob

    def mean(self, context):
        return self.cond_dist(context).mean

    def mean_stddev(self, context):
        dist = self.cond_dist(context)
        return dist.mean, dist.stddev

class StandardNormal(Distribution):
	def __init__(self,shape):
		super(StandardNormal, self).__init__()
		self.shape = torch.Size(shape)
		self.register_buffer('buffer',torch.zeros(1))

	def log_prob(self,x):
		log_base = -0.5*np.log(2*np.pi)
		log_inner = -0.5*x**2
		return sum_except_batch(log_base+log_inner)

	def sample(self, num_samples):
		return torch.randn(num_samples,*self.shape, device=self.buffer.device, dtype=self.buffer.dtype)

class StandardUniform(Distribution):
	def __init__(self, shape):
		super().__init__()
		self.shape = torch.Size(shape)
		self.register_buffer('zero',torch.zeros(1))
		self.register_buffer('one',torch.ones(1))

	def log_prob(self,x):
		lb = mean_except_batch(x.ge(self.zero).type(self.zero.dtype))
		ub = mean_except_batch(x.le(self.one).type(self.one.dtype))
		return torch.log(lb*ub)

	def sample(self, num_samples):
		return torch.rand((num_samples,)+self.shape,device=self.zero.device, dtype=self.zero.dtype)

class ConditionalBernoulli(ConditionalDistribution):
    """A Bernoulli distribution with conditional logits."""

    def __init__(self, net):
        super(ConditionalBernoulli, self).__init__()
        self.net = net

    def cond_dist(self, context):
        logits = self.net(context)
        return Bernoulli(logits=logits)

    def log_prob(self, x, context):
        dist = self.cond_dist(context)
        return sum_except_batch(dist.log_prob(x.float()))

    def sample(self, context):
        dist = self.cond_dist(context)
        return dist.sample().long()

    def sample_with_log_prob(self, context):
        dist = self.cond_dist(context)
        z = dist.sample()
        log_prob = dist.log_prob(z)
        log_prob = sum_except_batch(log_prob)
        return z.long(), log_prob

    def logits(self, context):
        return self.cond_dist(context).logits

    def probs(self, context):
        return self.cond_dist(context).probs

    def mean(self, context):
        return self.cond_dist(context).mean

    def mode(self, context):
        return (self.cond_dist(context).logits>=0).long()



########################################## 3 invertible transforms
# 	build based on nn.Module
#	contains polular image tensor flow transformation
#	serve as flow transforms
# 	composition of flow transforms make a flow model
########################################## 

class Transform(nn.Module):

	has_inverse = True

	@property
	def bijective(self):
		raise NotImplementError()

	@property
	def stochastic_forward(self):
		raise NotImplementError()

	@property
	def stochastic_inverse(self):
		raise NotImplementedError()
	@property
	def lower_bound(self):
		return self.stochastic_forward

	def forward(self,x):
		raise NotImplementError()

	def inverse(self,z):
		raise NotImplementError()


class StochasticTransform(Transform):

	has_inverse = True
	bijective = False
	stochastic_forward = True
	stochastic_inverse = True

class Bijection(Transform):

	bijective = True
	stochastic_forward = False
	stochastic_inverse = False
	lower_bound = False

class Surjection(Transform):

	bijective = False

	@property
	def stochastic_forward(self):
		raise NotImplementError()

	@property
	def stochastic_inverse(self):
		return not self.stochastic_forward




class VAE(StochasticTransform):

	def __init__(self, decoder, encoder):
		super(VAE,self).__init__()
		self.decoder = decoder
		self.encoder = encoder

	def forward(self,x):
		z, log_qz = self.encoder.sample_with_log_prob(context=x)
		log_px = self.decoder.log_prob(x,context=z)
		return z,log_px-log_qz

	def inverse(self,z):
		return self.decoder.sample(context=z)



class FlattenTransform(Transform):

	def __init__(self,in_shape):
		super(FlattenTransform,self).__init__()
		self.trans = Flatten()
		self.in_shape = in_shape

		has_inverse = True
		bijective = True
		stochastic_forward = False
		stochastic_inverse = False

	def forward(self,x):
		return self.trans(x)

	def inverse(self,x): 
		return x.view(self.in_shape)



class UniformDequantization(Surjection):

	stochastic_forward = True

	def __init__(self, num_bits=8):
		super(UniformDequantization, self).__init__()
		self.num_bits = num_bits
		self.quantization_bins = 2**num_bits
		self.register_buffer('ldj_per_dim',-torch.log(torch.tensor(self.quantization_bins, dtype=torch.float)))

	def _ldj(self, shape):
		batch_size = shape[0]
		num_dims = shape[1:].numel()
		ldj = self.ldj_per_dim*num_dims

		return ldj.repeat(batch_size)


	def forward(self,x):
		u = torch.randn(x.shape,device=self.ldj_per_dim.device,dtype=self.ldj_per_dim.dtype)
		z = (x.type(u.dtype) + u)/self.quantization_bins
		ldj = self._ldj(z.shape)
		return z,ldj

	def inverse(self,z):
		z = self.quantization_bins*z
		return z.floor().clamp(min=0,max=self.quantization_bins-1).long()

class QuantizationBijection(Bijection):

	def forward(self, x):
		z = x/256

		batch_size = x.shape[0]

		ldj = x.shape[1:].numel()*torch.full([batch_size],torch.log(torch.tensor(1/256)),device=x.device,dtype=x.dtype)

		return z,ldj

	def inverse(self, z):

		x = z*256

		return x.float()


class LogisticBijection1d(Bijection):

	def forward(self,x):
		z = torch.logit(x,eps=1e-7)
		_x = torch.clamp(x,1e-7,1-1e-7)
		ldj = sum_except_batch(-torch.log(_x)-torch.log(1-_x))
		return z,ldj

	def inverse(self,z):
		x = scipy.special.expit(z.cpu())
		return x

class SwitchBijection1d(Bijection):

	def forward(self,x):
		a,b = torch.chunk(x,2,1)
		z = torch.cat([b,a],dim=1)
		ldj = torch.zeros((x.shape[0],)).cuda()
		return z,ldj

	def inverse(self,z):
		a,b = torch.chunk(z,2,1)
		x = torch.cat([b,a],dim=1)
		return x


class CouplingBijection(Bijection):

	def __init__(self, coupling_net, split_dim=1, num_condition=None):
		super(CouplingBijection,self).__init__()
		assert split_dim >=1
		self.coupling_net = coupling_net
		self.split_dim = split_dim
		self.num_condition = num_condition

	def split_input(self, input):
		if self.num_condition:
			split_proportions = (self.num_condition, input.shape[self.split_dim]-self.num_condition)
			return torch.split(input, split_proportions, dim=self.split_dim)
		else:
			return torch.chunk(input, 2, dim=self.split_dim)

	def forward(self,x):

		id,x2 = self.split_input(x)
		elementwise_params = self.coupling_net(id)
		z2,ldj = self._elementwise_forward(x2, elementwise_params)
		z = torch.cat([id,z2],dim=self.split_dim)

		return z,ldj

	def inverse(self,z):
		with torch.no_grad():
			id,z2 = self.split_input(z)
			elementwise_params = self.coupling_net(id)
			x2 = self._elementwise_inverse(z2,elementwise_params)
			x = torch.cat([id,x2],dim=self.split_dim)
		return x

	def _output_dim_mutiplier(self):
		raise NotImplementError()

	def _elementwise_forward(self,x,elementwise_params):
		raise NotImplementError()

	def _elementwise_inverse(self,z,elementwise_params):
		raise NotImplementError()


class AdditiveCouplingBijection(CouplingBijection):

	def _output_dim_mutiplier(self):
		return 1

	def _elementwise_forward(self,x,elementwise_params):
		return x + elementwise_params, torch.zeros(x.shape[0],device=x.device, dtype=x.dtype)

	def _elementwise_inverse(self,z,elementwise_params):
		return z - elementwise_params

class AffineCouplingBijection(CouplingBijection):

	def __init__(self, coupling_net, split_dim=1, num_condition=None, scale_fn=lambda s:torch.exp(s)):
		super(AffineCouplingBijection,self).__init__(coupling_net=coupling_net, split_dim=split_dim,num_condition=num_condition)
		assert callable(scale_fn)
		self.scale_fn = scale_fn

	def _output_dim_mutiplier(self):
		return 2

	def _elementwise_forward(self,x, elementwise_params):
		assert elementwise_params.shape[-1] == self._output_dim_mutiplier()
		unconstrained_scale, shift = self._unconstrained_scale_and_shift(elementwise_params)
		scale = self.scale_fn(unconstrained_scale)
		z = scale*x + shift
		ldj = sum_except_batch(torch.log(scale))
		return z,ldj

	def _elementwise_inverse(self,z,elementwise_params):
		assert elementwise_params.shape[-1] == self._output_dim_mutiplier()
		unconstrained_scale, shift = self._unconstrained_scale_and_shift(elementwise_params)
		scale = self.scale_fn(unconstrained_scale)
		x = (z-shift)/scale
		return x

	def _unconstrained_scale_and_shift(self, elementwise_params):
		unconstrained_scale = elementwise_params[..., 0]
		shift = elementwise_params[...,1]
		return unconstrained_scale, shift


class AffineCouplingBijection1d(CouplingBijection):

	def __init__(self, coupling_net, split_dim=1, num_condition=None, scale_fn=lambda s:torch.exp(s),split_type='half'):
		super(AffineCouplingBijection1d,self).__init__(coupling_net=coupling_net, split_dim=split_dim,num_condition=num_condition)
		assert callable(scale_fn)
		self.scale_fn = scale_fn
		

		input_dim = self.coupling_net.input_size
		all_dim = self.coupling_net.output_size

		if split_type == 'half':
			self.coupling_index = list(range(2*int(self.coupling_net.input_size)))[:int(self.coupling_net.input_size)]
			self.no_coupling_index = list(range(2*int(self.coupling_net.input_size)))[int(self.coupling_net.input_size):]
		elif split_type == 'random':
			self.coupling_index = np.sort(np.random.choice(np.arange(all_dim),input_dim,replace=False))
			self.no_coupling_index = []
			for i in np.arange(all_dim):
				if i not in self.coupling_index:
					self.no_coupling_index.append(i)

	def _output_dim_mutiplier(self):
		return 2

	def split_input(self, input):
		if self.num_condition:
			split_proportions = (self.num_condition, input.shape[self.split_dim]-self.num_condition)
			return torch.split(input, split_proportions, dim=self.split_dim)
		else:
			if self.coupling_index is not None:
				id = input[:,self.coupling_index]
				x2 = input[:,self.no_coupling_index]
				return id,x2
			else:
				return torch.chunk(input, 2, dim=self.split_dim)

	def forward(self,x):

		id,x2 = self.split_input(x)
		elementwise_params = self.coupling_net(id)
		z2,ldj = self._elementwise_forward(x2, elementwise_params)
		z = torch.zeros(x.shape).cuda()
		z[:,self.coupling_index] = id
		z[:,self.no_coupling_index] = z2

		return z,ldj

	def inverse(self,z):
		with torch.no_grad():
			id,z2 = self.split_input(z)
			elementwise_params = self.coupling_net(id)
			x2 = self._elementwise_inverse(z2,elementwise_params)
			x = torch.zeros(z.shape).cuda()
			x[:,self.coupling_index] = id
			x[:,self.no_coupling_index] = x2

		return x

	def _elementwise_forward(self,x, elementwise_params):
		# assert elementwise_params.shape[-1] == self._output_dim_mutiplier()
		unconstrained_scale, shift = self._unconstrained_scale_and_shift(elementwise_params)
		unconstrained_scale = torch.clamp(unconstrained_scale,-2,2)
		scale = self.scale_fn(unconstrained_scale)

		z = scale*x + shift
		ldj = torch.sum(unconstrained_scale,dim=1)
		return z,ldj

	def _elementwise_inverse(self,z,elementwise_params):
		# assert elementwise_params.shape[-1] == self._output_dim_mutiplier()
		unconstrained_scale, shift = self._unconstrained_scale_and_shift(elementwise_params)
		unconstrained_scale = torch.clamp(unconstrained_scale,-2,2)

		scale = self.scale_fn(unconstrained_scale)
		x = (z-shift)/scale
		return x

	def _unconstrained_scale_and_shift(self, elementwise_params):
		unconstrained_scale,shift = torch.chunk(elementwise_params,2,self.split_dim)
		return unconstrained_scale, shift




class AutoregressiveBijection(Bijection):

	def __init__(self, autoregressive_net, autoregressive_order='ltr'):
		super(AutoregressiveBijection, self).__init__()
		assert isinstance(autoregressive_order,str) or isinstance(autoregressive_order,Iterable)
		assert autoregressive_order in {'ltr'}

		self.autoregressive_net = autoregressive_net
		self.autoregressive_order = autoregressive_order

	def forward(self,x):
		elementwise_params = self.autoregressive_net(x)
		z, ldj = self._elementwise_forward(x, elementwise_params)
		return z,ldj

	def inverse(self,z):
		with torch.no_grad():
			if self.autoregressive_order == 'ltr': return self._inverse_ltr(z)

	def _inverse_ltr(self,z):
		x = torch.zeros_like(z)
		for d in range(x.shape[1]):
			elementwise_params  = self.autoregressive_net(x)
			x[:,d] = self._elementwise_inverse(z[:,d],elementwise_params[:,d])

		return x

	def _output_dim_multiplier(self):
		raise NotImplementError()

	def _elementwise_forward(self, x, elementwise_params):
		raise NotImplementError()

	def _elementwise_inverse(self, z, elementwise_params):
		raise NotImplementError()


class AdditiveAutoregressiveBijection(AutoregressiveBijection):

	def _output_dim_multiplier(self):
		return 1

	def _elementwise_forward(self, x, elementwise_params):
		return x + elementwise_params, torch.zeros(x.shape[0], device=x.device, dtype=x.dtype)

	def _elementwise_inverse(self,z, elementwise_params):
		return z - elementwise_params



class AffineAutoregressiveBijection(AutoregressiveBijection):

	def __init__(self, autoregressive_net, autoregressive_order='ltr', scale_fn=lambda s:torch.exp(s)):
		super(AffineAutoregressiveBijection, self).__init__(autoregressive_net=autoregressive_net,autoregressive_order=autoregressive_order)
		assert callable(scale_fn)
		self.scale_fn = scale_fn

	def _output_dim_multiplier(self):
		return 2

	def _elementwise_forward(self,x, elementwise_params):
		assert elementwise_params.shape[-1] == self._output_dim_multiplier()
		unconstrained_scale, shift = self._unconstrained_scale_and_shift(elementwise_params)
		scale = self.scale_fn(unconstrained_scale)
		z = scale*x + shift

		ldj = sum_except_batch(torch.log(scale))

		return z,ldj

	def _elementwise_inverse(self, z, elementwise_params):
		assert elementwise_params.shape[-1] == self._output_dim_multiplier()
		unconstrained_scale,shift = self._unconstrained_scale_and_shift(elementwise_params)
		scale = self.scale_fn(unconstrained_scale)
		x = (z-shift)/scale
		return x

	def _unconstrained_scale_and_shift(self, elementwise_params):

		unconstrained_scale = elementwise_params[...,0]
		shift = elementwise_params[...,1]
		return unconstrained_scale,shift


class AffineAutoregressiveBijection1d(AutoregressiveBijection):

	def __init__(self, autoregressive_net, autoregressive_order='ltr', scale_fn=lambda s:torch.exp(s)):
		super(AffineAutoregressiveBijection1d, self).__init__(autoregressive_net=autoregressive_net,autoregressive_order=autoregressive_order)
		assert callable(scale_fn)
		self.scale_fn = scale_fn

	def _output_dim_multiplier(self):
		return 2

	def _elementwise_forward(self,x, elementwise_params):
		assert elementwise_params.shape[-1] == self._output_dim_multiplier()
		unconstrained_scale, shift = self._unconstrained_scale_and_shift(elementwise_params)
		scale = self.scale_fn(unconstrained_scale)
		z = scale*x + shift

		ldj = sum_except_batch(torch.log(scale))

		return z,ldj

	def _elementwise_inverse(self, z, elementwise_params):
		assert elementwise_params.shape[-1] == self._output_dim_multiplier()
		unconstrained_scale,shift = self._unconstrained_scale_and_shift(elementwise_params)
		scale = self.scale_fn(unconstrained_scale)
		x = (z-shift)/scale
		return x

	def _unconstrained_scale_and_shift(self, elementwise_params):

		unconstrained_scale = elementwise_params[...,0]
		shift = elementwise_params[...,1]
		return unconstrained_scale,shift


# net = MADE_Old(features=3072, num_params=2, hidden_features=[4], random_order=False, random_mask=False,
# 		random_seed=None, activation='relu',dropout_prob=0.0,batch_norm=False)

# layer = AffineAutoregressiveBijection(net)


class AutoregressiveBijection2d(Bijection):

	def __init__(self, autoregressive_net, autoregressive_order='raster_cwh'):
		super(AutoregressiveBijection2d,self).__init__()
		assert isinstance(autoregressive_order,str) or isinstance(autoregressive_order, Iterable)
		assert autoregressive_order in {'raster_cwh','raster_wh'}
		self.autoregressive_net = autoregressive_net
		self.autoregressive_order = autoregressive_order

	def forward(self,x):
		elementwise_params = self.autoregressive_net(x)
		z,ldj = self._elementwise_forward(x,elementwise_params)
		return z,ldj

	def inverse(self,z):
		with torch.no_grad:
			if self.autoregressive_order == 'raster_cwh': return self._inverse_raster_cwh(z)
			if self.autoregressive_order == 'raster_wh': return self._inverse_raster_wh(z)

	def _inverse_raster_cwh(self,z):
		x = torch.zeros_like(z)
		for h in range(x.shape[2]):
			for w in range(x.shape[3]):
				for c in range(x.shape[1]):
					elementwise_params = self.autoregressive_net(x)
					x[:,c,h,w] = self._elementwise_inverse(z[:,c,h,w], elementwise_params[:,c,h,w])

		return x

	def _inverse_raster_wh(self,z):
		x = torch.zeros_like(z)
		for h in range(x.shape[2]):
			for w in range(x.shape[3]):
				elementwise_params = self.autoregressive_net(x)
				x[:,:,h,w] = self._elementwise_inverse(z[:,:,h,w], elementwise_params[:,:,h,w])
		return x

	def _output_dim_multiplier(self):
		raise NotImplementError()

	def _elementwise_forward(self,x,elementwise_params):
		raise NotImplementError()

	def _elementwise_inverse(self,z,elementwise_params):
		raise NotImplementError()


class AdditiveAutoregressiveBijection2d(AutoregressiveBijection2d):

	def _output_dim_multiplier(self):
		return 1

	def _elementwise_forward(self, x, elementwise_params):
		return x + elementwise_params, torch.zeros(x.shape[0],device=x.device, dtype=x.dtype)

	def _elementwise_inverse(self, z, elementwise_params):
		return z - elementwise_params


class AffineAutoregressiveBijection2d(AutoregressiveBijection2d):

	def __init__(self, autoregressive_net, autoregressive_order='raster_cwh',scale_fn=lambda s:torch.exp(s)):
		super(AffineAutoregressiveBijection2d,self).__init__(autoregressive_net=autoregressive_net,autoregressive_order=autoregressive_order)
		assert callable(scale_fn)
		self.scale_fn = scale_fn

	def _output_dim_multiplier(self):
		return 2

	def _elementwise_forward(self, x, elementwise_params):
		assert elementwise_params.shape[-1] == self._output_dim_multiplier()
		unconstrained_scale,shift = self._unconstrained_scale_and_shift(elementwise_params)
		scale = self.scale_fn(unconstrained_scale)
		z = scale*x + shift
		ldj = sum_except_batch(torch.log(scale))

		return z,ldj

	def _elementwise_inverse(self, z, elementwise_params):
		assert elementwise_params.shape[-1] == self._output_dim_multiplier()
		unconstrained_scale,shift = self._unconstrained_scale_and_shift(elementwise_params)
		scale = self.scale_fn(unconstrained_scale)
		x = (z-shift)/scale
		return x

	def _unconstrained_scale_and_shift(self, elementwise_params):
		unconstrained_scale = elementwise_params[...,0]
		shift = elementwise_params[...,1]
		return unconstrained_scale,shift


class AutoregressiveBijection(Bijection):

	def __init__(self, autoregressive_net, autoregressive_order='ltr'):
		super(AutoregressiveBijection, self).__init__()
		assert isinstance(autoregressive_order,str) or isinstance(autoregressive_order,Iterable)
		assert autoregressive_order in {'ltr'}

		self.autoregressive_net = autoregressive_net
		self.autoregressive_order = autoregressive_order

	def forward(self,x):
		elementwise_params = self.autoregressive_net(x)
		z, ldj = self._elementwise_forward(x, elementwise_params)
		return z,ldj

	def inverse(self,z):
		with torch.no_grad():
			if self.autoregressive_order == 'ltr': return self._inverse_ltr(z)

	def _inverse_ltr(self,z):
		x = torch.zeros_like(z)
		for d in range(x.shape[1]):
			elementwise_params  = self.autoregressive_net(x)
			x[:,d] = self._elementwise_inverse(z[:,d],elementwise_params[:,d])

		return x

	def _output_dim_multiplier(self):
		raise NotImplementError()

	def _elementwise_forward(self, x, elementwise_params):
		raise NotImplementError()

	def _elementwise_inverse(self, z, elementwise_params):
		raise NotImplementError()

class ResidualBijection(Bijection):

	def __init__(self, residual_net,approx_trace_order, n_inverse_iters, approx_trace_method='precise'):
		super(ResidualBijection,self).__init__()
		assert isinstance(approx_trace_method,str) or isinstance(approx_trace_method, Iterable)
		assert approx_trace_method in {'russia_rollet','truncation','precise'}

		self.residual_net = residual_net
		self.approx_trace_method = approx_trace_method

		if self.approx_trace_method == 'russia_rollet':
			assert approx_trace_order is not None, 'russia_rollet trace approximation no need for approx_trace_order'

		self.approx_trace_order = approx_trace_order
		self.n_inverse_iters = n_inverse_iters

	def forward(self,x):
		z, ldj = self._elementwise_forward(x)
		return z,ldj

	def inverse(self,z):
		with torch.no_grad():
			return self._elementwise_inverse(z)

	def _output_dim_multiplier(self):
		raise NotImplementError()

	def _elementwise_forward(self, x, elementwise_params):
		raise NotImplementError()

	def _elementwise_inverse(self, z, elementwise_params):
		raise NotImplementError()


class AffineAutoregressiveBijection1d(AutoregressiveBijection):

	def __init__(self, autoregressive_net, autoregressive_order='ltr', scale_fn=lambda s:torch.exp(s)):
		super(AffineAutoregressiveBijection1d, self).__init__(autoregressive_net=autoregressive_net,autoregressive_order=autoregressive_order)
		assert callable(scale_fn)
		self.scale_fn = scale_fn

	def _output_dim_multiplier(self):
		return 2

	def _elementwise_forward(self,x, elementwise_params):
		assert elementwise_params.shape[-1] == self._output_dim_multiplier()
		unconstrained_scale, shift = self._unconstrained_scale_and_shift(elementwise_params)
		scale = self.scale_fn(unconstrained_scale)
		z = scale*x + shift

		ldj = sum_except_batch(torch.log(scale))

		return z,ldj

	def _elementwise_inverse(self, z, elementwise_params):
		assert elementwise_params.shape[-1] == self._output_dim_multiplier()
		unconstrained_scale,shift = self._unconstrained_scale_and_shift(elementwise_params)
		scale = self.scale_fn(unconstrained_scale)
		x = (z-shift)/scale
		return x

	def _unconstrained_scale_and_shift(self, elementwise_params):

		unconstrained_scale = elementwise_params[...,0]
		shift = elementwise_params[...,1]
		return unconstrained_scale,shift

 

class ResidualBijection1d(ResidualBijection):

	def __init__(self, residual_net,input_size, approx_trace_order=10,n_inverse_iters=10,approx_trace_method='precise'):
		super(ResidualBijection1d, self).__init__(residual_net=residual_net,
			approx_trace_method=approx_trace_method,
			approx_trace_order=approx_trace_order,
			n_inverse_iters=n_inverse_iters)

		self.input_size = input_size
		self.get_z = lambda x: x+self.residual_net(x)

	def _output_dim_multiplier(self):
		return 1

	def _elementwise_forward(self, x):

		jacobs = []

		for b_idx in range(x.shape[0]):

			xx = x[b_idx].unsqueeze(0).detach()
			# xx = torch.zeros_like(x[b_idx].unsqueeze(0)).to(x.device)

			# xx.data = x[b_idx].unsqueeze(0).data

			jacob = torch.autograd.functional.jacobian(self.get_z,xx,create_graph=True).squeeze().unsqueeze(0)

			jacobs.append(jacob)

		ldj = torch.logdet(torch.cat(jacobs,dim=0))

		z = x + self.residual_net(x)

		return z,ldj

	def _elementwise_inverse(self, z):

		xx = z

		for _ in range(self.n_inverse_iters):

			xx = z - self.residual_net(xx)
		 
		return xx





# def get_z(x):

# 	return x + residual_net(x)


# residual_net = MLP(int(3072), 3072,hidden_units=[100,100],
#                                 activation='relu',
#                                 in_lambda=None).to(x.device)

# ldj = torch.zeros(x.shape[0]).to(x.device)

# get_z = lambda x: x + residual_net(x)

# jacobs = []

# for b_idx in range(x.shape[0]):

# 	xx = torch.zeros_like(x[b_idx].unsqueeze(0)).to(x.device)

# 	xx.data = x[b_idx].unsqueeze(0).data

# 	xx.requires_grad = True

# 	jacob = torch.autograd.functional.jacobian(get_z,xx,create_graph=True).squeeze().unsqueeze(0)

# 	jacobs.append(jacob)

# 	# ldj[b_idx] = torch.logdet(jacob)

# torch.logdet(torch.cat(jacobs,dim=0))

# z = x + residual_net(x)

# jacob = torch.autograd.functional.jacobian(get_z,xx,create_graph=True).reshape(x.shape[1],x.shape[1])


# torch.autograd.functional.jacobian(get_z,x,create_graph=True).reshape(x.shape[1],x.shape[1])


# xx = z

# for _ in range(10):

# 	xx = z - residual_net(xx)




# residual_net = MLP(int(3072), 3072,hidden_units=[100,100],
#                                 activation='relu',
#                                 in_lambda=None).to(x.device)








# A_dim = A.shape[0]
# B = torch.eye(A_dim)
# C = torch.zeros_like(A)
# ind = 1
# for k in range(approx_trace_order):
# 	B = ind*B@A/(k+1)
# 	ind = -1*ind
# 	C = C + B







class _ActNormBijection(Bijection):

	def __init__(self, num_features, data_dep_init=True, eps=1e-6):
		super(_ActNormBijection,self).__init__()
		self.num_features = num_features
		self.data_dep_init = data_dep_init
		self.eps = eps

		self.register_buffer('initialized',torch.zeros(1) if data_dep_init else torch.ones(1))
		self.register_params()

	def data_init(self,x):
		self.initialized += 1.
		with torch.no_grad():
			x_mean, x_std = self.compute_stats(x)
			self.shift.data = x_mean
			self.log_scale.data = torch.log(x_std + self.eps)

	def forward(self,x):
		if self.training and not self.initialized: self.data_init(x)
		z = (x - self.shift)*torch.exp(-self.log_scale)
		ldj = torch.sum(-self.log_scale).expand([x.shape[0]])*self.ldj_multiplier(x)
		return z,ldj

	def inverse(self,z):
		return self.shift + z*torch.exp(self.log_scale)

	def register_params(self):
		raise NotImplementError()

	def compute_stats(self,x):
		raise NotImplementError()

	def ldj_multiplier(self,x):
		raise NotImplementError()


class ActNormBijection(_ActNormBijection):

	def register_params(self):

		self.register_parameter('shift',nn.Parameter(torch.zeros(1,self.num_features)))
		self.register_parameter('log_scale',nn.Parameter(torch.zeros(1,self.num_features)))

	def compute_stats(self,x):

		x_mean = torch.mean(x, dim=0, keepdim=True)
		x_std = torch.std(x, dim=0, keepdim=True)

		return x_mean, x_std

	def ldj_multiplier(self,x):

		return 1

class ActNormBijection1d(_ActNormBijection):

	def register_params(self):
		self.register_parameter('shift', nn.Parameter(torch.zeros(1,self.num_features,1)))
		self.register_parameter('log_scale', nn.Parameter(torch.zeros(1, self.num_features,1)))

	def compute_stats(self,x):

		x_mean = torch.mean(x, dim=[0,2], keepdim=True)
		x_std = torch.std(x, dim=[0,2], keepdim=True)

		return x_mean, x_std

	def ldj_multiplier(self,x):
		return x.shape[2]


class ActNormBijection2d(_ActNormBijection):

	def register_params(self):

		self.register_parameter('shift',nn.Parameter(torch.zeros(1,self.num_features,1,1)))
		self.register_parameter('log_scale',nn.Parameter(torch.zeros(1,self.num_features, 1,1)))
	
	def compute_stats(self,x):

		x_mean = torch.mean(x,dim=[0,2,3],keepdim=True)
		x_std = torch.std(x,dim=[0,2,3],keepdim=True)

		return x_mean, x_std

	def ldj_multiplier(self,x):
		return x.shape[2:4].numel()



class Conv1x1(Bijection):

	def __init__(self, num_channels, orthogonal_init=True, slogdet_cpu=True):
		super(Conv1x1, self).__init__()

		self.num_channels = num_channels
		self.slogdet_cpu = slogdet_cpu
		self.weight = nn.Parameter(torch.Tensor(num_channels,num_channels))
		self.reset_parameters(orthogonal_init)

	def reset_parameters(self, orthogonal_init):

		self.orthogonal_init = orthogonal_init

		if self.orthogonal_init:
			nn.init.orthogonal_(self.weight)
		else:
			bound = 1.0/ np.sqrt(self.num_channels)
			nn.init.uniform_(self.weight, -bound, bound)

	def _conv(self,weight, v):
		_,channel, *features = v.shape
		n_feature_dims = len(features)

		fill = (1,)*n_feature_dims
		weight = weight.view(channel, channel, *fill)

		if n_feature_dims == 1:
			return F.conv1d(v,weight)
		elif n_feature_dims == 2:
			return F.conv2d(v,weight)
		elif n_feature_dims == 3:
			return F.conv3d(v,weight)
		else:
			raise ValueError(f'Got {n_feature_dims}d tensor, expected 1d, 2d, or 3d')

	def _logdet(self, x_shape):
		b,c,*dims = x_shape
		if self.slogdet_cpu:
			_, ldj_per_pixel = torch.slogdet(self.weight.to('cpu'))
		else:
			_,ldj_per_pixel = torch.slogdet(self.weight)
		ldj = ldj_per_pixel * reduce(mul, dims)
		return ldj.expand([b]).to(self.weight.device)

	def forward(self,x):
		z = self._conv(self.weight,x)
		ldj = self._logdet(x.shape)

		return z,ldj

	def inverse(self,z):
		weight_inv = torch.inverse(self.weight)
		x = self._conv(weight_inv, z)
		return x





class ScalarAffineBijection(Bijection):

	def __init__(self, shift=None, scale=None):
		super(ScalarAffineBijection, self).__init__()
		assert isinstance(shift, float) or shift is None, 'shift must be a float or None'
		assert isinstance(scale, float) or scale is NOne, 'scale must be a float or None'

		if shift is None and scale is None:
			raise ValueError('At Least one of scale and shift must be provided.')
		if scale == 0:
			raise ValueError('Scale can not be zero.')

		self.register_buffer('_shift',torch.tensor(shift if (shift is not None) else 0.))
		self.register_buffer('_scale',torch.tensor(scale if (scale is not None) else 1.))

	@property 
	def _log_scale(self):
		return torch.log(torch.abs(self._scale))

	def forward(self, x):
		batch_size = x.shape[0]
		num_dims = x.shape[1:].numel()
		z = x*self._scale + self._shift
		ldj = torch.full([batch_size], self._log_scale*num_dims, device=x.device, dtype=x.dtype)

		return z, ldj

	def inverse(self,z):
		batch_size = z.shape[0]
		num_dims = z.shape[1:].numel()
		x = (z - self._shift)/self._scale

		return x

class Permute(Bijection):

	def __init__(self, permutation, dim=1):
		super(Permute, self).__init__()
		assert isinstance(dim, int), 'dim must be an integer'
		assert dim >= 1, 'dim must be >= 1 (0 corresponding to batch dimension)'
		assert isinstance(permutation, torch.Tensor) or isinstance(permutation, Iterable), 'permutation must be a torch.Tensor or Iterable'
		if isinstance(permutation, torch.Tensor):
			assert permutation.ndimension() == 1, 'permutation must be an 1D tensor, but was of shape {}'.format(permutation.shape)
		else:
			permutation = torch.tensor(permutation)

		self.dim = dim
		self.register_buffer('permutation',permutation)


	@property
	def inverse_permutation(self):
		return torch.argsort(self.permutation)
	def forward(self,x):
		return torch.index_select(x, self.dim, self.permutation), torch.zeros(x.shape[0],device=x.device, dtype=x.dtype)

	def inverse(self,z):
		return torch.index_select(z, self.dim, self.inverse_permutation)


class Shuffle(Permute):

	def __init__(self, dim_size, dim=1):
		super(Shuffle, self).__init__(torch.randperm(dim_size),dim)

class Reverse(Permute):

	def __init__(self, dim_size, dim=1):
		super(Reverse, self).__init__(torch.arange(dim_size-1, -1,-1),dim)


class PermuteAxes(Bijection):

	def __init__(self, permutation):
		super(PermuteAxes, self).__init__()
		assert isinstance(permutation, Iterable), 'permutation must be an Iterable'
		assert permutation[0] == 0, 'First element of permutation must be 0 (such that batch dimension stays intact)'

		self.permutation = permutation
		self.inverse_permutation = torch.argsort(torch.tensor(self.permutation)).tolist()

	def forward(self, x):
		z = x.permute(self.permutation).contiguous()
		ldj = torch.zeros((x.shape[0],),device=x.device, dtype=x.dtype)
		return z,ldj

	def inverse(self,z):
		x = z.permute(self.inverse_permutation).contiguous()
		return x

class StochasticPermutation(StochasticTransform):

	def __init__(self, dim=1):
		super(StochasticPermutation, self).__init__()
		self.register_buffer('buffer',torch.zeros(1))
		self.dim = dim

	def forward(self,x):
		rand = torch.rand(x.shape[0], x.shape[self.dim], device=x.device)
		permutation = rand.argsort(dim=1)

		for d in range(1, self.dim):
			permutation = permutation.unsqueeze(1)

		for d in range(self.dim +1, x.dim()):
			permutation = permutation.unsqueeze(-1)

		permutation = permutation.expand_as(x)
		z = torch.gather(x, self.dim, permutation)
		ldj = self.buffer.new_zeros(x.shape[0])
		return z,ldj

	def inverse(self,z):
		rand = torch.rand(z.shape[0], z.shape[self.dim], device=z.device)
		permutation = rand.argsort(dim=1)
		for d in range(1, self.dim):
			permutation = permutation.unsqueeze(1)
		for d in range(self.dim+1, z.dim()):
			permutation = permutation.unsqueeze(-1)
		permutation = permutation.expand_as(z)
		x = torch.gather(z, self.dim, permutation)

		return x



class Reshape(Bijection):

	def __init__(self, input_shape, output_shape):
		super(Reshape, self).__init__()
		self.input_shape = torch.Size(input_shape)
		self.output_shape = torch.Size(output_shape)
		assert self.input_shape.numel() == self.output_shape.numel()

	def forward(self,x):
		batch_size = (x.shape[0],)
		z = x.reshape(batch_size,+ self.output_shape)
		ldj = torch.zeros(batch_size, device=x.device, dtype=x.dtype)

		return z,ldj

	def inverse(self,z):
		batch_size = (z.shape[0],)
		x = z.reshape(batch_size + self.input_shape)
		return x


class Rotate(Bijection):

	def __init__(self, degrees, dim1, dim2):

		super(Rotate, self).__init__()
		assert isinstance(degrees, int), 'degrees must be an integer'
		assert isinstance(dim1, int), 'dim1 must be an integer'
		assert isinstance(dim2, int), 'dim2 must be an integer'

		assert degrees in {90,180,270}
		assert dim1 !=0
		assert dim2 != 0
		assert dim1 != dim2

		self.degrees = degrees
		self.dim1 = dim1
		self.dim2 = dim2

	def _rotate90(self,x):

		return x.transpose(self.dim1, self.dim2).flio(self.dim1)

	def _rotate90_inv(self,z):
		return z.flip(self.dim1).transpose(self.dim1,self.dim2)

	def _rotate180(self,x):
		return x.flip(self.dim1).flip(self.dim2)

	def _rotate180_inv(self,z):
		return z.flip(self.dim2).flip(self.dim1)

	def _rotate270(self,x):
		return x.transpose(self.dim1, self.dim2).flip(self.dim2)

	def _rotate270_inv(self,z):
		return z.flip(self.dim2).transpose(self.dim1,self.dim2)

	def forward(self,x):

		if self.degrees == 90: z = self._rotate90(x)
		elif self.degrees == 180: z = self._rotate180(x)
		elif self.degrees == 270: z = self._rotate270(x)

		ldj = torch.zeros(x.shape[0], device=x.device, dtype=x.dtype)

		return x, torch.zeros(x.shape[0], device=x.device, dtype=x.dtype)

	def inverse(self,z):
		if self.degrees == 90: x = self._rotate90_inv(z)
		elif self.degrees == 180: x = self._rotate180_inv(z)
		elif self.degrees == 270: x = self._rotate270_inv(z)
		return z








class Augment(Surjection):

	stochastic_forward = True

	def __init__(self, encoder, x_size, split_dim=1):
		super(Augment, self).__init__()
		assert split_dim >= 1
		self.encoder = encoder
		self.split_dim = split_dim
		self.x_size = x_size
		self.cond = isinstance(self.encoder, ConditionalDistribution)

	def split_z(self,z):
		split_proportions = (self.x_size, z.shape[self.split_dim]-self.x_size)
		return torch.split(z, split_proportions, dim=self.split_dim)

	def forward(self,x):
		if self.cond: z2, logqz2 = self.encoder.sample_with_log_prob(context=x)
		else: z2,logqz2=self.encoder.sample_with_log_prob(num_samples=x.shape[0])

		z = torch.cat([x,z2],dim=self.split_dim)
		ldj = -logqz2
		return z,ldj 

	def inverse(self, z):
		x, z2 = self.split_z(z)
		return x



class Slice(Surjection):

	stochastic_forward = False

	def __init__(self, decoder, num_keep, dim=1):
		super(Slice, self).__init__()
		assert dim >= 1
		self.decoder = decoder
		self.dim = dim 
		self.num_keep = num_keep
		self.cond = isinstance(self.decoder, ConditionalDistribution)

	def split_input(self, input):
		split_proportions = (self.num_keep, input.shape[self.dim]-self.num_keep)
		return torch.split(input, split_proportions, dim=self.dim)

	def forward(self,x):
		z, x2 = self.split_input(x)
		if self.cond: ldj = self.decoder.log_prob(x2, context=z)
		else: ldj = self.decoder.log_prob(x2)
		return z, ldj

	def inverse(self,z):
		if self.cond: x2 = self.decoder.sample(context=z)
		else: x2 = self.decoder.sample(num_samples=z.shape[0])
		x = torch.cat([z,x2],dim=self.dim)
		return x

class Squeeze2d(Bijection):

	def __init__(self,factor=2, ordered=False):
		super(Squeeze2d,self).__init__()
		assert isinstance(factor, int)
		assert factor >1
		self.factor = factor
		self.ordered = ordered

	def _squeeze(self,x):
		assert len(x.shape) == 4, 'Dimension should be 4, but was {}'.format(len(x.shape))
		batch_size,c,h,w = x.shape
		assert h % self.factor == 0, 'h = {} not multiplicative of {}'.format(h, self.factor)
		assert w % self.factor == 0, 'w = {} not multiplicative of {}'.format(w, self.factor)
		t = x.view(batch_size, c, h//self.factor, self.factor, w//self.factor, self.factor)
		if not self.ordered:
			t = t.permute(0,1,3,5,2,4).contiguous()
		else:
			t = t.permute(0,3,5,1,2,4).contiguous()

		z = t.view(batch_size, c*self.factor**2, h//self.factor, w//self.factor)
		return z


	def _unsqueeze(self, z):
		assert len(z.shape) == 4, 'Dimension should be 4, but was {}'.format(len(x.shape))
		batch_size,c,h,w = z.shape
		assert c % (self.factor ** 2) == 0, 'c = {} not multiplicative of {}'.format(c, self.factor ** 2)
		if not self.ordered:
			t = z.view(batch_size,c//self.factor**2, self.factor, self.factor,h,w)
			t = t.permute(0,1,4,2,5,3).contiguous()
		else:
			t = z.view(batch_size,self.factor, self.factor, c//self.factor**2, h,w)
			t = t.permute(0,3,4,1,5,2).contiguous()

		x = t.view(batch_size, c//self.factor**2, h*self.factor,w*self.factor)
		return x

	def forward(self,x):
		z = self._squeeze(x)
		ldj = torch.zeros(x.shape[0],device=x.device,dtype=x.dtype)
		return z,ldj

	def inverse(self,z):
		x = self._unsqueeze(z)
		return x

class Unsqueeze2d(Squeeze2d):

	def __init__(self, factor=2, ordered=False):
		super(Unsqueeze2d, self).__init__(factor=factor, ordered=ordered)

	def forward(self,x):
		z = self._unsqueeze(x)
		ldj = torch.zeros(x.shape[0], device=x.device, dtype=x.dtype)
		return z, ldj 

	def inverse(self,z):
		x = self._squeeze(z)
		return x



########################################## 4 flow model
# 	build based on nn.Module
# 	need base_dist and transforms to initialize
#	can encode image, compute loglikelihood and do samping
########################################## 

class Flow(Distribution):

	def __init__(self, base_dist, transforms):
		super(Flow,self).__init__()
		assert isinstance(base_dist, Distribution)
		if isinstance(transforms,Transform): transforms = [transforms]
		assert isinstance(transforms, Iterable)
		assert all(isinstance(transform, Transform) for transform in transforms)
		self.base_dist = base_dist
		self.transforms = nn.ModuleList(transforms)
		self.lower_bound = any(transform.lower_bound for transform in transforms)

	def log_prob(self,x):
		log_prob = torch.zeros(x.shape[0],device=x.device)
		for transform in self.transforms:
			x,ldj = transform(x)
			log_prob += ldj

		log_prob += self.base_dist.log_prob(x)
		return log_prob


	def sample(self, num_samples):
		z = self.base_dist.sample(num_samples)
		for transform in reversed(self.transforms):
			z = transform.inverse(z)

		return z

	def sample_with_log_prob(self, num_samples):
		raise RuntimeError("Flow does not support sample_with_log_prob, see InverseFlow instead.")



















