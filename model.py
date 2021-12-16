from activations import *
from transforms import *
from utils import *


# device = 'cuda' if torch.cuda.is_available() else 'cpu'





def coupling_net(input_dim):
	return MLP(int(input_dim/2), input_dim,hidden_units=[8,16,32,32,16,8],
	                            activation='elu',
	                            in_lambda=None)

# 4*(8+  8*16 +16*32+32*32+32*16 + 16*8 + 8*1) = 0.93w
def make_flow():
	return Flow(base_dist=StandardNormal((2,)),
			transforms=[
			AffineCouplingBijection1d(coupling_net(2)),
			SwitchBijection1d(),AffineCouplingBijection1d(coupling_net(2)),
			SwitchBijection1d(),AffineCouplingBijection1d(coupling_net(2)),
			SwitchBijection1d(),AffineCouplingBijection1d(coupling_net(2))]).to(device)

# 4*(2*8 + 8*16 + 16*32 + 32*32 + 32*16 + 16*8 + 8*1) = 0.93W
# def energy_net(input_dim):
# 	return MLP(int(input_dim), 1,hidden_units=[8,32,128,128,32,8],
#                                 activation='elu',
#                                 in_lambda=None)


def energy_net(input_dim):
	return MLP(int(input_dim), 1,hidden_units=[8,8,32,32,128,128,32,32,8,8],
                                activation='elu',
                                in_lambda=None)


