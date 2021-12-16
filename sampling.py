import numpy as np 
from tqdm import tqdm

# import banana_utils
# from banana_utils import *


# from os import mkdir
# from torch.utils import tensorboard
# tb_dir = os.path.join('./', "tensorboard")
# asset_dir = os.path.join('./','assets')
# if os.path.exists(tb_dir):
# 	pass
# else:
# 	mkdir(tb_dir)

# if os.path.exists(asset_dir):
# 	pass
# else:
# 	mkdir(asset_dir)


def calc_energy_score(x,e_model):
	x = x.to(device)
	x.requires_grad_(True)
	logp = e_model(x).sum()
	grad1 = grad(logp, x, create_graph=False)[0]
	return grad1

def one_hmc_iter(x,epsilon,M,e_model):
	r = torch.randn(x.shape).to(device)*torch.sqrt(M)

	r_ = r + 0.5*epsilon*calc_energy_score(x,e_model)
	xx = x + epsilon*M*r_
	rr = r_ + 0.5*epsilon*calc_energy_score(xx,e_model)

	u = torch.rand(x.shape[0]).to(device)

	jump = (1+ torch.sign(torch.exp(-e_model(x).squeeze() + 0.5*M*torch.sum(r*r,-1)+e_model(xx).squeeze() - 0.5*M*torch.sum(rr*rr,-1))-u))/2
	return jump.unsqueeze(-1)*xx + (1-jump.unsqueeze(-1))*x


def _one_hmc_iter(x,epsilon,M,e_model):
	r = torch.randn(x.shape).to(device)*torch.sqrt(M)

	r_ = r + 0.5*epsilon*(-x)
	xx = x + epsilon*M*r_
	rr = r_ + 0.5*epsilon*(-x)

	u = torch.rand(x.shape[0]).to(device)

	jump = (1+ torch.sign(torch.exp(-e_model(x).squeeze() + 0.5*M*torch.sum(r*r,-1)+e_model(xx).squeeze() - 0.5*M*torch.sum(rr*rr,-1))-u))/2
	return jump.unsqueeze(-1)*xx + (1-jump.unsqueeze(-1))*x


def l_hmc_iter(x,epsilon,M, L,e_model):
	xx = x
	for i in range(L):
		xx = one_hmc_iter(x,epsilon,M,e_model)
	return xx

# def e_model_hmc_sample(batch_size,e_model,tp='true'):
# 	if tp == 'model':
# 		epsilon = torch.tensor([0.1]).to(device)
# 		M = torch.tensor([1.0]).to(device)
# 		L = 10

# 		x = torch.randn((batch_size,2)).to(device)
# 		for i in range(100):
# 			for j in range(L):
# 				r = torch.randn(x.shape).to(device)*torch.sqrt(M)

# 				r_ = r + 0.5*epsilon*calc_energy_score(x,e_model)
# 				xx = x + epsilon*M*r_
# 				rr = r_ + 0.5*epsilon*calc_energy_score(xx,e_model)

# 				u = torch.rand(x.shape[0]).to(device)

# 				jump = (1+ torch.sign(torch.exp(-true_energy_func(x).squeeze() + 0.5*M*torch.sum(r*r,-1)+true_energy_func(xx).squeeze() - 0.5*M*torch.sum(rr*rr,-1))-u))/2
# 				jump = jump.unsqueeze(-1)
# 				x = jump*xx + (1-jump)*x

# 	elif tp == 'true':
# 		epsilon = torch.tensor([0.1]).to(device)
# 		M = torch.tensor([1.0]).to(device)
# 		L = 10

# 		x = torch.randn((batch_size,2)).to(device)
# 		for i in range(100):
# 			for j in range(L):
# 				r = torch.randn(x.shape).to(device)*torch.sqrt(M)

# 				r_ = r + 0.5*epsilon*calc_true_score(x)
# 				xx = x + epsilon*M*r_
# 				rr = r_ + 0.5*epsilon*calc_true_score(xx)

# 				u = torch.rand(x.shape[0]).to(device)

# 				jump = (1+ torch.sign(torch.exp(-true_energy_func(x).squeeze() + 0.5*M*torch.sum(r*r,-1)+true_energy_func(xx).squeeze() - 0.5*M*torch.sum(rr*rr,-1))-u))/2
# 				jump = jump.unsqueeze(-1)
# 				x = jump*xx + (1-jump)*x
# 	return x



def e_model_hmc_sample(batch_size,e_model,tp='model'):
	if tp == 'model':
		epsilon = torch.tensor([0.1]).to(device)
		M = torch.tensor([1.0]).to(device)
		L = 5

		x = torch.randn((int(batch_size),2)).to(device)
		for i in range(100):
			for j in range(L):
				r = torch.randn(x.shape).to(device)*torch.sqrt(M)

				r_ = r + 0.5*epsilon*calc_energy_score(x,e_model)
				xx = x + epsilon*M*r_
				rr = r_ + 0.5*epsilon*calc_energy_score(xx,e_model)

				u = torch.rand(x.shape[0]).to(device)

				jump = (1+ torch.sign(torch.exp(-e_model(x).squeeze() + 0.5*M*torch.sum(r*r,-1)+e_model(xx).squeeze() - 0.5*M*torch.sum(rr*rr,-1))-u))/2
				jump = jump.unsqueeze(-1)
				x = jump*xx + (1-jump)*x

	elif tp == 'true':
		epsilon = torch.tensor([0.1]).to(device)
		M = torch.tensor([1.0]).to(device)
		L = 10

		x = torch.randn((batch_size,2)).to(device)
		for i in range(100):
			for j in range(L):
				r = torch.randn(x.shape).to(device)*torch.sqrt(M)

				r_ = r + 0.5*epsilon*calc_true_score_gpu(x)
				xx = x + epsilon*M*r_
				rr = r_ + 0.5*epsilon*calc_true_score_gpu(xx)

				u = torch.rand(x.shape[0]).to(device)

				jump = (1+ torch.sign(torch.exp(-true_energy_func_gpu(x).squeeze() + 0.5*M*torch.sum(r*r,-1)+true_energy_func_gpu(xx).squeeze() - 0.5*M*torch.sum(rr*rr,-1))-u))/2
				jump = jump.unsqueeze(-1)
				x = jump*xx + (1-jump)*x
	return x



def _e_model(x):
	x = x.to(device)
	return -0.5*torch.sum(x*x,-1)


def calc_true_score_gpu(x):

	cnst_e = torch.exp(torch.tensor([1.])).to(device)

	grad1 = torch.stack([-0.5*cnst_e**2*x[:,0]**3 + cnst_e**2*x[:,1]*x[:,0] + (cnst_e**2 - 1)*x[:,0],
		-cnst_e**2*x[:,1] + 0.5*cnst_e**2*x[:,0]**2 - cnst_e**2],-1)
	return grad1

def true_energy_func_gpu(x):
	cnst_e = torch.exp(torch.tensor([1.])).to(device)

	return -0.5*x[:,0]**2 - 0.5*cnst_e**2*(x[:,1]-0.5*x[:,0]**2 + 1.0)**2


			# gs_sample = (torch.randn(100000,2)+torch.tensor([0.0, -1.0])).to(device)
			# mc_model_z = (torch.exp(e_model(gs_sample).squeeze() - e_model(torch.tensor([[0.0,-1.0]]).to(device)) + 0.5*((gs_sample-torch.tensor([0.0, -1.0]).to(device))**2).sum(-1))*0.5/3.141593).mean()
			# print(mc_model_z)



def estimate_norm_constant_sampling(energy_fun,est_rounds=100):
	with torch.no_grad():
		nc_ests = torch.tensor([0.0]).to(device)
		for i in tqdm(range(est_rounds)):
			gs_sample = (torch.randn(1000000,2)+torch.tensor([0.0, -1.0])).to(device)
			mc_model_z = torch.exp(energy_fun(gs_sample).squeeze() + 0.5*((gs_sample-torch.tensor([0.0, -1.0]).to(device))**2).sum(-1)).mean()*2*3.141592654
			nc_ests += mc_model_z.detach().cpu().data

		nc_ests = nc_ests/est_rounds

	del gs_sample,mc_model_z
	gc.collect()
	torch.cuda.empty_cache()
	return nc_ests


def fast_banana_sampling(num_points):
	x1 = torch.randn(num_points)
	x2_mean = 0.5*x1**2 -1
	x2_var = torch.exp(torch.Tensor([-2]))
	x2 = x2_mean + x2_var ** 0.5*torch.randn(num_points)
	data = torch.stack((x1,x2)).t()
	return data 

def fast_gauss_sampling(num_points):
	return torch.randn(num_points,2)


def estimate_norm_constant_sampling_accurate(energy_fun,est_rounds=100):
	with torch.no_grad():
		nc_ests = torch.tensor([0.0]).to(device)
		for i in tqdm(range(est_rounds)):
			gs_sample = fast_banana_sampling(1000000).to(device)
			mc_model_z = torch.exp(energy_fun(gs_sample).detach().squeeze() - true_energy_func_gpu(gs_sample).detach()).mean()*2.3114562546904662
			nc_ests += mc_model_z.detach().cpu().data

		nc_ests = nc_ests/est_rounds

	del gs_sample,mc_model_z
	gc.collect()
	torch.cuda.empty_cache()
	return nc_ests


def estimate_true_expectation(energy_fun, est_rounds):
	with torch.no_grad():
		nc_ests = torch.tensor([0.0]).to(device)
		for i in tqdm(range(est_rounds)):
			gs_sample = fast_banana_sampling(1000000).to(device)
			nc_ests += torch.exp(energy_fun(gs_sample).detach().mean()).cpu().data

	del gs_sample
	gc.collect()
	torch.cuda.empty_cache()
	return nc_ests/est_rounds




def estimate_norm_constant_integral(energy_fun,L_BOX = -10,R_BOX = 10,KNOTS = 100000):

	# L_BOX = -10
	# R_BOX = 10
	# KNOTS = 1000
	area = (R_BOX - L_BOX)/KNOTS

	u=np.linspace(L_BOX,R_BOX,KNOTS)
	xx,yy=np.meshgrid(u,u)

	yy = yy -1.0

	with torch.no_grad():
			
		anchors = torch.tensor(np.stack([xx.flatten(),yy.flatten()],-1),dtype=torch.float32).to(device)
		nc_ests = (energy_fun(anchors)*area**2).sum()

	del anchors,xx,yy,u
	gc.collect()
	torch.cuda.empty_cache()

	return nc_ests


def estimate_kl_integral(energy_fun, mc_model_z,mc_true_z):
	L_BOX = -5
	R_BOX = 5
	KNOTS = 2000
	area = (R_BOX - L_BOX)/KNOTS

	u=np.linspace(L_BOX,R_BOX,KNOTS)
	xx,yy=np.meshgrid(u,u)

	yy = yy -1.0

	anchors = torch.tensor(np.stack([xx.flatten(),yy.flatten()],-1),dtype=torch.float32).to(device)

	with torch.no_grad():

		true_loglike = true_energy_func_gpu(anchors.to(device)) - true_energy_func_gpu(torch.tensor([[0.0,-1.0]]).to(device))
		model_loglike = (energy_fun(anchors.to(device)) - energy_fun(torch.tensor([[0.0,-1.0]]).to(device))).squeeze()
		true_density = torch.exp(true_loglike)/mc_true_z

		kl_est = ((true_loglike - model_loglike)*true_density*area**2).sum() + torch.log(mc_model_z) - torch.log(mc_true_z)

	del anchors
	del true_loglike,u,model_loglike,true_density
	gc.collect()
	torch.cuda.empty_cache()
	return kl_est


def estimate_kl_sampling(energy_fun, mc_model_z,mc_true_z, true_samples):
	with torch.no_grad():
		
		true_loglike = true_energy_func_gpu(true_samples.to(device))
		model_loglike = (energy_fun(true_samples.to(device))).squeeze()

		kl_est = (true_loglike - model_loglike).mean() + torch.log(mc_model_z) - torch.log(mc_true_z)

	del true_loglike,model_loglike
	gc.collect()
	torch.cuda.empty_cache()
	return kl_est

def estimate_kl_sampling_gauss(energy_fun, mc_model_z,mc_true_z, true_samples):
	with torch.no_grad():
		
		true_loglike = gauss_energy_func_gpu(true_samples.to(device))
		model_loglike = (energy_fun(true_samples.to(device))).squeeze()

		kl_est = (true_loglike - model_loglike).mean().cpu().numpy() + np.log(mc_model_z) - np.log(mc_true_z)

	del true_loglike,model_loglike
	gc.collect()
	torch.cuda.empty_cache()
	return kl_est


 # - energy_fun(torch.tensor([[0.0,-1.0]]).to(device))

# print(estimate_kl_integral(true_energy = true_energy_func_gpu, e_model = e_model, mc_model_z=mc_model_z,mc_true_z = mc_true_z))
# print(estimate_kl_integral(true_energy = true_energy_func_gpu, e_model = flow_model.log_prob, mc_model_z=mc_flow_z,mc_true_z = mc_true_z))

# print(estimate_norm_constant(flow_model.log_prob,1000))


# DataSets = Crescent(train_samples=10000, test_samples=5000)
# train_loader, test_loader = DataSets.get_data_loaders(128)

# flow_model = make_flow()
# flow_model.load_state_dict(torch.load(os.path.join(asset_dir,'pretrained_banana_flow.pth')))
# # flow_model.eval()

# e_model_idsm = energy_net(2).to(device)
# e_model_idsm.load_state_dict(torch.load(os.path.join(asset_dir,'{}_banana_energy_model_try_{}_epoch_{}.pth'.format('IDSM_DIAGSST_TRAIN',0,799))))

# e_model_ism = energy_net(2).to(device)
# e_model_ism.load_state_dict(torch.load(os.path.join(asset_dir,'{}_banana_energy_model_try_{}_epoch_{}.pth'.format('ISM_TRAIN',0,799))))








# import gc


# TRAIN_NAME = 'ISM_TRAIN'

# # estimate normalizing constant
# gs_sample = torch.randn(100000,2).to(device)
# mc_ism_z = (torch.exp(e_model_ism(gs_sample).squeeze() - e_model_ism(torch.tensor([[0.0,-1.0]]).to(device)) + 0.5*(gs_sample**2).sum(-1))*0.5/3.141593).mean()

# gs_sample = torch.randn(100000,2).to(device)
# mc_idsm_z = (torch.exp(e_model_idsm(gs_sample).squeeze() - e_model_idsm(torch.tensor([[0.0,-1.0]]).to(device)) + 0.5*(gs_sample**2).sum(-1))*0.5/3.141593).mean()

# gs_sample = torch.randn(100000,2).to(device)
# mc_flow_z = (torch.exp(flow_model.log_prob(gs_sample).squeeze() - flow_model.log_prob(torch.tensor([[0.0,-1.0]]).to(device)) + 0.5*(gs_sample**2).sum(-1))*0.5/3.141593).mean()

# gs_sample = torch.randn(100000,2).to(device)
# mc_true_z = (torch.exp(true_energy_func(gs_sample).squeeze() - true_energy_func(torch.tensor([[0.0,-1.0]]).to(device)) + 0.5*(gs_sample**2).sum(-1))*0.5/3.141593).mean()


# # genrate samples
# flow_samples = flow_model.sample(5000).cpu()
# ism_samples = e_model_hmc_sample(5000,e_model_ism,'model').detach().cpu()
# idsm_samples = e_model_hmc_sample(5000,e_model_idsm,'model').detach().cpu()
# # true_samples = e_model_hmc_sample(5000,true_energy_func,'true').detach().cpu()
# true_samples = DataSets.test.data[0:5000]

# ism_kl = (true_energy_func(true_samples.to(device)) - true_energy_func(torch.tensor([[0.0,-1.0]]).to(device)) - e_model_ism(true_samples.to(device)) + e_model_ism(torch.tensor([[0.0,-1.0]]).to(device))).squeeze().mean() +torch.log(mc_ism_z) - torch.log(mc_true_z)
# idsm_kl = (true_energy_func(true_samples.to(device)) - true_energy_func(torch.tensor([[0.0,-1.0]]).to(device)) - e_model_idsm(true_samples.to(device)) + e_model_idsm(torch.tensor([[0.0,-1.0]]).to(device))).squeeze().mean() +torch.log(mc_idsm_z) - torch.log(mc_true_z)
# flow_kl = (true_energy_func(true_samples.to(device)) - true_energy_func(torch.tensor([[0.0,-1.0]]).to(device)) - flow_model.log_prob(true_samples.to(device)) + flow_model.log_prob(torch.tensor([[0.0,-1.0]]).to(device))).squeeze().mean() +torch.log(mc_flow_z) - torch.log(mc_true_z)

# print('ism KL div to true model : {}'.format(ism_kl.data))
# print('idsm KL div to true model : {}'.format(idsm_kl.data))
# print('flow model KL div to true model : {}'.format(flow_kl.data))




# del gs_sample
# gc.collect()

# fig = plt.figure()
# plt.scatter(true_samples[:,0],true_samples[:,1],s=1,label='data')
# plt.legend()
# plt.scatter(flow_samples[:,0],flow_samples[:,1],s=1,label='flow model')
# plt.legend()
# plt.scatter(ism_samples[:,0],ism_samples[:,1],s=1,label='ism model')
# plt.legend()
# plt.scatter(idsm_samples[:,0],idsm_samples[:,1],s=1,label='idsm model')
# plt.legend()
# scatter_fig = fig.get_figure()
# scatter_fig.savefig(os.path.join(asset_dir,'./banana_data_flow_and_energy_1201.png'), dpi = 800)

