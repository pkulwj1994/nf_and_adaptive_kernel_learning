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
ft_e_model_path = None

# TRAIN_NAME = 'ISM_TRAIN'
# # tries = [10,11,12,13,14,15,16,17,18,19]
# tries = [0]
# # ft_e_model_path = './assets/ISM_TRAIN_banana_energy_model_try_0_epoch_400.pth'
# ft_e_model_path = None



flow_model = make_flow()
flow_model.load_state_dict(torch.load(os.path.join(asset_dir,'pretrained_banana_flow.pth')))
# flow_model.eval()

DataSets = Crescent(train_samples=100000, test_samples=50000,train_data_path='./crescent_train.pt', test_data_path='./crescent_test.pt')
train_loader, test_loader = DataSets.get_data_loaders(5000)


current_tb_dir = os.path.join(tb_dir,time.strftime('%Y-%m-%d_%H:%M:%S',time.localtime(time.time())) + TRAIN_NAME)
mkdir(current_tb_dir)

writer = tensorboard.SummaryWriter(current_tb_dir)



ism_losses = {'ism':[]}
idsm_losses = {'diagsst':[],'diagsst_diaghess':[],'sst':[],'sst_hess':[], 'sstinv':[]}
tidsm_losses = {'diagsst':[],'diagsst_diaghess':[],'sst':[],'sst_hess':[], 'sstinv':[]}

ism_eval_losses = {'ism':[]}
idsm_eval_losses = {'diagsst':[],'diagsst_diaghess':[],'sst':[],'sst_hess':[], 'sstinv':[]}
tidsm_eval_losses = {'diagsst':[],'diagsst_diaghess':[],'sst':[],'sst_hess':[], 'sstinv':[]}


for tryy in tries:

	ism_loss = {'ism':[]}
	idsm_loss = {'diagsst':[],'diagsst_diaghess':[],'sst':[],'sst_hess':[], 'sstinv':[]}
	tidsm_loss = {'diagsst':[],'diagsst_diaghess':[],'sst':[],'sst_hess':[], 'sstinv':[]}

	ism_eval_loss = {'ism':[]}
	idsm_eval_loss = {'diagsst':[],'diagsst_diaghess':[],'sst':[],'sst_hess':[], 'sstinv':[]}
	tidsm_eval_loss = {'diagsst':[],'diagsst_diaghess':[],'sst':[],'sst_hess':[], 'sstinv':[]}

	e_model = energy_net(2).to(device)
	# ee_model = energy_net(2).to(device)
	# ee_model.load_state_dict(e_model.state_dict())
	# ee_model.eval()
	if ft_e_model_path is not None:
		e_model.load_state_dict(torch.load(ft_e_model_path))
	else:
		pass


	epoch = 0
	optimizer = Adam(e_model.parameters(), lr=1e-4)

	for epoch in range(epoch,2001):

		# if epoch == 600:
			# for param_group in optimizer.param_groups:
			# 	param_group['lr'] = 1e-4
		# else:
		# 	pass 

		# if epoch == 900:
			# for param_group in optimizer.param_groups:
			# 	param_group['lr'] = 5e-5
		# else:
		# 	pass 

		# if epoch == 1200:
		# 	for param_group in optimizer.param_groups:
		# 		param_group['lr'] = 1e-5
		# else:
		# 	pass 

		l = 0.0
		l1 = 0.0
		l2 = 0.0
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

		# if epoch %10 ==0:

		# 	banana_energy_plot(TRAIN_NAME,epoch)
		# 	banana_prob_plot(TRAIN_NAME,epoch,mc_model_z.cpu(),mc_true_z.cpu())
		# 	banana_score_norm_3D_plot(TRAIN_NAME,epoch)


		if epoch %10 == 0:

			# mc_model_z = print(estimate_norm_constant(lambda x: torch.exp(e_model(x)),1000))
			# mc_flow_z = print(estimate_norm_constant(lambda x: torch.exp(flow_model.log_prob(x)),1000))
			# mc_true_z = print(estimate_norm_constant(lambda x: torch.exp(true_energy_func_gpu(x)),1000))

			# mc_model_z = estimate_norm_constant_integral(lambda x: torch.exp(e_model(x) - e_model(torch.tensor([[0.0,-1.0]]).to(device))),L_BOX = -10,R_BOX = 10,KNOTS = 2000)

			mc_model_z = estimate_norm_constant_sampling(lambda x: e_model(x) - e_model(torch.tensor([[0.0,-1.0]]).to(device)),est_rounds=20)
			# mc_model_z = estimate_norm_constant_sampling_accurate(lambda x: e_model(x) - e_model(torch.tensor([[0.0,-1.0]]).to(device)),est_rounds=20)


			# mc_flow_z = estimate_norm_constant_integral(lambda x: torch.exp(flow_model.log_prob(x)),L_BOX = -20,R_BOX = 20,KNOTS = 2000)
			mc_flow_z = torch.tensor([1.0000]).to(device)
			# mc_true_z = estimate_norm_constant_integral(lambda x: torch.exp(true_energy_func_gpu(x)),L_BOX = -20,R_BOX = 20,KNOTS = 2000)
			mc_true_z = torch.tensor([2.3114562546904662]).to(device)

			# flow_samples = flow_model.sample(1000).cpu()
			energy_samples = e_model_hmc_sample(1000,e_model,'model').detach().cpu()
			# true_samples = DataSets.test.data[0:1000]

			# model_kl = estimate_kl_integral(e_model = e_model, mc_model_z=mc_model_z,mc_true_z = mc_true_z)
			# flow_kl = estimate_kl_integral(e_model = flow_model.log_prob, mc_model_z=mc_flow_z,mc_true_z = mc_true_z)
			true_samples = fast_banana_sampling(100000).detach().cpu()

			model_kl = estimate_kl_sampling(lambda x: e_model(x) - e_model(torch.tensor([[0.0,-1.0]]).to(device)), mc_model_z,mc_true_z, true_samples)
			flow_kl = estimate_kl_sampling(flow_model.log_prob, mc_flow_z,mc_true_z,true_samples)

			print('{} model KL div to true model : {}'.format(TRAIN_NAME, model_kl.data))
			print('flow model KL div to true model : {}'.format(flow_kl.data))
			writer.add_scalar("flow_KL", round(float(flow_kl.cpu()),4), epoch)
			writer.add_scalar("model_KL", round(float(model_kl.cpu()),4), epoch)


			fig = plt.figure()
			plt.title('{} model KL: {}, flow KL: {}'.format(TRAIN_NAME,round(float(model_kl.cpu()),4),round(float(flow_kl.cpu()),4)))
			plt.scatter(energy_samples[:,0],energy_samples[:,1],s=1,label='{} model'.format(TRAIN_NAME))
			plt.legend()
			plt.scatter(DataSets.test.data[0:1000][:,0],DataSets.test.data[0:1000][:,1],s=1,label='data')
			plt.legend()
			# plt.scatter(flow_samples[:,0],flow_samples[:,1],s=1,label='flow model')
			# plt.legend()
			scatter_fig = fig.get_figure()
			scatter_fig.savefig(os.path.join(asset_dir,'./banana_data_flow_and_energy_{}_try_{}_epoch_{}.png'.format(TRAIN_NAME,tryy,epoch)), dpi = 800)
			plt.close()

			banana_energy_plot(TRAIN_NAME,epoch,mc_model_z.cpu(),mc_true_z.cpu())
			banana_prob_plot(TRAIN_NAME,epoch,mc_model_z.cpu(),mc_true_z.cpu())
			banana_score_norm_3D_plot(TRAIN_NAME,epoch)


			del true_samples, energy_samples
			gc.collect()
			torch.cuda.empty_cache()


			torch.save(e_model.state_dict(),os.path.join(asset_dir,'{}_banana_energy_model_try_{}_epoch_{}.pth'.format(TRAIN_NAME,tryy,epoch)))


		for i,x in enumerate(train_loader):
			l1 += ((calc_ism_loss_fast(x))/(1+ALPHA*S2_CONST)).detach().cpu().item()
			l2 += ((ALPHA*calc_self_idsm_loss_diagsst_fast(x))/(1+ALPHA*S2_CONST)).detach().cpu().item()
			# l += calc_ism_loss_fast(x).detach().cpu().item()
			# ll_diagsst += l1+l2
			# ll_diagsst_diaghess += calc_idsm_loss_diagsst_diaghess(x).detach().cpu().item()
			# ll_sst += calc_idsm_loss_sst(x).detach().cpu().item()
			# ll_sst_hess += calc_idsm_loss_sst_hess(x).detach().cpu().item()
			# ll_sstinv += calc_idsm_loss_sstinv(x).detach().cpu().item()

			# lll_diagsst += calc_true_idsm_loss_id_diagsst_fast(x).detach().cpu().item()
			# lll_diagsst_diaghess += calc_true_idsm_loss_diagsst_diaghess(x).detach().cpu().item()
			# lll_sst += calc_true_idsm_loss_sst(x).detach().cpu().item()
			# lll_sst_hess += calc_true_idsm_loss_sst_hess(x).detach().cpu().item()
			# lll_sstinv += calc_true_idsm_loss_sstinv(x).detach().cpu().item()

			# l += 0
			# ll += 0
			# lll += 0		

			optimizer.zero_grad()

			# l1 = calc_ism_loss_fast(x)/(1+ALPHA*S2_CONST)
			# l2 = ALPHA*calc_idsm_loss_diagsst_fast(x)/(1+ALPHA*S2_CONST)
			loss = (calc_ism_loss_fast(x) + ALPHA*calc_self_idsm_loss_diagsst_fast(x))/(1+ALPHA*S2_CONST)
			# loss = calc_ism_loss_fast(x)

			loss.backward()

			optimizer.step()
			optimizer.zero_grad()

			# l += loss.detach()

			print('ism Epoch: {}/{}, Iter: {}/{}, AvgISM: {:.3f},AvgReg: {:.3f},AvgIDSM: {:.3f},Reg/ISM {:.3f}'.format(epoch+1,20,i+1,len(train_loader),l1/(i+1),l2/(i+1),(l1+l2)/(i+1),(l2+0.01)/(l1+0.01)),end='\r')
		ism_loss['ism'].append(l1/(i+1))
		idsm_loss['diagsst'].append((l1+l2)/(i+1))
		# idsm_loss['diagsst_diaghess'].append(ll_diagsst_diaghess/(i+1))
		# idsm_loss['sst'].append(ll_sst/(i+1))
		# idsm_loss['sst_hess'].append(ll_sst_hess/(i+1))
		# idsm_loss['sstinv'].append(ll_sstinv/(i+1))

		# tidsm_loss['diagsst'].append(lll_diagsst/(i+1))
		# tidsm_loss['diagsst_diaghess'].append(lll_diagsst_diaghess/(i+1))
		# tidsm_loss['sst'].append(lll_sst/(i+1))
		# tidsm_loss['sst_hess'].append(lll_sst_hess/(i+1))
		# tidsm_loss['sstinv'].append(lll_sstinv/(i+1))

		if tryy == 0:
			writer.add_scalar("{}_ism_loss".format('train'), l1/(i+1), epoch)			
			writer.add_scalar("{}_reg_loss".format('train'), l2/(i+1), epoch)
			writer.add_scalar("{}_idsm_diagsst_loss".format('train'), (l1+l2)/(i+1), epoch)
			writer.add_scalar("{}_reg_ism_ratio".format('train'), l2/l1, epoch)

			# writer.add_scalar("{}_idsm_loss_diagsst_diaghess".format('train'), ll_diagsst_diaghess/(i+1), epoch)
			# writer.add_scalar("{}_idsm_loss_sst".format('train'), ll_sst/(i+1), epoch)
			# writer.add_scalar("{}_idsm_loss_sst_hess".format('train'), ll_sst_hess/(i+1), epoch)
			# writer.add_scalar("{}_idsm_loss_sstinv".format('train'), ll_sstinv/(i+1), epoch)
			# writer.add_scalar("{}_tidsm_loss_diagsst".format('train'), lll_diagsst/(i+1), epoch)
			# writer.add_scalar("{}_tidsm_loss_diagsst_diaghess".format('train'), lll_diagsst_diaghess/(i+1), epoch)
			# writer.add_scalar("{}_tidsm_loss_sst".format('train'), lll_sst/(i+1), epoch)
			# writer.add_scalar("{}_tidsm_loss_sst_hess".format('train'), lll_sst_hess/(i+1), epoch)
			# writer.add_scalar("{}_tidsm_loss_sstinv".format('train'), lll_sstinv/(i+1), epoch)


		print('')

		if epoch %10 ==0:

			l = 0.0
			ll = 0.0
			lll = 0.0
			l1=0.0
			l2 = 0.0
			for i,x in enumerate(test_loader):
				l1 += ((calc_ism_loss_fast(x))/(1+ALPHA*S2_CONST)).detach().cpu().item()
				l2 += ((ALPHA*calc_self_idsm_loss_diagsst_fast(x))/(1+ALPHA*S2_CONST)).detach().cpu().item()
				# ll_diagsst_diaghess += calc_idsm_loss_diagsst_diaghess(x).detach().cpu().item()
				# ll_sst += calc_idsm_loss_sst(x).detach().cpu().item()
				# ll_sst_hess += calc_idsm_loss_sst_hess(x).detach().cpu().item()
				# ll_sstinv += calc_idsm_loss_sstinv(x).detach().cpu().item()

				# lll_diagsst += calc_true_idsm_loss_id_diagsst_fast(x).detach().cpu().item()
				# lll_diagsst_diaghess += calc_true_idsm_loss_diagsst_diaghess(x).detach().cpu().item()
				# lll_sst += calc_true_idsm_loss_sst(x).detach().cpu().item()
				# lll_sst_hess += calc_true_idsm_loss_sst_hess(x).detach().cpu().item()
				# lll_sstinv += calc_true_idsm_loss_sstinv(x).detach().cpu().item()

				# l += calc_ism_loss(x).detach().cpu().item()
				# ll += 0
				# lll+= 0
			ism_eval_loss['ism'].append(l1/(i+1))
			idsm_eval_loss['diagsst'].append((l1+l2)/(i+1))
			# idsm_eval_loss['diagsst_diaghess'].append(ll/(i+1))
			# idsm_eval_loss['sst'].append(ll/(i+1))
			# idsm_eval_loss['sst_hess'].append(ll/(i+1))
			# idsm_eval_loss['sstinv'].append(ll/(i+1))

			# tidsm_eval_loss['diagsst'].append(lll_diagsst/(i+1))
			# tidsm_eval_loss['diagsst_diaghess'].append(lll_diagsst_diaghess/(i+1))
			# tidsm_eval_loss['sst'].append(lll_sst/(i+1))
			# tidsm_eval_loss['sst_hess'].append(lll_sst_hess/(i+1))
			# tidsm_eval_loss['sstinv'].append(lll_sstinv/(i+1))
			if tryy == 0:
				writer.add_scalar("{}_ism_loss".format('eval'), l1/(i+1), epoch)			
				writer.add_scalar("{}_reg_loss".format('eval'), l2/(i+1), epoch)
				writer.add_scalar("{}_idsm_diagsst_loss".format('eval'), (l1+l2)/(i+1), epoch)
				writer.add_scalar("{}_reg_ism_ratio".format('eval'), l2/l1, epoch)
				# writer.add_scalar("{}_idsm_loss_diagsst_diaghess".format('eval'), ll_diagsst_diaghess/(i+1), epoch)
				# writer.add_scalar("{}_idsm_loss_sst".format('eval'), ll_sst/(i+1), epoch)
				# writer.add_scalar("{}_idsm_loss_sst_hess".format('eval'), ll_sst_hess/(i+1), epoch)
				# writer.add_scalar("{}_idsm_loss_sstinv".format('eval'), ll_sstinv/(i+1), epoch)
				# writer.add_scalar("{}_tidsm_loss_diagsst".format('eval'), lll_diagsst/(i+1), epoch)
				# writer.add_scalar("{}_tidsm_loss_diagsst_diaghess".format('eval'), lll_diagsst_diaghess/(i+1), epoch)
				# writer.add_scalar("{}_tidsm_loss_sst".format('eval'), lll_sst/(i+1), epoch)
				# writer.add_scalar("{}_tidsm_loss_sst_hess".format('eval'), lll_sst_hess/(i+1), epoch)
				# writer.add_scalar("{}_tidsm_loss_sstinv".format('eval'), lll_sstinv/(i+1), epoch)


	torch.save(e_model.state_dict(),os.path.join(asset_dir,'{}_banana_energy_model_try_{}_epoch_{}.pth'.format(TRAIN_NAME,tryy,epoch)))


	ism_losses['ism'].append(np.array(ism_loss['ism']))
	idsm_losses['diagsst'].append(np.array(idsm_loss['diagsst']))
	# idsm_losses['diagsst_diaghess'].append(np.array(idsm_loss['diagsst_diaghess']))
	# idsm_losses['sst'].append(np.array(idsm_loss['sst']))
	# idsm_losses['sst_hess'].append(np.array(idsm_loss['sst_hess']))
	# idsm_losses['sstinv'].append(np.array(idsm_loss['sstinv']))

	# tidsm_losses['diagsst'].append(np.array(tidsm_losses['diagsst']))
	# tidsm_losses['diagsst_diaghess'].append(np.array(tidsm_losses['diagsst_diaghess']))
	# tidsm_losses['sst'].append(np.array(tidsm_losses['sst']))
	# tidsm_losses['sst_hess'].append(np.array(tidsm_losses['sst_hess']))
	# tidsm_losses['sstinv'].append(np.array(tidsm_losses['sstinv']))

	ism_eval_losses['ism'].append(np.array(ism_eval_loss['ism']))
	idsm_eval_losses['diagsst'].append(np.array(idsm_eval_loss['diagsst']))
	# idsm_eval_losses['diagsst_diaghess'].append(np.array(idsm_eval_loss['diagsst_diaghess']))
	# idsm_eval_losses['sst'].append(np.array(idsm_eval_loss['sst']))
	# idsm_eval_losses['sst_hess'].append(np.array(idsm_eval_loss['sst_hess']))
	# idsm_eval_losses['sstinv'].append(np.array(idsm_eval_loss['sstinv']))

	# tidsm_eval_losses['diagsst'].append(np.array(tidsm_eval_losses['diagsst']))
	# tidsm_eval_losses['diagsst_diaghess'].append(np.array(tidsm_eval_losses['diagsst_diaghess']))
	# tidsm_eval_losses['sst'].append(np.array(tidsm_eval_losses['sst']))
	# tidsm_eval_losses['sst_hess'].append(np.array(tidsm_eval_losses['sst_hess']))
	# tidsm_eval_losses['sstinv'].append(np.array(tidsm_eval_losses['sstinv']))


	np.save(os.path.join(asset_dir,'{}_ism_losses_ism.npy'.format(TRAIN_NAME)),np.array(ism_losses['ism']))
	np.save(os.path.join(asset_dir,'{}_idsm_losses_diagsst.npy'.format(TRAIN_NAME)),np.array(idsm_losses['diagsst']))
	# np.save(os.path.join(asset_dir,'{}_idsm_losses_diagsst_diaghess.npy'.format(TRAIN_NAME)),np.array(idsm_losses['diagsst_diaghess']))
	# np.save(os.path.join(asset_dir,'{}_idsm_losses_sst.npy'.format(TRAIN_NAME)),np.array(idsm_losses['sst']))
	# np.save(os.path.join(asset_dir,'{}_idsm_losses_sst_hess.npy'.format(TRAIN_NAME)),np.array(idsm_losses['sst_hess']))
	# np.save(os.path.join(asset_dir,'{}_idsm_losses_sstinv.npy'.format(TRAIN_NAME)),np.array(idsm_losses['sstinv']))
	# np.save(os.path.join(asset_dir,'{}_tidsm_losses_diagsst.npy'.format(TRAIN_NAME)),np.array(tidsm_losses['diagsst']))
	# np.save(os.path.join(asset_dir,'{}_tidsm_losses_diagsst_diaghess.npy'.format(TRAIN_NAME)),np.array(tidsm_losses['diagsst_diaghess']))
	# np.save(os.path.join(asset_dir,'{}_tidsm_losses_sst.npy'.format(TRAIN_NAME)),np.array(tidsm_losses['sst']))
	# np.save(os.path.join(asset_dir,'{}_tidsm_losses_sst_hess.npy'.format(TRAIN_NAME)),np.array(tidsm_losses['sst_hess']))
	# np.save(os.path.join(asset_dir,'{}_tidsm_losses_sstinv.npy'.format(TRAIN_NAME)),np.array(tidsm_losses['sstinv']))

	np.save(os.path.join(asset_dir,'{}_ism_eval_losses_ism.npy'.format(TRAIN_NAME)),np.array(ism_eval_losses['ism']))
	np.save(os.path.join(asset_dir,'{}_idsm_eval_losses_diagsst.npy'.format(TRAIN_NAME)),np.array(idsm_eval_losses['diagsst']))
	# np.save(os.path.join(asset_dir,'{}_idsm_eval_losses_diagsst_diaghess.npy'.format(TRAIN_NAME)),np.array(idsm_eval_losses['diagsst_diaghess']))
	# np.save(os.path.join(asset_dir,'{}_idsm_eval_losses_sst.npy'.format(TRAIN_NAME)),np.array(idsm_eval_losses['sst']))
	# np.save(os.path.join(asset_dir,'{}_idsm_eval_losses_sst_hess.npy'.format(TRAIN_NAME)),np.array(idsm_eval_losses['sst_hess']))
	# np.save(os.path.join(asset_dir,'{}_idsm_eval_losses_sstinv.npy'.format(TRAIN_NAME)),np.array(idsm_eval_losses['sstinv']))
	# np.save(os.path.join(asset_dir,'{}_tidsm_eval_losses_diagsst.npy'.format(TRAIN_NAME)),np.array(tidsm_eval_losses['diagsst']))
	# np.save(os.path.join(asset_dir,'{}_tidsm_eval_losses_diagsst_diaghess.npy'.format(TRAIN_NAME)),np.array(tidsm_eval_losses['diagsst_diaghess']))
	# np.save(os.path.join(asset_dir,'{}_tidsm_eval_losses_sst.npy'.format(TRAIN_NAME)),np.array(tidsm_eval_losses['sst']))
	# np.save(os.path.join(asset_dir,'{}_tidsm_eval_losses_sst_hess.npy'.format(TRAIN_NAME)),np.array(tidsm_eval_losses['sst_hess']))
	# np.save(os.path.join(asset_dir,'{}_tidsm_eval_losses_sstinv.npy'.format(TRAIN_NAME)),np.array(tidsm_eval_losses['sstinv']))




