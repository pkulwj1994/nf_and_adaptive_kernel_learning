import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns
import os
import pandas


from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

import numpy as np
from matplotlib import cm



def banana_energy_plot(train_name, epoch,model_nc, energy_nc):
	with torch.no_grad():

		L_BOX = -10
		R_BOX = 10
		KNOTS = 1000

		u=np.linspace(L_BOX,R_BOX,KNOTS)
		xx,yy=np.meshgrid(u,u)
		x_in =  (torch.stack([torch.tensor(xx).flatten(), torch.tensor(yy).flatten()],-1) + torch.tensor([0,4])).to(device).to(torch.float32)

		zz = ((true_energy_func_gpu(x_in)).reshape(KNOTS,KNOTS) - torch.log(energy_nc.to(device))).cpu().numpy()
		z= ((e_model(x_in)- e_model(torch.tensor([[0.0,-1.0]]).to(device))).reshape(KNOTS,KNOTS) - torch.log(model_nc.to(device))).cpu().numpy()
		x_in = x_in.cpu()
		x1 = x_in[:,0].reshape(KNOTS,KNOTS).numpy()
		x2 = x_in[:,1].reshape(KNOTS,KNOTS).numpy()

		fig=plt.figure()
		plt.title('{} {} energy plot'.format(train_name,epoch))

		poss = [331,332,333,334,335,336,337,338,339]
		x_angles = [315,270,225,0,91,180,45,90,135]
		# x_angles = [0,45,90,135,91,180,225,270,315]

		for pos,x_angle in zip(poss,x_angles):
			ax=fig.add_subplot(pos,projection='3d')
			ax.plot_wireframe(x1,x2,z,rstride=50,cstride=50,cmap=cm.coolwarm,linewidth=0.3,color='black')
			ax.plot_surface(x1,x2,zz,rstride=10,cstride=10,cmap=cm.coolwarm)
			if x_angle != 91:
				ax.view_init(25,x_angle)
			else:
				ax.view_init(90,90)
		d3_fig = fig.get_figure()
		d3_fig.savefig(os.path.join(asset_dir,'./{}_banna_energy_3d_epoch_{}.png'.format(train_name, epoch)), dpi = 400)
		plt.close()

		del u,xx,yy,x_in,x1,x2,z,zz,fig,d3_fig
		gc.collect()
		torch.cuda.empty_cache()

	return None

def gauss_energy_plot(train_name, epoch,model_nc, energy_nc):
	with torch.no_grad():

		L_BOX = -10
		R_BOX = 10
		KNOTS = 1000

		u=np.linspace(L_BOX,R_BOX,KNOTS)
		xx,yy=np.meshgrid(u,u)
		x_in =  (torch.stack([torch.tensor(xx).flatten(), torch.tensor(yy).flatten()],-1) + torch.tensor([0,4])).to(device).to(torch.float32)

		zz = ((true_energy_func_gpu(x_in)).reshape(KNOTS,KNOTS) - torch.log(energy_nc.to(device))).cpu().numpy()
		z= ((e_model(x_in)- e_model(torch.tensor([[0.0,-1.0]]).to(device))).reshape(KNOTS,KNOTS) - torch.log(model_nc.to(device))).cpu().numpy()
		x_in = x_in.cpu()
		x1 = x_in[:,0].reshape(KNOTS,KNOTS).numpy()
		x2 = x_in[:,1].reshape(KNOTS,KNOTS).numpy()

		fig=plt.figure()
		plt.title('{} {} energy plot'.format(train_name,epoch))

		poss = [331,332,333,334,335,336,337,338,339]
		x_angles = [315,270,225,0,91,180,45,90,135]
		# x_angles = [0,45,90,135,91,180,225,270,315]

		for pos,x_angle in zip(poss,x_angles):
			ax=fig.add_subplot(pos,projection='3d')
			ax.plot_wireframe(x1,x2,z,rstride=50,cstride=50,cmap=cm.coolwarm,linewidth=0.3,color='black')
			ax.plot_surface(x1,x2,zz,rstride=10,cstride=10,cmap=cm.coolwarm)
			if x_angle != 91:
				ax.view_init(25,x_angle)
			else:
				ax.view_init(90,90)
		d3_fig = fig.get_figure()
		d3_fig.savefig(os.path.join(asset_dir,'./{}_banna_energy_3d_epoch_{}.png'.format(train_name, epoch)), dpi = 400)
		plt.close()

		del u,xx,yy,x_in,x1,x2,z,zz,fig,d3_fig
		gc.collect()
		torch.cuda.empty_cache()

	return None

def banana_prob_plot(train_name, epoch,model_nc, energy_nc):
	with torch.no_grad():

		L_BOX = -5
		R_BOX = 5
		KNOTS = 1000

		u=np.linspace(L_BOX,R_BOX,KNOTS)
		xx,yy=np.meshgrid(u,u)
		x_in =  (torch.stack([torch.tensor(xx).flatten(), torch.tensor(yy).flatten()],-1) + torch.tensor([0,2])).to(device).to(torch.float32)

		zz = ((true_energy_func_gpu(x_in)).reshape(KNOTS,KNOTS) - torch.log(energy_nc.to(device))).cpu().numpy()
		zz = np.exp(zz)
		z= ((e_model(x_in)- e_model(torch.tensor([[0.0,-1.0]]).to(device))).reshape(KNOTS,KNOTS) - torch.log(model_nc.to(device))).cpu().numpy()
		z = np.exp(z)
		x_in = x_in.cpu()
		x1 = x_in[:,0].reshape(KNOTS,KNOTS).numpy()
		x2 = x_in[:,1].reshape(KNOTS,KNOTS).numpy()

		fig=plt.figure()
		plt.title('{} {} prob plot'.format(train_name,epoch))

		poss = [331,332,333,334,335,336,337,338,339]
		x_angles = [315,270,225,0,91,180,45,90,135]
		# x_angles = [0,45,90,135,91,180,225,270,315]

		for pos,x_angle in zip(poss,x_angles):
			ax=fig.add_subplot(pos,projection='3d')
			ax.plot_wireframe(x1,x2,z,rstride=50,cstride=50,cmap=cm.coolwarm,linewidth=0.3,color='black')
			ax.plot_surface(x1,x2,zz,rstride=10,cstride=10,cmap=cm.coolwarm)
			if x_angle != 91:
				ax.view_init(25,x_angle)
			else:
				ax.view_init(90,90)
		d3_fig = fig.get_figure()
		d3_fig.savefig(os.path.join(asset_dir,'./{}_banna_prob_3d_epoch_{}.png'.format(train_name, epoch)), dpi = 400)
		plt.close()

		del u,xx,yy,x_in,x1,x2,z,zz,fig,d3_fig
		gc.collect()
		torch.cuda.empty_cache()

	return None


def banana_score_plot(train_name, epoch=0):
		
	L_BOX = -5
	R_BOX = 5
	KNOTS = 100

	fig=plt.figure()
	ax=fig.add_subplot(111)


	u=np.linspace(L_BOX,R_BOX,KNOTS)
	xx,yy=np.meshgrid(u,u)

	x_in =  (torch.stack([torch.tensor(xx).flatten(), torch.tensor(yy).flatten()],-1) + torch.tensor([0,-1]))
	zz1,zz2 = calc_true_score_cpu(x_in).detach().chunk(2,-1)

	zz1,zz2 = zz1.reshape(KNOTS,KNOTS),zz2.reshape(KNOTS,KNOTS)

	x_in = x_in.to(device).to(torch.float32)

	z1,z2 = calc_model_score(x_in).detach().chunk(2,-1)
	z1,z2 = z1.reshape(KNOTS,KNOTS),z2.reshape(KNOTS,KNOTS)

	ax.quiver(x_in[:,0].reshape(KNOTS,KNOTS).to('cpu').detach().numpy(),x_in[:,1].reshape(KNOTS,KNOTS).to('cpu').detach().numpy(),z1.cpu().detach().numpy()-zz1.numpy(),z2.cpu().detach().numpy()-zz2.numpy())

	d3_fig = fig.get_figure()
	d3_fig.savefig(os.path.join(asset_dir,'./{}_banna_score_epoch_{}.png'.format(train_name, epoch)), dpi = 400)
	plt.close()
	
	del u,xx,yy,x_in,z1,z2,zz1,zz2,fig,d3_fig
	gc.collect()
	torch.cuda.empty_cache()
	
	return None

def banana_true_lift_prob_plot(train_name, epoch=0):

	L_BOX = -10
	R_BOX = 10
	KNOTS = 1000

	u=np.linspace(L_BOX,R_BOX,KNOTS)
	xx,yy=np.meshgrid(u,u)
	x_in =  (torch.stack([torch.tensor(xx).flatten(), torch.tensor(yy).flatten()],-1) + torch.tensor([0,4])).to(device).to(torch.float32)
	x_in = x_in.cpu()
	x1 = x_in[:,0].reshape(KNOTS,KNOTS).numpy()
	x2 = x_in[:,1].reshape(KNOTS,KNOTS).numpy()
	# zz = torch.log(torch.norm(calc_true_score_cpu(x_in),2,dim=-1)**2).reshape(KNOTS,KNOTS).cpu().to(torch.float32).numpy()


	alpha = 0.0

	# for alpha in tqdm([0.00,0.01,0.05, 0.10,0.50,1.00,5.00,10.00,50.00,100.00,500.00,1000.00]):
	for alpha in tqdm(np.arange(1,51)/100):
		lift_energy = (true_energy_func_cpu(x_in)).reshape(KNOTS,KNOTS).numpy() + torch.log(1+alpha*torch.norm(calc_true_score_cpu(x_in),2,dim=-1)**2).reshape(KNOTS,KNOTS).cpu().to(torch.float32).numpy()
		lift_z = 1 + alpha*4.9700
		lift_energy = lift_energy - np.log(lift_z)
		lift_prob = np.exp(lift_energy)

		# mc_model_z = estimate_true_expectation(lambda x: torch.log(torch.norm(calc_true_score_cpu(x),2,dim=-1)**2).unsqueeze(-1).to(device),est_rounds=100)



		# z= calc_model_score(x_in).detach().chunk(2,-1)

		fig=plt.figure()
		plt.title('lift energy plot')

		poss = [331,332,333,334,335,336,337,338,339]
		x_angles = [315,270,225,0,91,180,45,90,135]
		# x_angles = [0,45,90,135,91,180,225,270,315]

		for pos,x_angle in zip(poss,x_angles):
			ax=fig.add_subplot(pos,projection='3d')
			# ax.plot_wireframe(x1,x2,z,rstride=50,cstride=50,cmap=cm.coolwarm,linewidth=0.3,color='black')
			ax.plot_surface(x1,x2,lift_energy,rstride=10,cstride=10,cmap=cm.coolwarm)
			if x_angle != 91:
				ax.view_init(25,x_angle)
			else:
				ax.view_init(90,90)
		d3_fig = fig.get_figure()
		d3_fig.savefig(os.path.join(asset_dir,'./banna_true_lifted_energy_3d_alpha_{:.2f}.png'.format(round(alpha,2))), dpi = 400)
		plt.close()

		# del u,xx,yy,x_in,x1,x2,z,zz,fig,d3_fig
		# gc.collect()
		# torch.cuda.empty_cache()



		fig=plt.figure()
		plt.title('lift prob plot')

		poss = [331,332,333,334,335,336,337,338,339]
		x_angles = [315,270,225,0,91,180,45,90,135]
		# x_angles = [0,45,90,135,91,180,225,270,315]

		for pos,x_angle in zip(poss,x_angles):
			ax=fig.add_subplot(pos,projection='3d')
			# ax.plot_wireframe(x1,x2,z,rstride=50,cstride=50,cmap=cm.coolwarm,linewidth=0.3,color='black')
			ax.plot_surface(x1,x2,lift_prob,rstride=10,cstride=10,cmap=cm.coolwarm)
			if x_angle != 91:
				ax.view_init(25,x_angle)
			else:
				ax.view_init(90,90)
		d3_fig = fig.get_figure()
		d3_fig.savefig(os.path.join(asset_dir,'./banna_true_lifted_prob_3d_alpha_{:.2f}.png'.format(round(alpha,2))), dpi = 400)
		plt.close()

	del u,xx,yy,x_in,x1,x2,lift_energy,lift_prob,fig,d3_fig
	gc.collect()
	torch.cuda.empty_cache()
	return None


def banana_truncate_lift_prob_plot(train_name, epoch=0):

	L_BOX = -10
	R_BOX = 10
	KNOTS = 1000

	u=np.linspace(L_BOX,R_BOX,KNOTS)
	xx,yy=np.meshgrid(u,u)
	x_in =  (torch.stack([torch.tensor(xx).flatten(), torch.tensor(yy).flatten()],-1) + torch.tensor([0,4])).to(device).to(torch.float32)
	x_in = x_in.cpu()
	x1 = x_in[:,0].reshape(KNOTS,KNOTS).numpy()
	x2 = x_in[:,1].reshape(KNOTS,KNOTS).numpy()
	# zz = torch.log(torch.norm(calc_true_score_cpu(x_in),2,dim=-1)**2).reshape(KNOTS,KNOTS).cpu().to(torch.float32).numpy()


	alpha = 0.0

	for alpha in tqdm([0.00,0.01,0.05, 0.10,0.50,1.00,5.00,10.00,50.00,100.00,500.00,1000.00]):
		lift_energy = (true_energy_func_cpu(x_in)).reshape(KNOTS,KNOTS).numpy() + torch.log(1+alpha*torch.clamp(torch.norm(calc_true_score_cpu(x_in),2,dim=-1)**2,-10.0,10.0)).reshape(KNOTS,KNOTS).cpu().to(torch.float32).numpy()
		lift_z = 1 + alpha*4.9700
		lift_energy = lift_energy - np.log(lift_z)
		lift_prob = np.exp(lift_energy)

		# mc_model_z = estimate_true_expectation(lambda x: torch.log(torch.norm(calc_true_score_cpu(x),2,dim=-1)**2).unsqueeze(-1).to(device),est_rounds=100)



		# z= calc_model_score(x_in).detach().chunk(2,-1)

		fig=plt.figure()
		plt.title('lift energy plot')

		poss = [331,332,333,334,335,336,337,338,339]
		x_angles = [315,270,225,0,91,180,45,90,135]
		# x_angles = [0,45,90,135,91,180,225,270,315]

		for pos,x_angle in zip(poss,x_angles):
			ax=fig.add_subplot(pos,projection='3d')
			# ax.plot_wireframe(x1,x2,z,rstride=50,cstride=50,cmap=cm.coolwarm,linewidth=0.3,color='black')
			ax.plot_surface(x1,x2,lift_energy,rstride=10,cstride=10,cmap=cm.coolwarm)
			if x_angle != 91:
				ax.view_init(25,x_angle)
			else:
				ax.view_init(90,90)
		d3_fig = fig.get_figure()
		d3_fig.savefig(os.path.join(asset_dir,'./banna_truncate_lifted_energy_3d_alpha_{:.2f}.png'.format(round(alpha,2))), dpi = 400)
		plt.close()

		# del u,xx,yy,x_in,x1,x2,z,zz,fig,d3_fig
		# gc.collect()
		# torch.cuda.empty_cache()



		fig=plt.figure()
		plt.title('lift prob plot')

		poss = [331,332,333,334,335,336,337,338,339]
		x_angles = [315,270,225,0,91,180,45,90,135]
		# x_angles = [0,45,90,135,91,180,225,270,315]

		for pos,x_angle in zip(poss,x_angles):
			ax=fig.add_subplot(pos,projection='3d')
			# ax.plot_wireframe(x1,x2,z,rstride=50,cstride=50,cmap=cm.coolwarm,linewidth=0.3,color='black')
			ax.plot_surface(x1,x2,lift_prob,rstride=10,cstride=10,cmap=cm.coolwarm)
			if x_angle != 91:
				ax.view_init(25,x_angle)
			else:
				ax.view_init(90,90)
		d3_fig = fig.get_figure()
		d3_fig.savefig(os.path.join(asset_dir,'./banna_truncate_lifted_prob_3d_alpha_{:.2f}.png'.format(round(alpha,2))), dpi = 400)
		plt.close()

	del u,xx,yy,x_in,x1,x2,lift_energy,lift_prob,fig,d3_fig
	gc.collect()
	torch.cuda.empty_cache()

	
	return None

def calc_gauss_energy(x):
	return -0.5*torch.norm(x,2,dim=-1)**2

def banana_true_lift_to_gauss_prob_plot(train_name, epoch=0):

	L_BOX = -10
	R_BOX = 10
	KNOTS = 1000

	u=np.linspace(L_BOX,R_BOX,KNOTS)
	xx,yy=np.meshgrid(u,u)
	x_in =  (torch.stack([torch.tensor(xx).flatten(), torch.tensor(yy).flatten()],-1) + torch.tensor([0,4])).to(device).to(torch.float32)
	x_in = x_in.cpu()
	x1 = x_in[:,0].reshape(KNOTS,KNOTS).numpy()
	x2 = x_in[:,1].reshape(KNOTS,KNOTS).numpy()
	# zz = torch.log(torch.norm(calc_true_score_cpu(x_in),2,dim=-1)**2).reshape(KNOTS,KNOTS).cpu().to(torch.float32).numpy()


	alpha = 0.0

	for alpha in np.arange(91,101)/100:
		lift_energy = (true_energy_func_cpu(x_in)).reshape(KNOTS,KNOTS).numpy() + alpha*(calc_gauss_energy(x_in) - true_energy_func_cpu(x_in) + torch.log(torch.tensor([2.3114562546904662]))).reshape(KNOTS,KNOTS).cpu().to(torch.float32).numpy()
		lift_z = estimate_true_expectation(lambda x: alpha*(calc_gauss_energy(x) - true_energy_func_gpu(x) + torch.log(torch.tensor([2.3114562546904662])).to(device)),est_rounds=100).cpu().numpy()
		lift_energy = lift_energy - np.log(lift_z)
		lift_prob = np.exp(lift_energy)

		# mc_model_z = estimate_true_expectation(lambda x: torch.log(torch.norm(calc_true_score_cpu(x),2,dim=-1)**2).unsqueeze(-1).to(device),est_rounds=100)



		# z= calc_model_score(x_in).detach().chunk(2,-1)

		fig=plt.figure()
		plt.title('lift 2 guass energy plot')

		poss = [331,332,333,334,335,336,337,338,339]
		x_angles = [315,270,225,0,91,180,45,90,135]
		# x_angles = [0,45,90,135,91,180,225,270,315]

		for pos,x_angle in zip(poss,x_angles):
			ax=fig.add_subplot(pos,projection='3d')
			# ax.plot_wireframe(x1,x2,z,rstride=50,cstride=50,cmap=cm.coolwarm,linewidth=0.3,color='black')
			ax.plot_surface(x1,x2,lift_energy,rstride=10,cstride=10,cmap=cm.coolwarm)
			if x_angle != 91:
				ax.view_init(25,x_angle)
			else:
				ax.view_init(90,90)
		d3_fig = fig.get_figure()
		d3_fig.savefig(os.path.join(asset_dir,'./banna_true_lifted2gauss_energy_3d_alpha_{}.png'.format(round(alpha,2))), dpi = 400)
		plt.close()

		# del u,xx,yy,x_in,x1,x2,z,zz,fig,d3_fig
		# gc.collect()
		# torch.cuda.empty_cache()



		fig=plt.figure()
		plt.title('lift 2 gauss prob plot')

		poss = [331,332,333,334,335,336,337,338,339]
		x_angles = [315,270,225,0,91,180,45,90,135]
		# x_angles = [0,45,90,135,91,180,225,270,315]

		for pos,x_angle in zip(poss,x_angles):
			ax=fig.add_subplot(pos,projection='3d')
			# ax.plot_wireframe(x1,x2,z,rstride=50,cstride=50,cmap=cm.coolwarm,linewidth=0.3,color='black')
			ax.plot_surface(x1,x2,lift_prob,rstride=10,cstride=10,cmap=cm.coolwarm)
			if x_angle != 91:
				ax.view_init(25,x_angle)
			else:
				ax.view_init(90,90)
		d3_fig = fig.get_figure()
		d3_fig.savefig(os.path.join(asset_dir,'./banna_true_lifted2gauss_prob_3d_alpha_{}.png'.format(round(alpha,2))), dpi = 400)
		plt.close()

	del u,xx,yy,x_in,x1,x2,lift_energy,lift_prob,fig,d3_fig
	gc.collect()
	torch.cuda.empty_cache()

	
	return None




def banana_score_norm_3D_plot(train_name, epoch,):

	L_BOX = -10
	R_BOX = 10
	KNOTS = 1000

	u=np.linspace(L_BOX,R_BOX,KNOTS)
	xx,yy=np.meshgrid(u,u)
	x_in =  (torch.stack([torch.tensor(xx).flatten(), torch.tensor(yy).flatten()],-1) + torch.tensor([0,4])).to(device).to(torch.float32)

	score_norm = (torch.norm(calc_true_score_gpu(x_in).detach()-calc_model_score(x_in).detach(),2,dim=-1)**2).reshape(KNOTS,KNOTS).cpu().numpy()

	x_in = x_in.cpu()
	x1 = x_in[:,0].reshape(KNOTS,KNOTS).numpy()
	x2 = x_in[:,1].reshape(KNOTS,KNOTS).numpy()

	fig=plt.figure()
	plt.title('{} {} score norm2 plot'.format(train_name,epoch))

	poss = [331,332,333,334,335,336,337,338,339]
	x_angles = [315,270,225,0,91,180,45,90,135]
	# x_angles = [0,45,90,135,91,180,225,270,315]

	for pos,x_angle in zip(poss,x_angles):
		ax=fig.add_subplot(pos,projection='3d')
		ax.plot_wireframe(x1,x2,score_norm,rstride=50,cstride=50,cmap=cm.coolwarm,linewidth=0.3,color='black')
		# ax.plot_surface(x1,x2,zz,rstride=10,cstride=10,cmap=cm.coolwarm)
		if x_angle != 91:
			ax.view_init(25,x_angle)
		else:
			ax.view_init(90,90)
	d3_fig = fig.get_figure()
	d3_fig.savefig(os.path.join(asset_dir,'./{}_banna_scorenorm_3d_epoch_{}.png'.format(train_name, epoch)), dpi = 400)
	plt.close()

	del u,xx,yy,x_in,x1,x2,fig,d3_fig
	gc.collect()
	torch.cuda.empty_cache()

	return None





# asset_dir = './assets'

# ism_train_ism_losses = np.load('./ism_train_ism_losses.npy')
# ism_train_idsm_losses = np.load('./ism_train_idsm_losses.npy')

# ism_losses_df = pd.DataFrame()

# fig = plt.figure()
# plt.title('ism train')
# plt.plot(np.arange(400),ism_train_ism_losses[0][:400],color='red',label='ism loss')
# plt.legend()
# plt.plot(np.arange(400),ism_train_idsm_losses[0][:400],color='blue',label='idsm loss')
# plt.legend()
# plot_fig = fig.get_figure()
# plot_fig.savefig(os.path.join(asset_dir,'./ism_train.png'), dpi = 400)


# fig = plt.figure()
# plt.title('ism train ism losses')
# for j in range(50):
# 	plt.plot(np.arange(800),ism_train_ism_losses[j][:800],color='red',label='ism loss')
# 	plt.legend()
# 	# plt.plot(np.arange(800),ism_train_idsm_losses[j][:800],color='blue',label='idsm loss')
# 	# plt.legend()
# plot_fig = fig.get_figure()
# plot_fig.savefig(os.path.join(asset_dir,'./ism_train_train_losses.png'), dpi = 400)


# fig = plt.figure()
# plt.title('ism train ism losses')
# for j in range(50):
# 	# plt.plot(np.arange(800),ism_train_ism_losses[j][:800],color='red',label='ism loss')
# 	# plt.legend()
# 	plt.plot(np.arange(800),ism_train_idsm_losses[j][:800],color='blue',label='idsm loss')
# 	plt.legend()
# plot_fig = fig.get_figure()
# plot_fig.savefig(os.path.join(asset_dir,'./ism_train_train_losses.png'), dpi = 800)