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

		L_BOX = -5
		R_BOX = 5
		KNOTS = 500

		fig=plt.figure()
		ax=fig.add_subplot(111,projection='3d')


		u=np.linspace(L_BOX,R_BOX,KNOTS)
		xx,yy=np.meshgrid(u,u)

		x_in =  (torch.stack([torch.tensor(xx).flatten(), torch.tensor(yy).flatten()],-1) + torch.tensor([0,0]))
		with torch.no_grad():
			zz = (true_energy_func_cpu(x_in)).reshape(KNOTS,KNOTS) - torch.log(energy_nc)

			x_in = x_in.to(device).to(torch.float32)

			z= (e_model(x_in)- e_model(torch.tensor([[0.0,-1.0]]).to(device))).reshape(KNOTS,KNOTS) - torch.log(model_nc.to(device))

			ax.plot_surface(x_in[:,0].reshape(KNOTS,KNOTS).to('cpu').detach().numpy(),x_in[:,1].reshape(KNOTS,KNOTS).to('cpu').detach().numpy(),z.cpu().detach().numpy(),rstride=4,cstride=4,cmap=cm.winter)


			ax.plot_surface(x_in[:,0].reshape(KNOTS,KNOTS).to('cpu').detach().numpy(),x_in[:,1].reshape(KNOTS,KNOTS).to('cpu').detach().numpy(),zz.numpy(),rstride=4,cstride=4,cmap=cm.coolwarm)
		ax.view_init(25, 30)
		d3_fig = fig.get_figure()
		d3_fig.savefig(os.path.join(asset_dir,'./{}_banna_energy_3d_epoch_{}.png'.format(train_name, epoch)), dpi = 400)
		plt.close()

		del u,xx,yy,x_in,z,zz,fig,d3_fig
		gc.collect()
		torch.cuda.empty_cache()
	return None

def banana_prob_plot(train_name, epoch,model_nc, energy_nc):
	with torch.no_grad():

		L_BOX = -5
		R_BOX = 5
		KNOTS = 500

		fig=plt.figure()
		ax=fig.add_subplot(111,projection='3d')


		u=np.linspace(L_BOX,R_BOX,KNOTS)
		xx,yy=np.meshgrid(u,u)

		x_in =  (torch.stack([torch.tensor(xx).flatten(), torch.tensor(yy).flatten()],-1) + torch.tensor([0,0]))
		with torch.no_grad():
			zz = (true_energy_func_cpu(x_in)).reshape(KNOTS,KNOTS)

			zz = torch.exp(zz)/energy_nc

			x_in = x_in.to(device).to(torch.float32)

			z= (e_model(x_in)- e_model(torch.tensor([[0.0,-1.0]]).to(device))).reshape(KNOTS,KNOTS)

			z = torch.exp(z)/model_nc.to(device)

			ax.plot_surface(x_in[:,0].reshape(KNOTS,KNOTS).to('cpu').detach().numpy(),x_in[:,1].reshape(KNOTS,KNOTS).to('cpu').detach().numpy(),zz.numpy(),rstride=4,cstride=4,cmap=cm.seismic)

			ax.plot_surface(x_in[:,0].reshape(KNOTS,KNOTS).to('cpu').detach().numpy(),x_in[:,1].reshape(KNOTS,KNOTS).to('cpu').detach().numpy(),z.cpu().detach().numpy(),rstride=4,cstride=4,cmap=cm.Spectral)

		ax.view_init(45, 45)
		d3_fig = fig.get_figure()
		d3_fig.savefig(os.path.join(asset_dir,'./{}_banna_prob_3d_epoch_{}.png'.format(train_name, epoch)), dpi = 400)
		# plt.show()
		plt.close()

		del u,xx,yy,x_in,z,zz,fig,d3_fig
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
	zz1,zz2 = calc_true_score_cpu(x_in).chunk(2,-1)

	zz1,zz2 = zz1.reshape(KNOTS,KNOTS),zz2.reshape(KNOTS,KNOTS)

	x_in = x_in.to(device).to(torch.float32)

	z1,z2 = calc_flow_score(x_in).chunk(2,-1)
	z1,z2 = z1.reshape(KNOTS,KNOTS),z2.reshape(KNOTS,KNOTS)

	ax.quiver(x_in[:,0].reshape(KNOTS,KNOTS).to('cpu').detach().numpy(),x_in[:,1].reshape(KNOTS,KNOTS).to('cpu').detach().numpy(),z1.cpu().detach().numpy()-zz1.numpy(),z2.cpu().detach().numpy()-zz2.numpy())

	d3_fig = fig.get_figure()
	d3_fig.savefig(os.path.join(asset_dir,'./{}_banna_score_epoch_{}.png'.format(train_name, epoch)), dpi = 400)
	plt.close()
	
	del u,xx,yy,x_in,z1,z2,zz1,zz2,fig,d3_fig
	gc.collect()
	torch.cuda.empty_cache()
	
	return None


def banana_score_norm_3D_plot(train_name, epoch=0):
		
	L_BOX = -5
	R_BOX = 5
	KNOTS = 500

	fig=plt.figure()
	ax=fig.add_subplot(111,projection='3d')


	u=np.linspace(L_BOX,R_BOX,KNOTS)
	xx,yy=np.meshgrid(u,u)

	x_in =  (torch.stack([torch.tensor(xx).flatten(), torch.tensor(yy).flatten()],-1) + torch.tensor([0,0]))
	zz1,zz2 = calc_true_score_cpu(x_in).chunk(2,-1)

	zz1,zz2 = zz1.reshape(KNOTS,KNOTS),zz2.reshape(KNOTS,KNOTS)

	x_in = x_in.to(device).to(torch.float32)

	z1,z2 = calc_flow_score(x_in).chunk(2,-1)
	z1,z2 = z1.reshape(KNOTS,KNOTS),z2.reshape(KNOTS,KNOTS)

	score_norm = ((z1-zz1.to(device))**2 + (z2-zz2.to(device))**2).detach()


	ax.plot_surface(x_in[:,0].reshape(KNOTS,KNOTS).to('cpu').detach().numpy(),x_in[:,1].reshape(KNOTS,KNOTS).to('cpu').detach().numpy(),score_norm.cpu().detach().numpy(),rstride=4,cstride=4,cmap=cm.coolwarm)
	
	ax.view_init(25, 40)
	d3_fig = fig.get_figure()
	d3_fig.savefig(os.path.join(asset_dir,'./{}_banna_scorenorm_3d_epoch_{}.png'.format(train_name, epoch)), dpi = 400)
	# plt.show()
	plt.close()
	del xx,yy,x_in,z1,z2,zz1,zz2,fig,d3_fig
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