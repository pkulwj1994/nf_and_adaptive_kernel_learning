import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns
import os
import pandas



asset_dir = './assets'



ism_train_ism_losses = np.load('./ism_train_ism_losses.npy')
ism_train_idsm_losses = np.load('./ism_train_idsm_losses.npy')

ism_losses_df = pd.DataFrame()






fig = plt.figure()
plt.title('ism train')
plt.plot(np.arange(400),ism_train_ism_losses[0][:400],color='red',label='ism loss')
plt.legend()
plt.plot(np.arange(400),ism_train_idsm_losses[0][:400],color='blue',label='idsm loss')
plt.legend()
plot_fig = fig.get_figure()
plot_fig.savefig(os.path.join(asset_dir,'./ism_train.png'), dpi = 400)









fig = plt.figure()
plt.title('ism train ism losses')
for j in range(50):
	plt.plot(np.arange(800),ism_train_ism_losses[j][:800],color='red',label='ism loss')
	plt.legend()
	# plt.plot(np.arange(800),ism_train_idsm_losses[j][:800],color='blue',label='idsm loss')
	# plt.legend()
plot_fig = fig.get_figure()
plot_fig.savefig(os.path.join(asset_dir,'./ism_train_train_losses.png'), dpi = 400)


fig = plt.figure()
plt.title('ism train ism losses')
for j in range(50):
	# plt.plot(np.arange(800),ism_train_ism_losses[j][:800],color='red',label='ism loss')
	# plt.legend()
	plt.plot(np.arange(800),ism_train_idsm_losses[j][:800],color='blue',label='idsm loss')
	plt.legend()
plot_fig = fig.get_figure()
plot_fig.savefig(os.path.join(asset_dir,'./ism_train_train_losses.png'), dpi = 800)