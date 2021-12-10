import torch
from datasets import *

DataSets = Crescent(train_samples=100000, test_samples=50000)


torch.save(DataSets.test.data,'./crescent_test.pt')
torch.save(DataSets.train.data,'./crescent_train.pt')



DataSets = MixtureOfGaussian5(train_samples=100000, test_samples=50000)


torch.save(DataSets.test.data,'./mog5_test.pt')
torch.save(DataSets.train.data,'./mog5_train.pt')





DataSets = MixtureOfGaussian5highlow(train_samples=100000, test_samples=50000)


torch.save(DataSets.test.data,'./mog5hl_test.pt')
torch.save(DataSets.train.data,'./mog5hl_train.pt')

vecs = DataSets.test.data[0:2000]


fig = plt.figure()
plt.scatter(vecs[:,0],vecs[:,1],s=1,label='mog4')
plt.legend()
# scatter_fig = fig.get_figure()
plt.savefig(os.path.join('./assets','mog5hl.png'), dpi = 400)







