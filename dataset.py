import torch
from datasets import *

DataSets = Crescent(train_samples=100000, test_samples=50000)


torch.save(DataSets.test.data,'./crescent_test.pt')
torch.save(DataSets.train.data,'./crescent_train.pt')



a = torch.load('./crescent_test.pt')














