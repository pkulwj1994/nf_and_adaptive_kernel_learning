


import matplotlib.pyplot as plt
import numpy as np


n = 2

ts = (np.arange(50000)+1)/10000

ts = np.concatenate([-1*ts,ts])


fig = plt.figure()
plt.scatter(ts,-0.5*ts**2,s=1,label='gaussian')
plt.legend()
plt.scatter(ts,-0.5*ts**2 + np.log(0+ ts**2),s=0.1,label='lifted n= 0')
plt.legend()
plt.scatter(ts,-0.5*ts**2 + np.log(1+ ts**2) - np.log(1),s=0.1,label='lifted n= 1')
plt.legend()
# plt.scatter(ts,-0.5*ts**2 + np.log(2+ ts**2) - np.log(2),s=0.1,label='lifted n= 2')
# plt.legend()
# plt.scatter(ts,-0.5*ts**2 + np.log(3+ ts**2) - np.log(3),s=0.1,label='lifted n= 3')
# plt.legend()
# plt.scatter(ts,-0.5*ts**2 + np.log(4+ ts**2) - np.log(4),s=0.1,label='lifted n= 4')
plt.legend()
plt.scatter(ts,-0.5*ts**2 + np.log(5+ ts**2) - np.log(5),s=0.1,label='lifted n= 5')
plt.legend()
plt.scatter(ts,-0.5*ts**2 + np.log(10+ ts**2) - np.log(10),s=0.1,label='lifted n= 10')
plt.legend()
plt.scatter(ts,-0.5*ts**2 + np.log(100+ ts**2) - np.log(100),s=0.1,label='lifted n= 100')
plt.legend()
scatter_fig = fig.get_figure()
scatter_fig.savefig(os.path.join('./assets','./gaussian_and_lifted_n_.png'), dpi = 800)
