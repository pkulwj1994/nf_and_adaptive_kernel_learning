import torch




from torch.autograd.functional import jacobian as jcb, hessian as hess
from torch.autograd import grad 
from torch.autograd import Variable


def compute_implicit_score_diff(x):
	x = x.to(device)
	x.requires_grad_(True)
	logp = e_model(x).sum()
	grad1 = grad(logp, x, create_graph=True)[0]

	hess1 = torch.stack([grad(grad1[:, i].sum(),x, create_graph=True, retain_graph=True)[0] for i in range(x.shape[-1])] ,-1)
	return 0.5*torch.einsum('bi,bj->bij',grad1,grad1) + hess1

def compute_model_score_and_hess(x):
	x = x.to(device)
	x.requires_grad_(True)
	logp = e_model(x).sum()
	grad1 = grad(logp, x, create_graph=True)[0]

	hess1 = torch.stack([grad(grad1[:, i].sum(),x, create_graph=True, retain_graph=True)[0] for i in range(x.shape[-1])] ,-1)
	x.requires_grad_(False)
	return grad1, hess1


def compute_model_score(x):
	x = x.to(device)
	x.requires_grad_(True)
	logp = e_model(x).sum()
	grad1 = grad(logp, x, create_graph=True)[0]

	return grad1


def compute_flow_score_and_hess(x):
	x = x.to(device)
	x.requires_grad_(True)
	logp = flow_model.log_prob(x).sum()
	grad1 = grad(logp, x, create_graph=True)[0]
	hess1 = torch.stack([grad(grad1[:, i].sum(),x, create_graph=True, retain_graph=True)[0] for i in range(x.shape[-1])] ,-1)
	return grad1, hess1

def compute_true_score_and_hess(x):
	x = x.to(device)
	grad1 = torch.stack([-0.5*torch_e()**2*x[:,0]**3 + torch_e()**2*x[:,1]*x[:,0] + (torch_e()**2 - 1)*x[:,0],
		-torch_e()**2*x[:,1] + 0.5*torch_e()**2*x[:,0]**2 - torch_e()**2],-1)

	hess11 = -1.5*torch_e()**2*x[:,0]**2 + (torch_e()**2+1)*x[:,1]+torch_e()**2 -1
	hess12 = torch_e()**2*x[:,0]
	hess21 = torch_e()**2*x[:,0]
	hess22 = -torch_e()**2 + 0.0*x[:,0]

	hess1 = torch.stack([hess11,hess12,hess21,hess22],-1).reshape(x.shape[0],2,2).detach()

	return grad1.detach(),hess1.detach()


def calc_ism_loss_old(x):

	# D = compute_batch_D(x)
	imp_mat_diff = compute_implicit_score_diff(x)

	return (imp_mat_diff*torch.eye(2).to(device)).sum()/x.shape[0]

def calc_ism_loss_slow(x):
	grad1,hess1 = compute_model_score_and_hess(x)

	return (0.5*(grad1*grad1).sum() + torch.einsum('bii->b',hess1).sum())/x.shape[0]

def calc_ism_loss_fast(x):
	grad1,hess1 = compute_model_score_and_hess(x)

	return (0.5*grad1**2 + torch.diagonal(hess1,dim1=-1,dim2=-2)).sum()/x.shape[0]


def calc_idsm_loss_old_wrong(x):

	x = x.to(device)
	x.requires_grad_(True)
	logp = flow_model.log_prob(x).sum()
	grad1 = grad(logp, x, create_graph=True)[0]
	hess1 = torch.stack([grad(grad1[:, i].sum(),x, create_graph=True, retain_graph=True)[0] for i in range(x.shape[-1])] ,-1)

	D = 0.5*torch.einsum('bi,bj->bij',grad1,grad1) + hess1
	imp_mat_diff = compute_implicit_score_diff(x)
	return (imp_mat_diff*D).sum()/x.shape[0]


def calc_idsm_loss_diagsst_slow(x):

	grad1, hess1 = compute_model_score_and_hess(x)
	grad2, hess2 = compute_flow_score_and_hess(x)

	D = 0.5*torch.einsum('bi,bj->bij',grad2,grad2)*torch.eye(2).to(device)

	loss1 = ((0.5*torch.einsum('bi,bj->bij',grad1,grad1) + hess1)*D).sum()
	loss2 = (torch.einsum('bi,bj->bij',grad1,grad2)*(hess2*torch.eye(2).to(device))).sum()

	return (loss1+loss2)/x.shape[0]



# x = torch.randn(256,2).to(device)

# tic = time.time()
# calc_ism_loss_slow(x)
# toc = time.time()
# print(toc-tic)

# tic = time.time()
# calc_ism_loss_fast(x)
# toc = time.time()
# print(toc-tic)


def calc_idsm_loss_diagsst_fast(x):

	# einsum*D is faster than quad form

	grad1, hess1 = compute_model_score_and_hess(x)
	grad2, hess2 = compute_flow_score_and_hess(x)

	loss1 = (0.5*(grad1**2*grad2**2) + (torch.diagonal(hess1,dim1=-1,dim2=-2)*grad2**2)+grad1*torch.diagonal(hess2,dim1=-1,dim2=-2)*grad2).sum()

	return loss1/x.shape[0]

	


def calc_idsm_loss_id_diagsst_fast(x):

	grad1, hess1 = compute_model_score_and_hess(x)
	grad2, hess2 = compute_flow_score_and_hess(x)

	loss1 = (0.5*(grad1**2*(1+grad2)**2) + (torch.diagonal(hess1,dim1=-1,dim2=-2)*(1+grad2)**2)+grad1*torch.diagonal(hess2,dim1=-1,dim2=-2)*(1+grad2)).sum()

	return loss1/x.shape[0]


def calc_self_idsm_loss_diagsst_fast(x):
	M = 5.0

	grad1, hess1 = compute_model_score_and_hess(x)

	
	grad2 = torch.clamp(grad1,-M,M).detach()
	hess2 = torch.clamp(hess1,-M,M).detach()

	loss1 = (0.5*(grad1**2*(1+grad2)**2) + (torch.diagonal(hess1,dim1=-1,dim2=-2)*(1+grad2)**2)+grad1*torch.diagonal(hess2,dim1=-1,dim2=-2)*(1+grad2)).sum()
	return loss1/x.shape[0]
	






def calc_idsm_loss_diagsst_diaghess(x):

	grad1, hess1 = compute_model_score_and_hess(x)
	x = x.to(device)
	x.requires_grad_(True)
	logp = flow_model.log_prob(x).sum()
	grad2 = grad(logp, x, create_graph=True)[0]
	hess2 = torch.stack([grad(grad2[:, i].sum(),x, create_graph=True, retain_graph=True)[0] for i in range(x.shape[-1])] ,-1)

	D = (0.5*torch.einsum('bi,bj->bij',grad2,grad2)+ hess2)*torch.eye(2).to(device)

	loss1 = ((0.5*torch.einsum('bi,bj->bij',grad1,grad1) + hess1)*D).sum()
	loss2 = (torch.einsum('bi,bj->bij',grad1,grad2)*(hess2*torch.eye(2).to(device))).sum()

	grad22 = torch.diagonal(hess2, dim1=-1,dim2=-2)
	grad222 = torch.stack([grad(grad22[:, i].sum(),x, create_graph=True, retain_graph=True)[0] for i in range(x.shape[-1])] ,-1)

	loss3 = (grad1*torch.diagonal(grad222, dim1=-1,dim2=-2)).sum()

	x.requires_grad_(False)



	return (loss1+loss2+loss3)/x.shape[0]


def calc_idsm_loss_sst(x):

	grad1, hess1 = compute_model_score_and_hess(x)
	grad2, hess2 = compute_flow_score_and_hess(x)

	D = 0.5*torch.einsum('bi,bj->bij',grad2,grad2)

	loss1 = ((0.5*torch.einsum('bi,bj->bij',grad1,grad1) + hess1)*D).sum()
	loss2 = (torch.einsum('bi,bj->bij',grad1,grad2)*(hess2*torch.eye(2).to(device))).sum()
	loss2 = (0.5*(grad1*grad2).sum(-1)*torch.diagonal(hess2, dim1=-2,dim2=-1).sum(-1)).sum()
	loss3 = (0.5*torch.einsum('bi,bj->bij',grad1,grad2)*hess2).sum()

	return (loss1+loss2+loss3)/x.shape[0]


def calc_idsm_loss_sst_hess(x):

	grad1, hess1 = compute_model_score_and_hess(x)
	x = x.to(device)
	x.requires_grad_(True)
	logp = flow_model.log_prob(x).sum()
	grad2 = grad(logp, x, create_graph=True)[0]
	hess2 = torch.stack([grad(grad2[:, i].sum(),x, create_graph=True, retain_graph=True)[0] for i in range(x.shape[-1])] ,-1)

	D = 0.5*torch.einsum('bi,bj->bij',grad2,grad2) + hess2

	loss1 = ((0.5*torch.einsum('bi,bj->bij',grad1,grad1) + hess1)*D).sum()
	loss2 = (torch.einsum('bi,bj->bij',grad1,grad2)*(hess2*torch.eye(2).to(device))).sum()
	loss2 = (0.5*(grad1*grad2).sum(-1)*torch.diagonal(hess2, dim1=-2,dim2=-1).sum(-1)).sum()
	loss3 = (0.5*torch.einsum('bi,bj->bij',grad1,grad2)*hess2).sum()

	grad22 = torch.diagonal(hess2, dim1=-2,dim2=-1).sum(-1).sum()
	grad222 = grad(grad22, x, create_graph=True)[0]

	loss4 = (grad1*grad222).sum()

	return (loss1+loss2+loss3+loss4)/x.shape[0]


def calc_idsm_loss_sstinv(x):

	grad1, hess1 = compute_model_score_and_hess(x)
	grad2, hess2 = compute_flow_score_and_hess(x)

	D = 0.5*torch.einsum('bi,bj-> bij',1/grad2,1/grad2)*torch.eye(2).to(device)

	loss1 = ((0.5*torch.einsum('bi,bj->bij',grad1,grad1) + hess1)*D).sum()
	loss2 = -1*(torch.einsum('bi,bj->bij',grad1,grad2)*D*D*hess2).sum()


	return (loss1+loss2)/x.shape[0]


def calc_true_idsm_loss_old(x):
	grad1 = torch.stack([-0.5*torch_e()*x[:,0]**3 + torch_e()**2*x[:,1]*x[:,0] + (torch_e()**2 - 1)*x[:,0],
		-torch_e()**2*x[:,1] + 0.5*torch_e()**2*x[:,0]**2 - torch_e()**2],-1)

	hess11 = -1.5*torch_e()**2*x[:,0]**2 + (torch_e()**2+1)*x[:,1]+torch_e()**2 -1
	hess12 = torch_e()**2*x[:,0]
	hess21 = torch_e()**2*x[:,0]
	hess22 = -torch_e()**2 + 0.0*x[:,0]

	hess1 = torch.stack([hess11,hess12,hess21,hess22],-1).reshape(x.shape[0],2,2)

	D = (0.5*torch.einsum('bi,bj->bij',grad1,grad1) + hess1).to(device)
	imp_mat_diff = compute_implicit_score_diff(x)
	return (imp_mat_diff*D).sum()/x.shape[0]


def calc_true_idsm_loss_diagsst(x):

	grad1, hess1 = compute_model_score_and_hess(x)
	grad2, hess2 = compute_true_score_and_hess(x)

	D = 0.5*torch.einsum('bi,bj->bij',grad2,grad2)*torch.eye(2).to(device)

	loss1 = ((0.5*torch.einsum('bi,bj->bij',grad1,grad1) + hess1)*D).sum()
	loss2 = (torch.einsum('bi,bj->bij',grad1,grad2)*(hess2*torch.eye(2).to(device))).sum()

	return (loss1+loss2)/x.shape[0]


def calc_true_idsm_loss_id_diagsst_fast(x):

	grad1, hess1 = compute_model_score_and_hess(x)
	grad2, hess2 = compute_true_score_and_hess(x)

	loss1 = (0.5*(grad1**2*(1+grad2)**2) + (torch.diagonal(hess1,dim1=-1,dim2=-2)*(1+grad2)**2)+grad1*torch.diagonal(hess2,dim1=-1,dim2=-2)*(1+grad2)).sum()

	return loss1/x.shape[0]
	

def calc_true_idsm_loss_diagsst_diaghess(x):
	x = x.to(device)

	grad1, hess1 = compute_model_score_and_hess(x)
	grad2, hess2 = compute_true_score_and_hess(x)

	D = (0.5*torch.einsum('bi,bj->bij',grad2,grad2)+ hess2)*torch.eye(2).to(device)

	loss1 = ((0.5*torch.einsum('bi,bj->bij',grad1,grad1) + hess1)*D).sum()
	loss2 = (torch.einsum('bi,bj->bij',grad1,grad2)*(hess2*torch.eye(2).to(device))).sum()



	grad222 = torch.stack([-3*torch_e()**2*x[:,0],
		torch.zeros(x.shape[0]).to(device)],-1).detach()

	loss3 = (grad1*grad222).sum()


	return (loss1+loss2+loss3)/x.shape[0]


def calc_true_idsm_loss_sst(x):

	grad1, hess1 = compute_model_score_and_hess(x)
	grad2, hess2 = compute_true_score_and_hess(x)

	D = 0.5*torch.einsum('bi,bj->bij',grad2,grad2)

	loss1 = ((0.5*torch.einsum('bi,bj->bij',grad1,grad1) + hess1)*D).sum()
	loss2 = (torch.einsum('bi,bj->bij',grad1,grad2)*(hess2*torch.eye(2).to(device))).sum()
	loss2 = (0.5*(grad1*grad2).sum(-1)*torch.diagonal(hess2, dim1=-2,dim2=-1).sum(-1)).sum()
	loss3 = (0.5*torch.einsum('bi,bj->bij',grad1,grad2)*hess2).sum()

	return (loss1+loss2+loss3)/x.shape[0]


def calc_true_idsm_loss_sst_hess(x):
	x = x.to(device)

	grad1, hess1 = compute_model_score_and_hess(x)
	grad2, hess2 = compute_true_score_and_hess(x)

	D = 0.5*torch.einsum('bi,bj->bij',grad2,grad2) + hess2

	loss1 = ((0.5*torch.einsum('bi,bj->bij',grad1,grad1) + hess1)*D).sum()
	loss2 = (torch.einsum('bi,bj->bij',grad1,grad2)*(hess2*torch.eye(2).to(device))).sum()
	loss2 = (0.5*(grad1*grad2).sum(-1)*torch.diagonal(hess2, dim1=-2,dim2=-1).sum(-1)).sum()
	loss3 = (0.5*torch.einsum('bi,bj->bij',grad1,grad2)*hess2).sum()

	grad222 = torch.stack([-3*torch_e()**2*x[:,0],torch_e()+torch.zeros(x.shape[0]).to(device)],-1).detach()

	loss4 = (grad1*grad222).sum()

	return (loss1+loss2+loss3+loss4)/x.shape[0]


def calc_true_idsm_loss_sstinv(x):

	grad1, hess1 = compute_model_score_and_hess(x)
	grad2, hess2 = compute_flow_score_and_hess(x)

	D = 0.5*torch.einsum('bi,bj-> bij',1/grad2,1/grad2)*torch.eye(2).to(device)

	loss1 = ((0.5*torch.einsum('bi,bj->bij',grad1,grad1) + hess1)*D).sum()
	loss2 = -1*(torch.einsum('bi,bj->bij',grad1,grad2)*D*D*hess2).sum()


	return (loss1+loss2)/x.shape[0]	



def calc_flow_score(x):
	x = x.to(device)
	x.requires_grad_(True)
	logp = flow_model.log_prob(x).sum()
	grad1 = grad(logp, x, create_graph=True)[0]
	x.requires_grad_(False)
	return grad1

def calc_model_score(x):
	x = x.to(device)
	x.requires_grad_(True)
	logp = e_model(x).sum()
	grad1 = grad(logp, x, create_graph=True)[0]
	x.requires_grad_(False)
	return grad1

def calc_true_score_cpu(x):

	# cnst_e = torch.exp(torch.tensor([1.]))
	cnst_e = np.e

	grad1 = torch.stack([-0.5*cnst_e**2*x[:,0]**3 + cnst_e**2*x[:,1]*x[:,0] + (cnst_e**2 - 1)*x[:,0],
		-cnst_e**2*x[:,1] + 0.5*cnst_e**2*x[:,0]**2 - cnst_e**2],-1)
	return grad1



def torch_e():
	return np.e



def true_energy_func_cpu(x):
	cnst_e = np.e

	return -0.5*x[:,0]**2 - 0.5*cnst_e**2*(x[:,1]-0.5*x[:,0]**2 + 1)**2


def gauss_energy_func_cpu(x):
	return -0.5*torch.norm(x,2,dim=-1)

def gauss_energy_func_gpu(x):
	return -0.5*torch.norm(x,2,dim=-1)




	