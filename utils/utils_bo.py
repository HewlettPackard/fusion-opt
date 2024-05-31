from scipy.stats import norm
from attrdict import AttrDict
import torch
from torch.utils.data import Dataset

def get_expected_improvement(max_mean_y, mean_y,sigma_y_new, xi = 0):
    diff_y = mean_y - max_mean_y - xi
    #print(diff_y)
    z = (diff_y) / sigma_y_new
    exp_imp = (diff_y) * norm.cdf(z) + sigma_y_new * norm.pdf(z)
    exp_imp = exp_imp.ravel()
    #print("exp_imp", exp_imp.shape)
    return exp_imp

def twod_to_task(data, num_ctx=None, max_num_points=None, target_all=False, t_noise=None, max_num_target_points=None, test = False, max_val_x=0, max_val_y=0, min_val_x=0, min_val_y=0, cuda = True):

	num_samples = len(data[0][0])
	if test:
		num_samples = len(data[0])
	batch = AttrDict()
	max_num_points = max_num_points or num_samples
	if num_samples < max_num_points:
		max_num_points = num_samples
	if max_num_target_points is None:
		max_num_target_points = max_num_points

	num_ctx = num_ctx or \
			torch.randint(low=3, high=max_num_points-3, size=[1]).item()
	
	num_tar = max_num_points - num_ctx if target_all else \
			torch.randint(low=3, high=min(max_num_points-num_ctx, max_num_target_points), size=[1]).item()
	num_points = num_ctx + num_tar

	idxs = torch.randint(low=0,high=num_samples, size=(num_points,))
	if not test:
		data_x = data[0].squeeze(0)[idxs]
		data_y = data[1].squeeze(0)[idxs]
		batch.x = torch.tensor(data_x.unsqueeze(0))
		batch.y = torch.tensor(data_y.unsqueeze(0))
	else:
		data_x = data[0][idxs]
		data_y = data[1][idxs]
		batch.x = torch.tensor(data_x)
		batch.y = torch.tensor(data_y)
	# batch.x = -0.5 + (batch.x - min_val_x) / (max_val_x - min_val_x)
	# batch.y = -0.5 + (batch.y - min_val_y) / (max_val_y - min_val_y)

	#print("max_num_points: ", max_num_points)
	batch.xc = batch.x[:, :num_ctx]	

	batch.xt = batch.x[:,num_ctx:]
	batch.yc = batch.y[:,:num_ctx]
	batch.yt = batch.y[:,num_ctx:]

	if cuda:
		batch.x = batch.x.cuda()
		batch.y = batch.y.cuda()
		batch.xc = batch.xc.cuda()
		batch.xt = batch.xt.cuda()
		batch.yc = batch.yc.cuda()
		batch.yt = batch.yt.cuda()

	return batch

def unormalize(value, max_val, min_val):
    return (value + 0.5) * (max_val - min_val) + min_val


class Dataset(Dataset):
    def __init__(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train
        self.features = X_train
        self.labels = y_train
    def __len__(self):
        return len(self.X_train)

    def __getitem__(self, idx):
        #print("IDX: ", idx)
        return self.features[idx], self.labels[idx]