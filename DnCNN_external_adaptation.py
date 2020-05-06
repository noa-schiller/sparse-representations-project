import os.path
import time
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim

from utils import utils_image as util
from models.network_dncnn import DnCNN


# parameters 
n_channels = 1
model_pool = 'model_zoo'
model_name = 'dncnn_50'
sigma = 50

test_im # complete
train_im # complete

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
criterion = nn.MSELoss(reduction='sum')
lr = 1e-5
epochs = 50

test_results['psnr_before'] = []
test_results['psnr_after_psnr'] = []

model_path = os.path.join(model_pool, model_name+'.pth')
test_path = os.path.join(testsets, test_im)
train_path = os.path.join(testsets, train_im)

# load model
model = DnCNN(in_nc=n_channels, out_nc=n_channels, nc=64, nb=17, act_mode='R')
model.load_state_dict(torch.load(model_path), strict=True)
model = model.to(device)
model.eval()

# load test image  
x = util.imread_uint(test_path, n_channels=n_channels)
orig_im = x.squeeze()
x = util.uint2single(x)
np.random.seed(seed=0)  # for reproducibility
y = x + np.random.normal(0, sigma/255., x.shape) # add gaussian noise
y = util.single2tensor4(y)
y = y.to(device)

# denoise the image to compare PSNR before and after adaptation
with torch.no_grad():
x_ = model(y)

# compute PSNR
denoised_im = util.tensor2uint(x_)
prev_psnr = util.calculate_psnr(denoised_im, orig_im, border=0)
test_results['psnr_before'] = prev_psnr

# external adaptation

# open train image
x_train = util.imread_uint(train_path, n_channels=n_channels)
x_train = util.uint2single(x_train)
x_train_comp = util.single2tensor4(x_train)
x_train_comp = x_train_comp.to(device)

model.train()
optimizer = optim.Adam(model.parameters(), lr=lr)

# training loop
start_time = time.time()
for epoch in range(epochs):

	# add noise to train image
	y_train = x_train + np.random.normal(0, sigma/255., x_train.shape)
	y_train = util.single2tensor4(y_train)
	y_train = y_train.to(device)

	optimizer.zero_grad()
	loss = criterion(model(y_train), x_train_comp)
	loss.backward()
	optimizer.step()

# denoise the image again after adaptation
model.eval()
with torch.no_grad():
	x_ = model(y)

denoised_im = util.tensor2uint(x_)
psnr = util.calculate_psnr(denoised_im, orig_im, border=0)
test_results['after_psnr'] = psnr

print('Before adaptation - PSNR: {:.2f} dB; After adaptation - PSNR: {:.2f} dB'
.format(prev_psnr, psnr))