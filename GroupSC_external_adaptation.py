import dataloaders_v2
import numpy as np
from tqdm import tqdm
import os, sys
import time
import cv2

import torch
from torchvision import  transforms
import torch.nn.functional as F
import torch.nn as nn

from ops.utils_blocks import block_module
from model.gray_group import ListaParams
from model.gray_group import  groupLista as Lista

# parameters
# the same setting used in the original code
kernel_size = 9
num_filters = 256
stride = 1
unfoldings = 24
freq_corr_update_test = 6
freq_corr_update_train = 100
corr_update_test = 3
corr_update_train = 2
lmbda_prox = 0.02
rescaling_init_val = 1.0
spams_init = 1
multi_theta = 1
center_windows = 1
diag_rescale_gamma = 1
diag_rescale_patch = 1
patch_size = 56
nu_init = 1
mask_windows = 1
multi_std = 0
train_batch = 25
aug_scale = 0

pad_block = 1
pad_patch = 0
no_pad = False
custom_pad = None
stride_test = 12
test_batch = 10

model_name = 'trained_model/gray/corr_update%3_freq%6_kernel_size%9_lr_step%80_noise_level%50_train_batch%25_/ckpt'
test_im # complete
train_im # complete

sigma = 50
noise_std = sigma / 255

lr = 1e-5
epochs = 2 # epoch is going through all the patches of the image
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
criterion = nn.MSELoss(reduction='sum')

out_dir = os.path.join(model_name)
ckpt_path = os.path.join(out_dir)
checkpoint = torch.load(ckpt_path, map_location=device)

params_test = ListaParams(kernel_size=kernel_size, num_filters=num_filters, stride=stride,
					 unfoldings=unfoldings, freq=freq_corr_update_test, corr_update=corr_update_test,
					 lmbda_init=lmbda_prox, h=rescaling_init_val, spams=spams_init,
					 multi_lmbda=multi_theta,
					 center_windows=center_windows, std_gamma=diag_rescale_gamma,
					 std_y=diag_rescale_patch, block_size=patch_size, nu_init=nu_init,
					 mask=mask_windows, multi_std=multi_std)

block_params = {
			'crop_out_blocks': 0,
			'ponderate_out_blocks': 1,
			'sum_blocks': 0,
			'pad_even': 1,  # otherwise pad with 0 for las
			'centered_pad': 0,  # corner pixel have only one estimate
			'pad_block': pad_block,  # pad so each pixel has S**2 estimate
			'pad_patch': pad_patch,  # pad so each pixel from the image has at least S**2 estimate from 1 block
			'no_pad': no_pad,
			'custom_pad': custom_pad,
			'avg': 1}

l = kernel_size // 2
mask = F.conv_transpose2d(torch.ones(1, 1, patch_size - 2 * l, patch_size - 2 * l),
					  torch.ones(1, 1, kernel_size, kernel_size))
mask /= mask.max()
mask = mask.to(device=device)

# load model
model = Lista(params_test).to(device)
model.load_state_dict(checkpoint['state_dict'],strict=True)
model.eval()

img = cv2.imread(test_im, 0)  # cv2.IMREAD_GRAYSCALE
img = np.expand_dims(img, axis=2)
img = np.float32(img/255.)
np.random.seed(10)
noisy_img = img + np.random.normal(0, noise_std, img.shape)
noisy_img = torch.from_numpy(np.ascontiguousarray(noisy_img)).permute(2, 0, 1).float().unsqueeze(0)
img = torch.from_numpy(np.ascontiguousarray(img)).permute(2, 0, 1).float().unsqueeze(0)

noisy_img = noisy_img.to(device)
img = img.to(device)

# denoise the image to compare PSNR before and after adaptation
with torch.no_grad():
	block = block_module(patch_size, stride_test, kernel_size, block_params)
	batch_noisy_blocks = block._make_blocks(noisy_img)
	patch_loader = torch.utils.data.DataLoader(batch_noisy_blocks, batch_size=test_batch, drop_last=False)
	batch_out_blocks = torch.zeros_like(batch_noisy_blocks)

	for i, inp in enumerate(patch_loader):
	  id_from, id_to = i * patch_loader.batch_size, (i + 1) * patch_loader.batch_size
	  batch_out_blocks[id_from:id_to] = model(inp)
	  
	output = block._agregate_blocks(batch_out_blocks)
	psnr_batch = -10 * torch.log10((output.clamp(0., 1.) - img).pow(2).flatten(2, 3).mean(2)).mean()
	psnr_before = psnr_batch.item()

# external adaptation

model.train()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

# open train image
train_im = cv2.imread(train_im, 0)  # cv2.IMREAD_GRAYSCALE
train_im = np.expand_dims(train_im, axis=2)
train_im = np.float32(train_im/255.)
train_im = torch.from_numpy(np.ascontiguousarray(train_im)).permute(2, 0, 1).float().unsqueeze(0)
train_im = train_im.to(device)

for epoch in range(epochs):

	block = block_module(patch_size, stride_test, kernel_size, block_params)
	batch_noisy_blocks = block._make_blocks(train_im)
	patch_loader = torch.utils.data.DataLoader(batch_noisy_blocks, batch_size=5, drop_last=False)
	batch_out_blocks = torch.zeros_like(batch_noisy_blocks)
	
	# add noise to each patch and optimize with it and the noisy patch
	for i, inp in enumerate(tqdm(patch_loader)):

	  noise = torch.randn_like(inp) * noise_std
	  noisy_inp = inp + noise
	  optimizer.zero_grad()
	  retrain_output = model(noisy_inp)
	  loss = (mask * (retrain_output - inp)).pow(2).sum() / retrain_output.shape[0]
	  loss.backward()
	  optimizer.step()

# denoise the image again after adaptation
model.eval()
with torch.no_grad():
	block = block_module(patch_size, stride_test, kernel_size, block_params)
	batch_noisy_blocks = block._make_blocks(noisy_img)
	patch_loader = torch.utils.data.DataLoader(batch_noisy_blocks, batch_size=test_batch, drop_last=False)
	batch_out_blocks = torch.zeros_like(batch_noisy_blocks)

	for i, inp in enumerate(patch_loader):  # if it doesnt fit in memory
	  id_from, id_to = i * patch_loader.batch_size, (i + 1) * patch_loader.batch_size
	  batch_out_blocks[id_from:id_to] = model(inp)

	output = block._agregate_blocks(batch_out_blocks)
	psnr_batch = -10 * torch.log10((output.clamp(0., 1.) - img).pow(2).flatten(2, 3).mean(2)).mean()
	psnr_after = psnr_batch.item()

print('Before adaptation - PSNR: {:.2f} dB; After adaptation - _SNR: {:.2f} dB'.format(psnr_before, psnr_after))