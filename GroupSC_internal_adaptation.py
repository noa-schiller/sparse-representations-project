import dataloaders_v2
import numpy as np
from tqdm import tqdm
import os, sys
import time

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

model_name = 'trained_model/gray/corr_update%3_freq%6_kernel_size%9_lr_step%80_noise_level%25_train_batch%25_/ckpt'
sigma = 25
noise_std = sigma / 255
dataset_name='BSD68'

lr = 1e-5
epochs = 2 # epoch is going through all the patches of the image
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
criterion = nn.MSELoss(reduction='sum')

out_dir = os.path.join(model_name)
ckpt_path = os.path.join(out_dir)
checkpoint = torch.load(ckpt_path, map_location=device)

data_path = 'datasets'
test_path = [f'{data_path}/{dataset_name}/']
train_path = [f'{data_path}/BSD400/']
val_path = train_path
loaders = dataloaders_v2.get_dataloaders(train_path, test_path,train_path, crop_size=patch_size,
								  batch_size=train_batch, downscale=aug_scale,concat=1)

loader = loaders['test']

params = ListaParams(kernel_size=kernel_size, num_filters=num_filters, stride=stride,
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
			'pad_even': 1, 
			'centered_pad': 0,
			'pad_block': pad_block,
			'pad_patch': pad_patch,
			'no_pad': no_pad,
			'custom_pad': custom_pad,
			'avg': 1}

l = kernel_size // 2
mask = F.conv_transpose2d(torch.ones(1, 1, patch_size - 2 * l, patch_size - 2 * l),
					  torch.ones(1, 1, kernel_size, kernel_size))
mask /= mask.max()
mask = mask.to(device=device)

psnr_before = []
psnr_after = []

for idx,batch in enumerate(tqdm(loader)):

	start_time = time.time()

	# load model
	model = Lista(params).to(device)
	model.load_state_dict(checkpoint['state_dict'],strict=True)
	model.eval()

	batch = batch.to(device=device)
	torch.manual_seed(0)  # for reproducibility
	noise = torch.randn_like(batch) * noise_std
	noisy_batch = batch + noise

	# denoise the image by denoising each patch and reconstructe the image  
	with torch.no_grad():
	  block = block_module(patch_size, stride_test, kernel_size, block_params)
	  batch_noisy_blocks = block._make_blocks(noisy_batch)
	  patch_loader = torch.utils.data.DataLoader(batch_noisy_blocks, batch_size=test_batch, drop_last=False)
	  batch_out_blocks = torch.zeros_like(batch_noisy_blocks)

	  for i, inp in enumerate(patch_loader):
		  id_from, id_to = i * patch_loader.batch_size, (i + 1) * patch_loader.batch_size
		  batch_out_blocks[id_from:id_to] = model(inp)

	  output = block._agregate_blocks(batch_out_blocks)
	  psnr_batch = -10 * torch.log10((output.clamp(0., 1.) - batch).pow(2).flatten(2, 3).mean(2)).mean()
	  psnr_before.append(psnr_batch.item())

	# internal adaptation
	model.train()
	optimizer = torch.optim.Adam(model.parameters(), lr=lr)

	for epoch in range(epochs):
	
	  block = block_module(patch_size, stride_test, kernel_size, block_params)
	  batch_noisy_blocks = block._make_blocks(output)
	  patch_loader = torch.utils.data.DataLoader(batch_noisy_blocks, batch_size=test_batch, drop_last=False)
	  batch_out_blocks = torch.zeros_like(batch_noisy_blocks)

	  # add noise to each patch and optimize with it and the noisy patch
	  for inp in tqdm(patch_loader):

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
	  batch_noisy_blocks = block._make_blocks(noisy_batch)
	  patch_loader = torch.utils.data.DataLoader(batch_noisy_blocks, batch_size=test_batch, drop_last=False)
	  batch_out_blocks = torch.zeros_like(batch_noisy_blocks)

	  for i, inp in enumerate(patch_loader):  # if it doesnt fit in memory
		  id_from, id_to = i * patch_loader.batch_size, (i + 1) * patch_loader.batch_size
		  batch_out_blocks[id_from:id_to] = model(inp)

	  output = block._agregate_blocks(batch_out_blocks)

	psnr_batch = -10 * torch.log10((output.clamp(0., 1.) - batch).pow(2).flatten(2, 3).mean(2)).mean()
	psnr_after.append(psnr_batch.item())

	elapsed_time = time.time() - start_time
	print('{} - {:.2f} second - Before adaptation: {:.2f} dB; After adaptation {:.2f} dB'.format(idx, elapsed_time, psnr_before[-1], psnr_after[-1]))

avg_psnr_before = np.mean(psnr_before)
avg_psnr_after = np.mean(psnr_after)
print('Average PSNR - {} - Before adaptation: {:.2f} dB; After adaptation {:.2f} dB'.format(idx, avg_psnr_before, avg_psnr_after))