import os.path
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.modules.loss import _Loss
from collections import OrderedDict

from utils import utils_image as util
from models.network_dncnn import DnCNN

# parameters 
n_channels = 1
model_pool = 'model_zoo'
testsets = 'testsets'
model_name = 'dncnn_25'
sigma = 25
testset_name = 'bsd68'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
criterion = nn.MSELoss(reduction='sum')
lr = 1e-5
epochs = 100

test_results = OrderedDict()
test_results['psnr_before'] = []
test_results['psnr_after'] = []

model_path = os.path.join(model_pool, model_name+'.pth')
test_paths = os.path.join(testsets, testset_name)
test_paths = util.get_image_paths(test_paths)

# training loop
for idx, img in enumerate(test_paths):
  
  start_time = time.time()

  # load model
  model = DnCNN(in_nc=n_channels, out_nc=n_channels, nc=64, nb=17, act_mode='R')
  model.load_state_dict(torch.load(model_path), strict=True)
  model = model.to(device)
  model.eval()

  # load test image  
  img_name, ext = os.path.splitext(os.path.basename(img))
  x = util.imread_uint(img, n_channels=n_channels)
  orig_im = x.squeeze()
  x = util.uint2single(x)
  np.random.seed(seed=0)  # for reproducibility
  y = x + np.random.normal(0, sigma/255., x.shape) # add gaussian noise
  y = util.single2tensor4(y)
  y = y.to(device)
    
  # denoise image using the universal network
  with torch.no_grad():
    x_ = model(y)

  # compute PSNR
  denoised_im = util.tensor2uint(x_)
  prev_psnr = util.calculate_psnr(denoised_im, orig_im, border=0)
  test_results['psnr_before'].append(prev_psnr)

  denoised = x_.view(x.shape[0], x.shape[1], 1)
  denoised = denoised.cpu()
  denoised = denoised.detach().numpy().astype(np.float32)

  # internal adaptation
  model.train()
  optimizer = optim.Adam(model.parameters(), lr=lr)
  for epoch in range(epochs):

      # add noise to the denoised image
      y2 = denoised + np.random.normal(0, sigma/255., x.shape)
      y2 = util.single2tensor4(y2)
      y2 = y2.to(device)

	  # optimize using the denoised image + the noisy denoised image
      optimizer.zero_grad()
      loss = criterion(model(y2), x_)
      loss.backward()
      optimizer.step()
    
    # denoise the image again after adaptation
  model.eval()
  with torch.no_grad():
    x_ = model(y)

  denoised_im = util.tensor2uint(x_)
  psnr = util.calculate_psnr(denoised_im, orig_im, border=0)
  test_results['psnr_after'].append(psnr)

  elapsed_time = time.time() - start_time
  print('{:s} - Before adaptation - PSNR: {:.2f} dB; After adaptation - PSNR: {:.2f} dB.'.format(img_name+ext, prev_psnr, psnr))

avg_psnr_after = sum(test_results['psnr_after']) / len(test_results['psnr_after'])
avg_psnr_before = sum(test_results['psnr_before']) / len(test_results['psnr_before'])

print('Average PSNR: - Before adaptation: PSNR: {:.2f} dB; After adaptation: - PSNR: {:.2f} dB.'.format(avg_psnr_before,avg_psnr_after))