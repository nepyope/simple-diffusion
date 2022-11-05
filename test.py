#encodes test.jpg, then decodes and saves it to reconstructed.jpg

import torch
import os
import numpy as np

from PIL import Image
from einops import rearrange
from pytorch_lightning import seed_everything
from vae import AutoencoderKL
import gdown

if not os.path.exists("vae.ckpt"):
    os.system('gdown 1-tgv_x4jIrsGkkw4BTbwbiuVjy_NGjn2')

ddconfig = {'double_z': True, 'z_channels': 4, 'resolution': 256, 'in_channels': 3, 'out_ch': 3, 'ch': 128, 'ch_mult': [1, 2, 4, 4], 'num_res_blocks': 2, 'attn_resolutions': [], 'dropout': 0.0}

def load_img(path):
    image = Image.open(path).convert("RGB")
    w, h = image.size
    print(f"loaded input image of size ({w}, {h}) from {path}")
    w, h = map(lambda x: x - x % 32, (w, h))  # resize to integer multiple of 32
    image = image.resize((w, h), resample=Image.LANCZOS)
    image = np.array(image).astype(np.float32) / 255.0
    image = image[None].transpose(0, 3, 1, 2)
    image = torch.from_numpy(image).to(device)
    return 2.*image - 1. 

seed_everything(42)

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

vae = AutoencoderKL(lossconfig={'target':'torch.nn.Identity'},embed_dim = 4,ddconfig=ddconfig)
vae.load_state_dict(torch.load('vae.ckpt'))
vae.cuda()
init_image = load_img('test.jpg').to(device)

with torch.no_grad():
    z=vae.encode(init_image)
    x_samples_ddim = vae.decode(z.sample())

    x_samples_ddim = torch.clamp((x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0)
    x_samples_ddim = x_samples_ddim.cpu().permute(0, 2, 3, 1).numpy()

    x_checked_image = x_samples_ddim

    x_checked_image_torch = torch.from_numpy(x_checked_image).permute(0, 3, 1, 2)
    x_sample = 255. * rearrange(x_checked_image_torch[0].cpu().numpy(), 'c h w -> h w c')
    img = Image.fromarray(x_sample.astype(np.uint8))
    img.save('reconstructed.jpg')