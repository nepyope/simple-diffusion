import torch
import os
import numpy as np
from PIL import Image
from einops import rearrange
from pytorch_lightning import seed_everything
from vae import AutoencoderKL
import gdown
import pafy
import cv2
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-url", help="url of the youtube video")
args = parser.parse_args()

if not os.path.exists("vae.ckpt"):
    os.system('gdown 1-tgv_x4jIrsGkkw4BTbwbiuVjy_NGjn2')

ddconfig = {'double_z': True, 'z_channels': 4, 'resolution': 256, 'in_channels': 3, 'out_ch': 3, 'ch': 128, 'ch_mult': [1, 2, 4, 4], 'num_res_blocks': 2, 'attn_resolutions': [], 'dropout': 0.0}

seed_everything(42)

def load_frame(frame):
    image = Image.fromarray(frame).convert("RGB")
    w, h = image.size
    print(f"loaded input frame of size ({w}, {h})")
    w, h = map(lambda x: x - x % 32, (w, h))  # resize to integer multiple of 32
    image = image.resize((w, h), resample=Image.LANCZOS)
    image = np.array(image).astype(np.float32) / 255.0
    image = image[None].transpose(0, 3, 1, 2)
    image = torch.from_numpy(image).to(device)
    return 2.*image - 1. 

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

vae = AutoencoderKL(lossconfig={'target':'torch.nn.Identity'},embed_dim = 4,ddconfig=ddconfig)
vae.load_state_dict(torch.load('vae.ckpt'))
vae.cuda()

url = args.url
vPafy = pafy.new(url)
play = vPafy.getbest(preftype="mp4")

cap = cv2.VideoCapture(play.url)

with torch.no_grad():
    for i in range(20):
        base_count = len(os.listdir('dump'))
        ret,frame = cap.read()

        z=vae.encode(load_frame(frame))
        torch.save(z, f"dump/{base_count:05}.pt")