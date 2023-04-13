## Assume your Real Data saved in the form of /data/ImageNet2012/train/n04125021/n04125021_10433.JPEG ...

from diffusers.models import AutoencoderKL
from torchvision import transforms
import numpy as np
from PIL import Image
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import torch
import os
import io
import tensorflow as tf
from glob import glob


def center_crop_arr(pil_image, image_size):
    """
    Center cropping implementation from ADM.
    https://github.com/openai/guided-diffusion/blob/8fb3ad9197f16bbc40620447b2742e13458d2831/guided_diffusion/image_datasets.py#L126
    """
    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )

    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )

    arr = np.array(pil_image)
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2
    return Image.fromarray(arr[crop_y: crop_y + image_size, crop_x: crop_x + image_size])

def save_latents(datadir, savedir, batch_size):
    device = "cuda"
    vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-ema").to(device)
    folders = glob(savedir +"*")
    folders.sort()

    transform = transforms.Compose([
        transforms.Lambda(lambda pil_image: center_crop_arr(pil_image, 256)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)
    ])
    dataset = ImageFolder(datadir, transform=transform)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        drop_last=False
    )
    count = 0
    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        with torch.no_grad():
            x = vae.encode(x).latent_dist.sample().mul_(0.18215)
            x = x.to("cpu").numpy()
        print(y)
        for index in range(len(x)):
            with tf.io.gfile.GFile(os.path.join(folders[y[index]], f"samples_{count}.npz"), "wb") as fout:
                io_buffer = io.BytesIO()
                np.savez_compressed(io_buffer, samples=x[index], label=y[index].cpu().numpy())
                fout.write(io_buffer.getvalue())
            count += 1

def sample2datafoler(sampledir, datadir):
    sub_folders = os.listdir(datadir)
    sub_folders.sort()
    npzs = glob(sampledir + "*.npz")
    count = 0
    for npz in npzs:
        file = np.load(npz)
        images =file["samples"]
        labels = file["label"]
        labels = labels[:int(len(labels)/2)]
        for i in range(len(labels)):
            save_name = datadir + sub_folders[labels[i]] + f"/{sub_folders[labels[i]]}_{count}.JPEG"
            im = Image.fromarray(images[i])
            im.save(save_name)
            count += 1

def make_folders(datadir, savedir):
    sub_folders = os.listdir(datadir)
    for sub_folder in sub_folders:
        abs_sub_folder = savedir + sub_folder
        os.makedirs(abs_sub_folder, exist_ok=True)



## Set Real / Fake data
realdir = os.getcwd() + "data/ImageNet2012/train/"
sampledir = os.getcwd() + "/samples/"
fakedir = os.getcwd() + "/genpath/ImageNet2012/train/"
os.makedirs(fakedir, exist_ok=True)
make_folders(realdir, fakedir)
sample2datafoler(sampledir, fakedir)

## Extract latents and save
gen_latents = os.getcwd() + "/gen_latents/"
real_latents = os.getcwd() + "/real_latents/"
os.makedirs(gen_latents, exist_ok=True)
os.makedirs(real_latents, exist_ok=True)
make_folders(realdir, gen_latents)
make_folders(realdir, real_latents)
save_latents(realdir, real_latents, 128)
save_latents(fakedir, gen_latents, 128)