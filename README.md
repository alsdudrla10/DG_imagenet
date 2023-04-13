## Refining Generative Process with Discriminator Guidance in Score-based Diffusion Models (DG) (under review) <br><sub>Official PyTorch implementation of the Discriminator Guidance </sub>
**[Dongjun Kim](https://sites.google.com/view/dongjun-kim) \*, [Yeongmin Kim](https://sites.google.com/view/yeongmin-space/%ED%99%88) \*, Se Jung Kwon, Wanmo Kang, and Il-Chul Moon**   
<sup> * Equal contribution </sup> <br>

| [paper](https://arxiv.org/abs/2211.17091) |  <br>
**Camera-ready final version will be released within this month. Stay tuned!** <br>
**See [https://github.com/alsdudrla10/DG]([https://github.com/alsdudrla10/DG_imagenet](https://github.com/alsdudrla10/DG)) for the Cifar-10 code release.** <br>

## Overview
![Teaser image](./figures/Figure1_v2.PNG)

## Step-by-Step running of Discriminator Guidance

### 1) Fake sample generation
  - command: python3 sample.py --LT_cfg=1.5 --ST_cfg=1.5 --time_min=1000

### 2) Prepare real data
  - Download ImageNet2012 
  - save_directory: data/ImageNet2012/train/n01440764/n01440764_9981.JPEG

### 3) Latent extraction
  - For efficient training, we save vae's latent space
  - command: python3 data_tool.py

### 4) Prepare pretrained classifier
  - download [here](https://drive.google.com/drive/folders/1_NlbYX9l7yW_y8Wnmb2Diyy59j95hznM)
  - save_directory: DG/checkpoints/ADM_classifier/32x32_classifier.pt

### 5) Discriminator training
  - command: python3 train.py
  - downalod checkpoint [here](https://drive.google.com/drive/folders/1_NlbYX9l7yW_y8Wnmb2Diyy59j95hznM)

### 6) Generation with Discriminator Guidance
  - command: python3 sample.py
  


## Results on data diffusion
|FID-50k |ImageNet256|
|------------|------------|
|Privious SOTA|4.59|
|+ DG|3.18|

## Results on latent diffusion
|FID-50k|ImageNet256|
|------------|------------|
|Privious SOTA|2.27|
|+ DG|1.83|


## Samples from ADM
![Teaser image](./figures/Figure2_v2.PNG)
