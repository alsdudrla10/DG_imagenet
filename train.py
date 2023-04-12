import click
import os
import classifier_lib
import torch
import numpy as np
import dnnlib
from guided_diffusion.image_datasets import load_data_latent
import random

@click.command()
@click.option('--savedir',                    help='Save directory',  metavar='PATH',                 type=str, required=True,        default="/checkpoints/discriminator")
@click.option('--gendir',                     help='Fake sample absolute directory', metavar='PATH',  type=str, required=True,        default="/gen_latents")
@click.option('--datadir',                    help='Real sample absolute directory', metavar='PATH',  type=str, required=True,        default="/real_latents")
@click.option('--img_resolution',             help='Image resolution', metavar='INT',                 type=click.IntRange(min=1),     default=256)

@click.option('--pretrained_classifier_ckpt', help='Path of ADM classifier',  metavar='STR',          type=str,                       default='/checkpoints/ADM_classifier/32x32_classifier.pt')
@click.option('--batch_size',                 help='Num samples', metavar='INT',                      type=click.IntRange(min=1),     default=1024)
@click.option('--epoch',                      help='Num samples', metavar='INT',                      type=click.IntRange(min=1),     default=60)
@click.option('--lr',                         help='Learning rate', metavar='FLOAT',                  type=click.FloatRange(min=0),   default=3e-4)
@click.option('--device',                     help='Device', metavar='STR',                           type=str,                       default='cuda:0')

def main(**kwargs):
    opts = dnnlib.EasyDict(kwargs)
    savedir = os.getcwd() + opts.savedir
    os.makedirs(savedir, exist_ok=True)

    ## Prepare fake/real data
    gen_train_loader = load_data_latent(
        data_dir=opts.gendir,
        batch_size=int(batch_size / 2),
        image_size=32,
        class_cond=True,
        random_crop=False,
        random_flip=False,
    )
    real_train_loader = load_data_latent(
        data_dir=opts.datadir,
        batch_size=int(batch_size / 2),
        image_size=32,
        class_cond=True,
        random_crop=False,
        random_flip=False,
    )

    ## Extractor & Disciminator
    pretrained_classifier_ckpt = os.getcwd() + opts.pretrained_classifier_ckpt
    classifier  = classifier_lib.load_classifier(pretrained_classifier_ckpt, 32, opts.device, eval=False)
    discriminator = classifier_lib.load_discriminator(None, opts.device, True, eval=False)

    ## Prepare training
    vpsde = classifier_lib.vpsde()
    optimizer = torch.optim.Adam(discriminator.parameters(), lr=3e-4, weight_decay=1e-7)
    loss = torch.nn.BCELoss()

    iterator = iter(gen_train_loader)
    ## Training
    for i in range(epoch):
        outs = []
        cors = []
        num_data = 0
        for data in real_train_loader:
            optimizer.zero_grad()
            real_inputs, real_condition = data
            real_condition = real_condition.to(device)
            real_inputs = real_inputs.to(device)
            real_labels = torch.ones(real_inputs.shape[0]).to(device)

            ## Real data perturbation
            real_t, _ = vpsde.get_diffusion_time(real_inputs.shape[0], real_inputs.device, 1e-5, importance_sampling=True)
            mean, std = vpsde.marginal_prob(real_t)
            z = torch.randn_like(real_inputs)
            perturbed_real_inputs = mean[:, None, None, None] * real_inputs + std[:, None, None, None] * z

            ## Fake data
            try:
                fake_inputs, fake_condition = next(iterator)
            except:
                iterator = iter(gen_train_loader)
                fake_inputs, fake_condition = next(iterator)
            fake_condition = fake_condition.to(device)
            fake_inputs = fake_inputs.to(device)
            fake_labels = torch.zeros(fake_inputs.shape[0]).to(device)

            ## Fake data perturbation
            fake_t, _ = vpsde.get_diffusion_time(fake_inputs.shape[0], fake_inputs.device, 1e-5, importance_sampling=True)
            mean, std = vpsde.marginal_prob(fake_t)
            z = torch.randn_like(fake_inputs)
            perturbed_fake_inputs = mean[:, None, None, None] * fake_inputs + std[:, None, None, None] * z

            ## Combine data
            inputs = torch.cat([real_inputs, fake_inputs])
            perturbed_inputs = torch.cat([perturbed_real_inputs, perturbed_fake_inputs])
            labels = torch.cat([real_labels, fake_labels])
            condition = torch.cat([real_condition, fake_condition])
            t = torch.cat([real_t, fake_t])
            c = list(range(inputs.shape[0]))
            random.shuffle(c)
            inputs, perturbed_inputs, labels, condition, t = inputs[c], perturbed_inputs[c], labels[c], condition[c], t[c]


            ## Forward
            with torch.no_grad():
                pretrained_feature = classifier(perturbed_inputs, timesteps=t, feature=True)
            label_prediction = discriminator(pretrained_feature, t, condition=condition, sigmoid=True).view(-1)
            label_prediction = torch.clip(label_prediction, min=1e-5, max=1 - 1e-5)

            ## Backward
            out = loss(label_prediction, labels)
            out.backward()
            optimizer.step()

            ## Report
            cor = ((label_prediction > 0.5).float() == labels).float().mean()
            outs.append(out.item())
            cors.append(cor.item())
            num_data += inputs.shape[0]
            print(f"{i}-th epoch BCE loss: {np.mean(outs)}, correction rate: {np.mean(cors)}")

        ## Save
        torch.save(shallow_discriminator.state_dict(), savedir + f"/discriminator_{i+1}.pt")

#----------------------------------------------------------------------------
if __name__ == "__main__":
    main()
#----------------------------------------------------------------------------