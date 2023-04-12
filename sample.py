import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
from torchvision.utils import save_image
from diffusion import create_diffusion
from diffusers.models import AutoencoderKL
from download import find_model
from models import DiT_models
import argparse
import numpy as np
import tensorflow as tf
import os
import io
import classifier_lib

def main(args):
    sample_dir = os.getcwd() + args.sample_dir
    model_dir = os.getcwd() + args.model_dir
    tf.io.gfile.makedirs(sample_dir)
    device = "cuda"

    ## Load backbone score model:
    latent_size = args.image_size // 8
    model = DiT_models[args.model](input_size=latent_size, num_classes=1000).to(device)

    ## Auto-download a pre-trained model or load a custom DiT checkpoint from train.py:
    ckpt_path = args.ckpt or f"DiT-XL-2-{args.image_size}x{args.image_size}.pt"
    state_dict = find_model(ckpt_path)
    model.load_state_dict(state_dict)
    model.eval()
    diffusion = create_diffusion(str(args.num_sampling_steps))
    vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-{args.vae}").to(device)

    ## Load discriminator
    classifier_path = model_dir + "/ADM_classifier/32x32_classifier.pt"
    discriminator_ckpt_path = model_dir + f"/discriminator/discriminator_{args.ck_epoch}.pt"
    discriminator = classifier_lib.get_adm_discriminator(classifier_path, discriminator_ckpt_path, True, 32, device='cuda', enable_grad=True)

    ## Discriminator Guidance
    def cond_fn(x, t, y=None, **kwargs):
        if t[0] > args.time_min:
            half = x[: len(x) // 2]
            with torch.enable_grad():
                x_in = half.detach().requires_grad_(True)
                pr = discriminator(x_in, t[:int(t.shape[0] / 2)] / 999, condition=y[:int(t.shape[0] / 2)])
                pr = torch.clip(pr, min=1e-5, max=1 - 1e-5)

                log_density_ratio = torch.log(pr) - torch.log(1 - pr)
                dg = torch.autograd.grad(log_density_ratio.sum(), x_in)[0] * args.dg_scale

            dg = torch.cat([dg, dg], dim=0)
            return dg
        else:
            return torch.zeros_like(x)

    ## Sampling
    count = 0
    while True:
        y = torch.randint(low=0, high=1000, size=(args.batch_size,), device=device)
        n = y.shape[0]
        z = torch.randn(n, 4, latent_size, latent_size, device=device)

        # Setup classifier-free guidance:
        z = torch.cat([z, z], 0)
        y_null = torch.tensor([1000] * n, device=device)
        y = torch.cat([y, y_null], 0)
        model_kwargs = dict(y=y, LT_cfg=args.LT_cfg, ST_cfg=args.ST_cfg, branch_time=args.branch_time)

        # Sample images:
        samples = diffusion.p_sample_loop(model.forward_with_dg, z.shape, z, clip_denoised=False, model_kwargs=model_kwargs, progress=True, device=device, cond_fn=cond_fn)
        samples, _ = samples.chunk(2, dim=0)  # Remove null class samples
        with torch.no_grad():
            samples = vae.decode(samples / 0.18215).sample

        # Save and display images:
        r = np.random.randint(1000000)
        if count == 0:
            save_image(samples, sample_dir + f"/sample_{r}.png", nrow=int(np.sqrt(samples.shape[0])), normalize=True, value_range=(-1, 1))
        images_np = (samples * 127.5 + 128).clip(0, 255).to(torch.uint8).permute(0, 2, 3, 1).cpu().numpy()
        with tf.io.gfile.GFile(sample_dir + f"/sample_{r}.npz", "wb") as fout:
            io_buffer = io.BytesIO()
            np.savez_compressed(io_buffer, samples=images_np, label=y.cpu().numpy())
            fout.write(io_buffer.getvalue())
        count += args.batch_size
        print(count, " sampled")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, choices=list(DiT_models.keys()), default="DiT-XL/2")
    parser.add_argument("--vae", type=str, choices=["ema", "mse"], default="mse")
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--num-sampling-steps", type=int, default=250)
    parser.add_argument("--ckpt", type=str, default=None, help="Optional path to a DiT checkpoint (default: auto-download a pre-trained DiT-XL/2 model).")

    parser.add_argument("--LT_cfg", type=float, default=1.25)
    parser.add_argument("--ST_cfg", type=float, default=3.0)
    parser.add_argument("--branch_time", type=int, default=200)
    parser.add_argument("--time_min", type=int, default=100)

    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--sample_dir", type=str, default="/samples")
    parser.add_argument("--model_dir", type=str, default="/pretrained_models")
    parser.add_argument("--ck_epoch", type=int, default=7)
    parser.add_argument("--dg_scale", type=float, default=1.0)

    args = parser.parse_args()
    main(args)
