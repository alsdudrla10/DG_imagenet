import torch
from guided_diffusion.script_util import create_classifier
import numpy as np

def get_adm_discriminator(latent_extractor_ckpt, discriminator_ckpt, condition=True, img_resolution=32, device='cuda', enable_grad=True):
    classifier = load_classifier(latent_extractor_ckpt, img_resolution, device, eval=True)
    discriminator = load_discriminator(discriminator_ckpt, device, condition, eval=True)
    def evaluate(perturbed_inputs, timesteps=None, condition=None):
        with torch.enable_grad() if enable_grad else torch.no_grad():
            pretrained_feature = classifier(perturbed_inputs, timesteps=timesteps, feature=True)
            prediction = discriminator(pretrained_feature, timesteps, sigmoid=True, condition=condition).view(-1)
        return prediction
    return evaluate

def load_classifier(ckpt_path, img_resolution, device, eval=True):
    latent_extractor_args = dict(
      image_size=img_resolution,
      classifier_use_fp16=False,
      classifier_width=128,
      classifier_depth=2,
      classifier_attention_resolutions="32,16,8",
      classifier_use_scale_shift_norm=True,
      classifier_resblock_updown=True,
      classifier_pool="attention",
      out_channels=1000,
      in_channels=4,
    )
    adm_classifier = create_classifier(**latent_extractor_args)
    adm_classifier.to(device)
    classifier_state = torch.load(ckpt_path, map_location="cpu")
    adm_classifier.load_state_dict(classifier_state)
    if eval:
      adm_classifier.eval()
    return adm_classifier

def load_discriminator(ckpt_path, device, condition, eval=False, channel=384):
    latent_extractor_args = dict(
      image_size=8,
      classifier_use_fp16=False,
      classifier_width=128,
      classifier_depth=2,
      classifier_attention_resolutions="32,16,8",
      classifier_use_scale_shift_norm=True,
      classifier_resblock_updown=True,
      classifier_pool="attention",
      out_channels = 1,
      in_channels = channel,
      condition = condition,
    )
    discriminator = create_classifier(**latent_extractor_args)
    discriminator.to(device)
    if ckpt_path != None:
        discriminator_state = torch.load(ckpt_path, map_location="cpu")
        discriminator.load_state_dict(discriminator_state)
    if eval:
        discriminator.eval()
    return discriminator

class vpsde():
    def __init__(self):
        self.beta_0 = 0.1
        self.beta_1 = 20.
        self.s = 0.008
        self.f_0 = np.cos(self.s / (1. + self.s) * np.pi / 2.) ** 2

    @property
    def T(self):
        return 1

    def compute_tau(self, std_wve_t):
        tau = -self.beta_0 + torch.sqrt(self.beta_0 ** 2 + 2. * (self.beta_1 - self.beta_0) * torch.log(1. + std_wve_t ** 2))
        tau /= self.beta_1 - self.beta_0
        return tau

    def marginal_prob(self, t):
        log_mean_coeff = -0.25 * t ** 2 * (self.beta_1 - self.beta_0) - 0.5 * t * self.beta_0
        mean = torch.exp(log_mean_coeff)
        std = torch.sqrt(1. - torch.exp(2. * log_mean_coeff))
        return mean, std

    def transform_unnormalized_wve_to_normalized_vp(self, t, std_out=False):
        tau = self.compute_tau(t)
        mean_vp_tau, std_vp_tau = self.marginal_prob(tau)
        if std_out:
            return mean_vp_tau, std_vp_tau, tau
        return mean_vp_tau, tau

    def compute_t_cos_from_t_lin(self, t_lin):
        sqrt_alpha_t_bar = torch.exp(-0.25 * t_lin ** 2 * (self.beta_1 - self.beta_0) - 0.5 * t_lin * self.beta_0)
        time = torch.arccos(np.sqrt(self.f_0) * sqrt_alpha_t_bar)
        t_cos = self.T * ((1. + self.s) * 2. / np.pi * time - self.s)
        return t_cos

    def get_diffusion_time(self, batch_size, batch_device, t_min, importance_sampling=True):
        if importance_sampling:
            Z = self.normalizing_constant(t_min)
            u = torch.rand(batch_size, device=batch_device)
            return (-self.beta_0 + torch.sqrt(self.beta_0 ** 2 + 2 * (self.beta_1 - self.beta_0) *
                    torch.log(1. + torch.exp(Z * u + self.antiderivative(t_min))))) / (self.beta_1 - self.beta_0), Z.detach()
        else:
            return torch.rand(batch_size, device=batch_device) * (self.T - t_min) + t_min, 1

    def antiderivative(self, t, stabilizing_constant=0.):
        if isinstance(t, float) or isinstance(t, int):
            t = torch.tensor(t).float()
        return torch.log(1. - torch.exp(- self.integral_beta(t)) + stabilizing_constant) + self.integral_beta(t)

    def normalizing_constant(self, t_min):
        return self.antiderivative(self.T) - self.antiderivative(t_min)

    def integral_beta(self, t):
        return 0.5 * t ** 2 * (self.beta_1 - self.beta_0) + t * self.beta_0