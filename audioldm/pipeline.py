

import os

import argparse
import yaml
import torch

from audioldm import LatentDiffusion
from audioldm.utils import default_audioldm_config

import time

def make_batch_for_text_to_audio(text, batchsize=2):
    text = [text] * batchsize
    if batchsize < 2:
        print("Warning: Batchsize must be at least 2. Batchsize is set to 2.")
    fbank = torch.zeros((batchsize, 1024, 64))  # Not used, here to keep the code format
    stft = torch.zeros((batchsize, 1024, 512))  # Not used
    waveform = torch.zeros((batchsize, 160000))  # Not used
    fname = ["%s.wav" % x for x in range(batchsize)]
    batch = (
        fbank,
        stft,
        None,
        fname,
        waveform,
        text,
    )  
    return batch

def text_to_audio(text, batchsize=2, guidance_scale=2.5, n_gen=1, config=None):
    if(torch.cuda.is_available()):
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")
        
    if(config is not None):
        assert type(config) is str
        config = yaml.load(open(config, "r"), Loader=yaml.FullLoader)
    else:
        config = default_audioldm_config()

    # Use text as condition instead of using waveform during training
    config["model"]["params"]["device"] = device
    config["model"]["params"]["cond_stage_key"] = "text"

    # No normalization here
    latent_diffusion = LatentDiffusion(**config["model"]["params"])

    resume_from_checkpoint = "./ckpt/ldm_trimmed.ckpt"

    checkpoint = torch.load(resume_from_checkpoint)
    latent_diffusion.load_state_dict(checkpoint["state_dict"])

    latent_diffusion.eval()
    latent_diffusion = latent_diffusion.to(device)

    latent_diffusion.cond_stage_model.embed_mode = "text"

    batch = make_batch_for_text_to_audio(text, batchsize=batchsize)

    with torch.no_grad():
        waveform = latent_diffusion.generate_sample(
            [batch],
            unconditional_guidance_scale=guidance_scale,
            n_gen=n_gen,
        )
    return waveform
