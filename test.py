import cv2
import torch
import os
import pdb
from basicsr.utils import img2tensor, tensor2img, scandir, get_time_str, get_root_logger, get_env_info
from data.cocoart_dataset import COCOARTDataset
from torch.utils.data import DataLoader
import argparse
from ldm.models.diffusion.scheduling_pndm import PNDMScheduler
from omegaconf import OmegaConf
from ldm.util import instantiate_from_config,default
from ldm.modules.encoders.adapter import Adapter,NoRes_Adapter
from PIL import Image
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
import time
import os.path as osp
from basicsr.utils.options import copy_opt_file, dict2str
import logging
from dist_util import init_dist, master_only, get_bare_model, get_dist_info

from tqdm.auto import tqdm

def load_model_from_config(config, ckpt, verbose=False):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cpu")
    if "global_step" in pl_sd:
        print(f"Global Step: {pl_sd['global_step']}")
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)

    if len(m) > 0 and verbose:
        print("missing keys:")
        print(m)
    if len(u) > 0 and verbose:
        print("unexpected keys:")
        print(u)

    model.eval()
    return model


parser = argparse.ArgumentParser()
parser.add_argument(
    "--bsize",
    type=int,
    default=1,
    help="batch size on each progress"
)
parser.add_argument(
    "--model_resume_path",
    type=str,
    default="pretrained_models/PHDiffusionWithRes.pth",
    help="the path to adapter and fusion module pth"
)
parser.add_argument(
    "--ckpt",
    type=str,
    default="pretrained_models/sd-v1-4.ckpt",
    help="path to checkpoint of model",
)
parser.add_argument(
    "--config",
    type=str,
    default="configs/test.yaml",
    help="path to config which constructs model",
)
parser.add_argument(
    "--W",
    type=int,
    default=512,
    help="image height, in pixel space",
)
parser.add_argument(
    "--H",
    type=int,
    default=512,
    help="image width, in pixel space",
)
parser.add_argument(
    "--num_inference_steps",
    type=int,
    default=50,
    help="number of sampling steps",
)

parser.add_argument(
    "--strength",
    type=float,
    default=0.7,
    help="strength for controlling total denoising steps",
)

parser.add_argument(
    "--gpus",
    default=[0,1,2,3],
    help="gpu idx",
)
parser.add_argument(
    '--local_rank',
    default=0,
    type=int,
    help='node rank for distributed training'
)
parser.add_argument(
    '--launcher',
    default='pytorch',
    type=str,
    help='node rank for distributed training'
)
parser.add_argument(
    "--test_root",
    type=str,
    default="test_examples",
    help="path to test root",
)
parser.add_argument(
    "--no_residual",
    action='store_true',
    help="use adapter without residual",
)


opt = parser.parse_args()

if __name__ == '__main__':
    config = OmegaConf.load(f"{opt.config}")
    opt.name = config['name']

    # distributed setting

    torch.distributed.init_process_group(backend='nccl')
    device = 'cuda'
    torch.cuda.set_device(opt.local_rank)

    test_dataset = COCOARTDataset(load_size_H=opt.H,load_size_W=opt.W, is_for_train=False,test_root=opt.test_root)


    test_dataloader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=1,
        pin_memory=False)

    # stable diffusion + dual encoder fusion
    model = load_model_from_config(config, f"{opt.ckpt}").to(device)


    # adapter
    if opt.no_residual:
        model_ad = NoRes_Adapter(cin=int(64 * 4), channels=[320, 640, 1280, 1280][:4], nums_rb=2, ksize=1, sk=True,
                       use_conv=False)
    else:
        model_ad = Adapter(cin=int(64 * 4), channels=[320, 640, 1280, 1280][:4], nums_rb=2, ksize=1, sk=True,
                       use_conv=False)

    # resume state
    model_resume_state = torch.load(opt.model_resume_path,map_location='cpu')


    # load adapter weight
    model_ad.load_state_dict(model_resume_state['ad'])

    # load fusion weight
    model.model.diffusion_model.interact_blocks.load_state_dict(model_resume_state['interact'])

    model_ad=model_ad.to(device)
    model=model.to(device)


    model_ad = torch.nn.parallel.DistributedDataParallel(
        model_ad,
        device_ids=[opt.local_rank],
        output_device=opt.local_rank)
    model = torch.nn.parallel.DistributedDataParallel(
        model,
        device_ids=[opt.local_rank],
        output_device=opt.local_rank)


    experiments_root = osp.join('experiments', opt.name)


    output_repo=osp.join(experiments_root, 'output')
    if not osp.exists(str(output_repo)):
        os.makedirs(output_repo)


    for data in test_dataloader:
        with torch.no_grad():
         
            scheduler = PNDMScheduler(
                                  beta_end=0.012,
                                  beta_schedule='scaled_linear',
                                  beta_start=0.00085,
                                  num_train_timesteps=1000,
                                  set_alpha_to_one=False,
                                  skip_prk_steps=True,
                                  steps_offset=1,
                               )

            num_inference_steps=opt.num_inference_steps
            scheduler.set_timesteps(num_inference_steps, device=device)

            init_timestep = min(int(num_inference_steps * opt.strength), num_inference_steps)
            t_start = max(num_inference_steps - init_timestep, 0)
            timesteps = scheduler.timesteps[t_start:]
            num_inference_steps=num_inference_steps - t_start


            c = model.module.get_learned_conditioning(data['text'])

            mask = data['mask'].to(device)
            comp = data['comp'].to(device)
            path_name = data['path_name']

            batch_size=mask.shape[0]

            latent_timestep = timesteps[:1].repeat(batch_size)


            x_0 = model.module.encode_first_stage(comp.cuda(non_blocking=True))
            x_0 = model.module.get_first_stage_encoding(x_0)
            mask_latents = torch.nn.functional.interpolate(mask, size=x_0.shape[-2:]).to(device)

            noise = torch.randn(x_0.shape, device=x_0.device, dtype=x_0.dtype)
            latents = scheduler.add_noise(x_0, noise, latent_timestep)

            adapter_input = torch.cat((comp, mask), dim=1).to(dtype=comp.dtype)
            features_adapter = model_ad(adapter_input)

            num_warmup_steps = len(timesteps) - num_inference_steps * scheduler.order

            with tqdm(total=num_inference_steps) as progress_bar:
                for i, t in enumerate(timesteps):

                    noise = torch.randn(x_0.shape, device=x_0.device, dtype=x_0.dtype)
                    t_latents = scheduler.add_noise(x_0, noise, t)
                    latents = latents * mask_latents + t_latents * (1 - mask_latents)

                    latent_model_input = latents
                    latent_model_input = scheduler.scale_model_input(latent_model_input, t)

                    # predict the noise residual

                    noise_pred = model.module.model.diffusion_model(x=latent_model_input,fg_mask=mask_latents, timesteps=t.repeat(batch_size), context=c, features_adapter=features_adapter)

                    # compute the previous noisy sample x_t -> x_t-1

                    latents = scheduler.step(noise_pred, t, latents).prev_sample

                    # call the callback, if provided
                    if i == len(timesteps) - 1 or (
                            (i + 1) > num_warmup_steps and (i + 1) % scheduler.order == 0):
                        progress_bar.update()


            x_samples_ddim = model.module.decode_first_stage(latents)

            #x_samples_ddim = x_samples_ddim * mask+(1-mask)*style

            x_samples_ddim = torch.clamp((x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0)
            x_samples_ddim = x_samples_ddim.cpu().permute(0, 2, 3, 1).numpy()
            for id_sample, x_sample in enumerate(x_samples_ddim):
                x_sample = 255. * x_sample
                img = x_sample.astype(np.uint8)

                output_path=path_name[0]+'.png'
                cv2.imwrite(os.path.join(output_repo,output_path), img[:, :, ::-1])



