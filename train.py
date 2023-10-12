import cv2
import torch
import os
import pdb
from basicsr.utils import img2tensor, tensor2img, scandir, get_time_str, get_root_logger, get_env_info
from data.cocoart_dataset import COCOARTDataset
from torch.utils.data import DataLoader
import argparse
from omegaconf import OmegaConf
from ldm.util import instantiate_from_config,default
from ldm.modules.encoders.adapter import Adapter
from PIL import Image
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
import time
import os.path as osp
from basicsr.utils.options import copy_opt_file, dict2str
import logging
from dist_util import init_dist, master_only, get_bare_model, get_dist_info
from loss.VGG_Loss import Style_Loss
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


@master_only
def mkdir_and_rename(path):
    """mkdirs. If path exists, rename it with timestamp and create a new one.

    Args:
        path (str): Folder path.
    """
    if osp.exists(path):
        new_name = path + '_archived_' + get_time_str()
        print(f'Path already exists. Rename it to {new_name}', flush=True)
        os.rename(path, new_name)
    os.makedirs(path, exist_ok=True)
    os.makedirs(osp.join(experiments_root, 'models'))
    os.makedirs(osp.join(experiments_root, 'training_states'))
    os.makedirs(osp.join(experiments_root, 'visualization'))


def load_resume_state(opt):
    resume_state_path = None
    model_resume_state_path=None

    if opt.auto_resume:
        state_path = osp.join('experiments', opt.name, 'training_states')
        if osp.isdir(state_path):
            states = list(scandir(state_path, suffix='state', recursive=False, full_path=False))
            if len(states) != 0:
                states = [float(v.split('.state')[0]) for v in states]

                resume_state_path = osp.join(state_path, f'{max(states):.0f}.state')

        model_state_path = osp.join('experiments', opt.name, 'models')
        if osp.isdir(model_state_path):
            model_states = list(scandir(model_state_path, suffix='pth', recursive=False, full_path=False))
            if len(model_states) != 0:
                model_states = [float((v.split('.pth')[0]).replace('model_ad_','',1)) for v in model_states]

                model_resume_state_path = osp.join(model_state_path, f'model_ad_{max(model_states):.0f}.pth')

    if resume_state_path is None:
        resume_state = None
    else:
        print('resume from', resume_state_path)
        resume_state = torch.load(resume_state_path, map_location='cpu')

    if model_resume_state_path is None:
        model_resume_state = None
    else:
        print('model resume from', model_resume_state_path)
        model_resume_state = torch.load(model_resume_state_path,map_location='cpu')
    return resume_state,model_resume_state



parser = argparse.ArgumentParser()
parser.add_argument(
    "--bsize",
    type=int,
    default=1,
    help="the batch size on each progress"
)
parser.add_argument(
    "--epochs",
    type=int,
    default=10000,
    help="total epochs for training"
)
parser.add_argument(
    "--resume_epoch",
    type=int,
    default=0,
    help="the epoch to resume"
)
parser.add_argument(
    "--num_workers",
    type=int,
    default=2,
    help="the prompt to render"
)

parser.add_argument(
    "--auto_resume",
    action='store_true',
    help="auto resume",
)
parser.add_argument(
    "--config",
    type=str,
    default="configs/train_style.yaml",
    help="path to config which constructs model",
)
parser.add_argument(
    "--print_fq",
    type=int,
    default=100,
    help="print frequency for logs when training",
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
    "--C",
    type=int,
    default=4,
    help="latent channels",
)
parser.add_argument(
    "--f",
    type=int,
    default=8,
    help="downsampling factor",
)
parser.add_argument(
    "--compare_num",
    type=int,
    default=3,
    help="number of compare style for contrastive learning",
)
parser.add_argument(
    "--lambda_1",
    type=float,
    default=60,
    help="ratio for LDM loss",
)
parser.add_argument(
    "--lambda_2",
    type=float,
    default=5,
    help="ratio for contrastive style loss",
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
    help='launcher for distributed training'
)
parser.add_argument(
    "--ckpt",
    type=str,
    default="pretrained_models/sd-v1-4.ckpt",
    help="path to checkpoint of model",
)
parser.add_argument(
    "--train_root",
    type=str,
    default="/data/Datasets/MSCOCO/train2014",
    help="path to training root",
)
parser.add_argument(
    "--mask_root",
    type=str,
    default="/data/SegmentationClass_select",
    help="path to mask root for training",
)
parser.add_argument(
    "--style_root",
    type=str,
    default="/data/wikiart",
    help="path to style root for training",
)
parser.add_argument(
    "--vgg_path",
    type=str,
    default="pretrained_models/vgg_normalised.pth",
    help="path to vgg",
)


opt = parser.parse_args()

if __name__ == '__main__':
    config = OmegaConf.load(f"{opt.config}")
    opt.name = config['name']

    # distributed setting

    torch.distributed.init_process_group(backend='nccl')
    device = 'cuda'
    torch.cuda.set_device(opt.local_rank)

    # dataset
    train_dataset = COCOARTDataset(load_size_H=opt.H,load_size_W=opt.W,is_for_train=True,train_root=opt.train_root,mask_root=opt.mask_root,style_root=opt.style_root,compare_num=opt.compare_num)

    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=opt.bsize,
        shuffle=(train_sampler is None),
        num_workers=opt.num_workers,
        pin_memory=True,
        sampler=train_sampler)

    # stable diffusion + dual encoder fusion
    model = load_model_from_config(config, f"{opt.ckpt}").to(device)


    # adapter
    model_ad = Adapter(cin=int(64 * 4), channels=[320, 640, 1280, 1280][:4], nums_rb=2, ksize=1, sk=True,
                       use_conv=False)

    # resume state
    resume_state, model_resume_state = load_resume_state(opt)

    if model_resume_state is not None:
        # adapter
        model_ad.load_state_dict(model_resume_state['ad'])
        # dual encoder fusion
        model.model.diffusion_model.interact_blocks.load_state_dict(model_resume_state['interact'])

    model_ad=model_ad.to(device)
    model=model.to(device)

    # to gpus
    model_ad = torch.nn.parallel.DistributedDataParallel(
        model_ad,
        device_ids=[opt.local_rank],
        output_device=opt.local_rank)
    model = torch.nn.parallel.DistributedDataParallel(
        model,
        device_ids=[opt.local_rank],
        output_device=opt.local_rank)


    vgg_loss = Style_Loss(compare_num=opt.compare_num,vgg_path=opt.vgg_path)
    vgg_loss=vgg_loss.cuda()
    vgg_loss = torch.nn.parallel.DistributedDataParallel(
        vgg_loss,
        device_ids=[opt.local_rank],
        output_device=opt.local_rank)



    # optimizer
    params = list(model_ad.parameters())

    for param in model.module.model.diffusion_model.interact_blocks.parameters():
            param.requires_grad=True
            params.append(param)

    optimizer = torch.optim.AdamW(params, lr=config['training']['lr'])

    experiments_root = osp.join('experiments', opt.name)



    if resume_state is None:
        mkdir_and_rename(experiments_root)
        start_epoch = 0
        current_iter = 0
        # WARNING: should not use get_root_logger in the above codes, including the called functions
        # Otherwise the logger will not be properly initialized
        log_file = osp.join(experiments_root, f"train_{opt.name}_{get_time_str()}.log")
        logger = get_root_logger(logger_name='basicsr', log_level=logging.INFO, log_file=log_file)
        logger.info(get_env_info())
        logger.info(dict2str(config))
    else:
        # WARNING: should not use get_root_logger in the above codes, including the called functions
        # Otherwise the logger will not be properly initialized
        log_file = osp.join(experiments_root, f"train_{opt.name}_{get_time_str()}.log")
        logger = get_root_logger(logger_name='basicsr', log_level=logging.INFO, log_file=log_file)
        logger.info(get_env_info())
        logger.info(dict2str(config))
        resume_optimizers = resume_state['optimizers']
        optimizer.load_state_dict(resume_optimizers)

        for state in optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.cuda()



        logger.info(f"Resuming training from epoch: {resume_state['epoch']}, " f"iter: {resume_state['iter']}.")
        start_epoch = resume_state['epoch']
        current_iter = resume_state['iter']


    # training
    logger.info(f'Start training from epoch: {start_epoch}, iter: {current_iter}')
    for epoch in range(start_epoch, opt.epochs):
        train_dataloader.sampler.set_epoch(epoch)

        # train
        for _, data in enumerate(train_dataloader):

            current_iter += 1
            with torch.no_grad():
                c = model.module.get_learned_conditioning(data['text'])
                z = model.module.encode_first_stage((data['comp']).cuda(non_blocking=True))
                z = model.module.get_first_stage_encoding(z)

            comp = data['comp'].to(device)
            mask = data['mask'].to(device)
            style = data['style'].to(device)
            style_comparison =data['style_comparison'].to(device)

            features_adapter = torch.cat((comp, mask), dim=1)

            optimizer.zero_grad()
            model.zero_grad()

            features_adapter = model_ad(features_adapter)

            l_pixel, loss_dict, pred_x0 = model(z, c=c, features_adapter=features_adapter,mask=mask)

            pred_x0 = model.module.decode_first_stage_training(pred_x0)

            l_c, l_s = vgg_loss.module.get_all_loss(comp,style, mask, pred_x0)

            l_contra=vgg_loss.module.get_contrastive_loss(style, mask, pred_x0,style_comparison)

            loss_dict.update({'train/style_loss': l_s})
            loss_dict.update({'train/content_loss': l_c})
            loss_dict.update({'train/contra_style_loss': l_contra})

            loss = l_pixel*opt.lambda_1 + l_s +l_contra*opt.lambda_2 + l_c

            loss.backward()

            optimizer.step()

            if (current_iter + 1) % opt.print_fq == 0:
                logger.info(loss_dict)

            # save checkpoint
            rank, _ = get_dist_info()
            if (rank == 0) and ((current_iter + 1) % config['training']['save_freq'] == 0):

                # save adapter
                save_filename = f'model_ad_{current_iter + 1}.pth'
                save_path = os.path.join(experiments_root, 'models', save_filename)
                save_dict = {}
                model_ad_bare = get_bare_model(model_ad)
                state_dict = model_ad_bare.state_dict()
                for key, param in state_dict.items():
                    if key.startswith('module.'):  # remove unnecessary 'module.'
                        key = key[7:]
                    save_dict[key] = param.cpu()

                    
                # save dual encoder fusion
                save_dict_interact = {}
                model_interact_bare = get_bare_model(model.module.model.diffusion_model.interact_blocks)
                state_dict_interact = model_interact_bare.state_dict()
                for key, param in state_dict_interact.items():
                    if key.startswith('module.'):
                        key = key[7:]
                    save_dict_interact[key] = param.cpu()

                two_model_state = {'ad': save_dict,  'interact': save_dict_interact}

                torch.save(two_model_state, save_path)


                # save state
                state = {'epoch': epoch, 'iter': current_iter + 1, 'optimizers': optimizer.state_dict()}
                save_filename = f'{current_iter + 1}.state'
                save_path = os.path.join(experiments_root, 'training_states', save_filename)
                torch.save(state, save_path)
