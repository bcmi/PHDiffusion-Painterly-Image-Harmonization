import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import pdb
import numpy
import random
from scipy.stats import entropy


vgg = nn.Sequential(
    nn.Conv2d(3, 3, (1, 1)),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(3, 64, (3, 3)),
    nn.ReLU(),  # relu1-1
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(64, 64, (3, 3)),
    nn.ReLU(),  # relu1-2
    nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(64, 128, (3, 3)),
    nn.ReLU(),  # relu2-1
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(128, 128, (3, 3)),
    nn.ReLU(),  # relu2-2
    nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(128, 256, (3, 3)),
    nn.ReLU(),  # relu3-1
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)),
    nn.ReLU(),  # relu3-2
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)),
    nn.ReLU(),  # relu3-3
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)),
    nn.ReLU(),  # relu3-4
    nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 512, (3, 3)),
    nn.ReLU(),  # relu4-1, this is the last layer used
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu4-2
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu4-3
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu4-4
    nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu5-1
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu5-2
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu5-3
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU()  # relu5-4
)
projection_style = nn.Sequential(
    nn.Linear(in_features=256, out_features=128),
    nn.ReLU(),
    nn.Linear(in_features=128, out_features=128)
)

def get_foreground_mean_std(features, mask, eps=1e-5):
    region = features * mask.detach()
    sum = torch.sum(region, dim=[2, 3])  # (B, C)
    num = torch.sum(mask, dim=[2, 3])  # (B, C)
    mu = sum / (num + eps)
    mean = mu[:, :, None, None]
    var = torch.sum((region + (1 - mask) * mean - mean) ** 2, dim=[2, 3]) / (num + eps)
    var = var[:, :, None, None]
    std = torch.sqrt(var + eps)
    mu_sigma = torch.cat((mean, std), dim=1)
    return mean, std, mu_sigma


def calc_mean_std(feat, eps=1e-5):
    # eps is a small value added to the variance to avoid divide-by-zero.
    size = feat.size()
    assert (len(size) == 4)
    N, C = size[:2]
    feat_var = feat.contiguous().view(N, C, -1).var(dim=2) + eps
    feat_std = feat_var.sqrt().view(N, C, 1, 1)
    feat_mean = feat.contiguous().view(N, C, -1).mean(dim=2).view(N, C, 1, 1)
    return feat_mean, feat_std


class Style_Loss(nn.Module):
    def __init__(self, style_loss='fg',compare_num=0,vgg_path=''):
        super(Style_Loss, self).__init__()

        self.compare_num=compare_num
        self.vgg_path=vgg_path

        vggnet = vgg
        vggnet.load_state_dict(torch.load(self.vgg_path))
        vggnet = nn.Sequential(*list(vggnet.children())[:31])
        vgg_layers = list(vggnet.children())

        self.enc_1 = nn.Sequential(*vgg_layers[:4])  # input -> relu1_1
        self.enc_2 = nn.Sequential(*vgg_layers[4:11])  # relu1_1 -> relu2_1
        self.enc_3 = nn.Sequential(*vgg_layers[11:18])  # relu2_1 -> relu3_1
        self.enc_4 = nn.Sequential(*vgg_layers[18:31])  # relu3_1 -> relu4_1
        self.mse_loss = nn.MSELoss()
        self.mse_loss_sum = nn.MSELoss(size_average=True)

        self.style_loss = style_loss

        self.cross_entropy_loss = nn.CrossEntropyLoss()
        # projection
        self.proj_style = projection_style

        # fix the encoder
        for name in ['enc_1', 'enc_2', 'enc_3', 'enc_4']:
            for param in getattr(self, name).parameters():
                param.requires_grad = False


    # extract relu1_1, relu2_1, relu3_1, relu4_1 from input image
    def encode_with_intermediate(self, input):
        results = [input]
        for i in range(4):
            func = getattr(self, 'enc_{:d}'.format(i + 1))
            results.append(func(results[-1]))
        return results[1:]

    def encode_with_small_intermediate(self, input):

        for i in range(3):
            func = getattr(self, 'enc_{:d}'.format(i + 1))
            input=func(input)
        return input

    def encode_with_intermediate_res(self, input, mask, res_feats):
        results = [input]
        for i in range(4):
            func = getattr(self, 'enc_{:d}'.format(i + 1))
            if i == 2 or i == 3:
                mask_ds = F.interpolate(mask, size=res_feats[i - 2].size()[2:])
                res_feat = results[-1] + res_feats[i - 2] * mask_ds
                results.append(func(res_feat))
            else:
                results.append(func(results[-1]))
        return results[1:]


    def downsample(self, image_tensor, width, height):
        image_upsample_tensor = torch.nn.functional.interpolate(image_tensor, size=[width, height])
        image_upsample_tensor = image_upsample_tensor.clamp(0, 1)
        return image_upsample_tensor

    def calc_content_loss(self, gen, comb):
        loss = self.mse_loss(gen, comb)
        return loss

    def calc_style_loss_mulitple_fg(self, combs, styles, mask):
        loss = 0.0
        for i in [0,1,2,3]:  # 4 layers
            width = height = combs[i].size(-1)
            downsample_mask = self.downsample(mask, width, height)
            downsample_mask_style = torch.ones(downsample_mask.size()).to(combs[0].device)
            mu_cs, sigma_cs, _ = get_foreground_mean_std(combs[i], downsample_mask)
            mu_target, sigma_target, _ = get_foreground_mean_std(styles[i], downsample_mask_style)
            loss_i = self.mse_loss(mu_cs, mu_target) + self.mse_loss(sigma_cs, sigma_target)

            loss += loss_i


        return loss

    def calc_style_loss_mulitple(self, combs, styles):

        loss = 0.0
        for i in range(0, 4):
            width = height = combs[i].size(-1)
            input_mean, input_std = calc_mean_std(combs[i])
            target_mean, target_std = calc_mean_std(styles[i])
            loss_i = self.mse_loss(input_mean, target_mean) + self.mse_loss(input_std, target_std)
            loss += loss_i
        return loss

    def compute_contrastive_loss(self, feat_q, feat_k, tau, index):

        out = torch.mm(feat_q, feat_k.transpose(1, 0)) / tau

        loss = self.cross_entropy_loss(out, torch.tensor([index], dtype=torch.long, device=feat_q.device))

        return loss

    def style_feature_contrastive(self, input):

        out = torch.sum(input, dim=[2, 3])
        out = self.proj_style(out)
        out = out / torch.norm(out, p=2, dim=1, keepdim=True)
        return out


    def mask_all(self,mask):
        x_left = []
        x_right = []
        y_bottom = []
        y_top = []

        for i in range(mask.shape[0]):
            x1,x2,y1,y2=mask_bboxregion_coordinate(mask[i][0])
            x_left.append(x1)
            x_right.append(x2)
            y_bottom.append(y1)
            y_top.append(y2)

        return x_left, x_right, y_bottom, y_top


    def mask_bboxregion_coordinate(self,mask):

        valid_index = torch.argwhere(mask == 1)  # [length,2]
        if valid_index.shape[0] < 1:
            x_left = 0
            x_right = 0
            y_bottom = 0
            y_top = 0
        else:
            x_left = torch.min(valid_index[:, 0])
            x_right = torch.max(valid_index[:, 0])
            y_bottom = torch.min(valid_index[:, 1])
            y_top = torch.max(valid_index[:, 1])

        return x_left, x_right, y_bottom, y_top


    def calc_contrastive_loss(self,style,comparison,pred,mask):

        b,c,h,w=pred.shape

        style_contrastive_loss = 0

        downsample_mask = self.downsample(mask, h, w)

        x_left, x_right, y_bottom, y_top=self.mask_bboxregion_coordinate(downsample_mask[0][0])

        width=x_right-x_left+1
        height=y_top-y_bottom+1

        total_crop=torch.zeros([comparison.shape[0]+1, c, height, width]).to(pred.device)

        reference_crop=pred[0,:,y_bottom:y_top+1,x_left:x_right+1].unsqueeze(0)


        random_left=random.randint(0,max(0, w-width-2))
        random_bottom = random.randint(0, max(0,h - height - 2))

        total_crop[0]=style[0][:,random_bottom:random_bottom+height,random_left:random_left+width]

        for i in range(comparison.shape[0]):
            total_crop[i+1]=comparison[i][:,random_bottom:random_bottom+height,random_left:random_left+width]


        reference_crop = self.style_feature_contrastive(reference_crop)
        total_crop=self.style_feature_contrastive(total_crop)


        style_contrastive_loss += self.compute_contrastive_loss(reference_crop, total_crop, 0.2, 0)
        return style_contrastive_loss


    def get_contrastive_loss(self, style, mask, pred,style_comparison):

        style_feats = self.encode_with_small_intermediate(style)

        fine_feats = self.encode_with_small_intermediate(pred)

        # only work when batch size on each progress is 1
        # this part of code will be improved to support all batch sizes on our final release
        if(style.shape[0]==1):

            style_comparison=style_comparison.squeeze(0)
            style_comparison_feats=[]
            for i in range(self.compare_num):
                style_comparison_feats.append(self.encode_with_small_intermediate(style_comparison[i:i+1]))
            style_comparison_feats=torch.cat(style_comparison_feats,dim=0)

            loss_contra=self.calc_contrastive_loss(style_feats,style_comparison_feats,fine_feats,mask)
        else:
            loss_contra = 0.0

        return loss_contra



    def get_style_loss(self,style, mask, pred):


        style_feats = self.encode_with_intermediate(style)

        fine_feats = self.encode_with_intermediate(pred)

        loss_s = self.calc_style_loss_mulitple_fg(fine_feats, style_feats, mask)

        return loss_s

    def get_all_loss(self, comp, style, mask, pred):

        style_feats = self.encode_with_intermediate(style)

        comb_feats = self.encode_with_intermediate(comp)

        fine_feats = self.encode_with_intermediate(pred)

        loss_c = self.calc_content_loss(fine_feats[-1], comb_feats[-1])

        loss_s = self.calc_style_loss_mulitple_fg(fine_feats, style_feats, mask)

        return loss_c, loss_s