import os
import sys
import torch
import torch.nn as nn
import numpy as np
import math
import cv2

os.environ['CUDA_VISIBLE_DEVICES']='1'

import kornia as K

sys.path.append(os.path.join(os.path.dirname(__file__),'..'))

from models.model import LiftFeatSPModel
from models.interpolator import InterpolateSparse2d
from utils.config import featureboost_config


class NonMaxSuppression(torch.nn.Module):
    def __init__(self, rep_thr=0.1, top_k=4096):
        super(NonMaxSuppression,self).__init__()
        self.max_filter = torch.nn.MaxPool2d(kernel_size=5, stride=1, padding=2)
        self.rep_thr = rep_thr
        self.top_k=top_k
        
        
    def NMS(self, x, threshold = 0.05, kernel_size = 5):
        B, _, H, W = x.shape
        pad=kernel_size//2
        local_max = nn.MaxPool2d(kernel_size=kernel_size, stride=1, padding=pad)(x)
        pos = (x == local_max) & (x > threshold)
        pos_batched = [k.nonzero()[..., 1:].flip(-1) for k in pos]

        pad_val = max([len(x) for x in pos_batched])
        pos = torch.zeros((B, pad_val, 2), dtype=torch.long, device=x.device)

        #Pad kpts and build (B, N, 2) tensor
        for b in range(len(pos_batched)):
            pos[b, :len(pos_batched[b]), :] = pos_batched[b]

        return pos
        
    def forward(self, score):
        pos = self.NMS(score,self.rep_thr)
        
        return pos

def load_model(model, weight_path):
    pretrained_weights = torch.load(weight_path)

    model_keys = set(model.state_dict().keys())
    pretrained_keys = set(pretrained_weights.keys())

    missing_keys = model_keys - pretrained_keys
    unexpected_keys = pretrained_keys - model_keys

    if missing_keys:
        print("Missing keys in pretrained weights:", missing_keys)
    else:
        print("No missing keys in pretrained weights.")

    if unexpected_keys:
        print("Unexpected keys in pretrained weights:", unexpected_keys)
    else:
        print("No unexpected keys in pretrained weights.")

    if not missing_keys and not unexpected_keys:
        model.load_state_dict(pretrained_weights)
        print("Pretrained weights loaded successfully.")
    else:
        model.load_state_dict(pretrained_weights, strict=False)
        print("There were issues with the keys.")
    return model


def load_torch_image(fname, device=torch.device('cpu')):
    img = K.image_to_tensor(cv2.imread(fname), False).float() / 255.
    img = K.color.bgr_to_rgb(img.to(device))
    
    image=cv2.imread(fname)
    H,W,C=image.shape[0],image.shape[1],image.shape[2]

    _H=math.ceil(H/32)*32
    _W=math.ceil(W/32)*32

    pad_h=_H-H
    pad_w=_W-W

    image=cv2.copyMakeBorder(image,0,pad_h,0,pad_w,cv2.BORDER_CONSTANT,None,(0, 0, 0))
    
    pad_info=[0,pad_h,0,pad_w]
    
    image = K.image_to_tensor(image, False).float() / 255.
    image = image.to(device)
        
    return image,pad_info


class LiftFeat(nn.Module):
    def __init__(self,weight,top_k=4096,detect_threshold=0.1):
        super().__init__()
        self.net=LiftFeatSPModel(featureboost_config)
        self.top_k=top_k
        self.sampler=InterpolateSparse2d('bicubic')
        self.net=load_model(self.net,weight)
        self.detector=NonMaxSuppression(rep_thr=detect_threshold)
    
    @torch.inference_mode()
    def extract(self,image,pad_info):
        B,_,_H1,_W1=image.shape
        M1,K1,D1=self.net.forward1(image)
        refine_M=self.net.forward2(M1,K1,D1)
        
        refine_M=refine_M.reshape(M1.shape[0],M1.shape[2],M1.shape[3],-1).permute(0,3,1,2)
        refine_M=torch.nn.functional.normalize(refine_M,2,dim=1)
        
        descs_map=refine_M
        # descs_map=M1
        
        scores=torch.softmax(K1,dim=1)[:,:64]
        heatmap=scores.permute(0,2,3,1).reshape(scores.shape[0],scores.shape[2],scores.shape[3],8,8)
        heatmap=heatmap.permute(0,1,3,2,4).reshape(scores.shape[0],1,scores.shape[2]*8,scores.shape[3]*8)
        
        pos=self.detector(heatmap)
        kpts=pos.squeeze(0)
        mask_w=kpts[...,0]<(_W1-pad_info[-1])
        kpts=kpts[mask_w]
        mask_h=kpts[..., 1]<(_H1-pad_info[1])
        kpts=kpts[mask_h]
        
        descs=self.sampler(descs_map,kpts.unsqueeze(0),_H1,_W1)
        descs=torch.nn.functional.normalize(descs,p=2,dim=1)
        descs=descs.squeeze(0)
        
        return {
            'descriptors':descs,
            'keypoints':kpts
        }
        
    def match_liftfeat(self, img1, pad_info1, img2, pad_info2, min_cossim=-1):
        # import pdb;pdb.set_trace()
        data1=self.extract(img1, pad_info1)
        data2=self.extract(img2, pad_info2)
        
        kpts1,feats1=data1['keypoints'],data1['descriptors']
        kpts2,feats2=data2['keypoints'],data2['descriptors']
        
        cossim = feats1 @ feats2.t()
        cossim_t = feats2 @ feats1.t()

        _, match12 = cossim.max(dim=1)
        _, match21 = cossim_t.max(dim=1)

        idx0 = torch.arange(len(match12), device=match12.device)
        mutual = match21[match12] == idx0

        if min_cossim > 0:
            cossim, _ = cossim.max(dim=1)
            good = cossim > min_cossim
            idx0 = idx0[mutual & good]
            idx1 = match12[mutual & good]
        else:
            idx0 = idx0[mutual]
            idx1 = match12[mutual]
            
        mkpts1,mkpts2=kpts1[idx0],kpts2[idx1]

        return mkpts1, mkpts2
    
weight=os.path.join(os.path.dirname(__file__),'../weights/LiftFeat.pth')

liftfeat=LiftFeat(weight)

save_file=os.path.join(os.path.dirname(__file__),'../weights/LiftFeat.pt')

liftfeat_script=torch.jit.script(liftfeat)
liftfeat_script.save(save_file)

# checkpoint = {
#     'model_name': 'LiftFeat',
#     'model_args': {
#         'top_k': 4096,
#         'detect_threshold': 0.1
#     },
#     'state_dict': liftfeat.state_dict()
# }

# torch.save(checkpoint,os.path.join(os.path.dirname(__file__),'../weights/LiftFeat.ckpt'))

