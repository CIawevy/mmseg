# Thank xiexinch: https://github.com/open-mmlab/mmsegmentation/issues/2434#issuecomment-1441392574
import os

import torch
import cv2
import numpy as np
from mmseg.visualization import SegLocalVisualizer
from mmseg.apis import init_model
from mmseg.utils import register_all_modules
from mmengine.model import revert_sync_batchnorm
from tools.data_process import data_assistant
from os import path  as osp

# prefix = "mmsegmentation-1.0.0rc5/"
# prefix = ""
# config = prefix + r"log\7_ttpla_p2t_t_20k\ttpla_p2t_t_20k.py"
# checkpoint = prefix + r"log\7_ttpla_p2t_t_20k\iter_8000.pth"
da = data_assistant()
# config = "/data/hszhu/code/mmseg_project/configs/sam/Tamp_Sam_base.py"
# checkpoint = "/data/hszhu/code/mmseg_project/pretrained/vit-base-p16_SAM-pre_3rdparty_sa1b-1024px_20230411-2320f9cc.pth"
save_prefix="/data/Forgery/test/_TEST_CASIAv1+/Tp/Heat_map/"
img_iter=iter(sorted(da.get_data(srcpath="/data/Forgery/test/_TEST_CASIAv1+/Tp/image/",depth=1)))
count=0
tmp=""
# layer_num=12 #SAM base
# layer_num=24 # SAM large
layer_num=32 #SAM Huge


def draw_heatmap(featmap):
    featmap=featmap[0]
    # path="/data/Forgery/test/Columbia/TP/Heat_map/"
    # featmap=featmap.permute(1,2,0).cpu().detach().numpy()
    # cv2.imwrite(osp.join(path,'original.jpg'),featmap)
    # a=1/0
    global count,tmp,img_iter,layer_num
    vis = SegLocalVisualizer()
    count+= 1 if count <layer_num+1 else -layer_num
    if count == 1:
        img_path = next(img_iter)
        tmp=img_path
    else:
        img_path=tmp
    savedir = osp.join(save_prefix, da.get_filename(img_path))
    if not osp.exists(savedir):
        os.mkdir(savedir)
    ori_img = cv2.imread(img_path)
    out = vis.draw_featmap(featmap, ori_img,overlaid=False)
    n=1
    newname=osp.join(savedir,'feature'+str(n)+'.'+da.get_suffix(img_path))
    while osp.exists(newname):
        n+=1
        newname=osp.join(savedir,'feature'+str(n)+'.'+da.get_suffix(img_path))
    # print(newname)
    cv2.imwrite(newname,out)
    # cv2.imshow('cam', out)
    # cv2.waitKey(0)

def generate_featmap(config, checkpoint, img_path):
    register_all_modules()

    model = init_model(config, checkpoint, device='cpu')
    model = revert_sync_batchnorm(model)
    vis = SegLocalVisualizer()

    ori_img = cv2.imread(img_path)
    img = torch.from_numpy(ori_img.astype(np.single)).permute(2, 0, 1).unsqueeze(0)

    logits = model(img)
    out = vis.draw_featmap(logits[0], ori_img)

    cv2.imshow('cam', out)
    cv2.waitKey(0)

if __name__ == "__main__":
    generate_featmap(config, checkpoint, img_path)