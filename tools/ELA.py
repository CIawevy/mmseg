import os

from PIL import Image
from data_process import  data_assistant
import os.path as osp
from PIL import ImageChops,ImageEnhance
from tqdm import tqdm
import numpy as np
file_path = "/data/ipad/Forgery/pristine/CocoGlide/TP/image/"
dst_path0 = "/data/ipad/Forgery/pristine/CocoGlide/TP/JPEG/"
dst_path1 = "/data/ipad/Forgery/pristine/CocoGlide/TP/ELA/"
for dir in [dst_path0,dst_path1]:
    if not osp.exists(dir):
        os.makedirs(dir)
da = data_assistant()
def ELA_analysis(file_path,dst_path0,dst_path1,QF,enhance):
    file_list = None
    if file_path[-4] == '.': # is file
        file_list = [file_path]
    else: #is dir
        file_list= da.get_data(file_path)
    for path in tqdm(file_list):
        im = Image.open(path).convert('RGB')
        recompress_save = osp.join(dst_path0,osp.basename(path))
        ela_save = osp.join(dst_path1,osp.basename(path))
        im.save(recompress_save,'JPEG',quality=QF)#?
        resaved_im = Image.open(recompress_save)
        ela_im = ImageChops.difference(im,resaved_im)
        extrema = ela_im.getextrema()
        max_diff = max([ex[1] for ex in extrema])
        if max_diff == 0:
            max_diff = 1
        scale = 255.0/max_diff
        if enhance is True:
            ela_im = ImageEnhance.Brightness(ela_im).enhance(scale*4) #H W 3
            ela_im.save(ela_save,lossless=True)






if __name__ == "__main__":
    ELA_analysis(file_path,dst_path0,dst_path1,90,True
                 )
