import os
from os import path as osp
from tqdm import tqdm
from glob import glob
import numpy as np
import shutil
from collections import defaultdict
from PIL import Image
from typing import Callable, Any, Union, Optional
import cv2
from skimage import io
from PIL import ImageFile



class data_assistant:
    def __init__(self,
                 srcpath: str = None,
                 dstpath: str = None,
                 metainfo: dict = None,
                 datalist: list = None,
                 comparision: list[tuple] = None,
                 ) -> None:
        self.srcpath = srcpath
        self.dstpath = dstpath

    @classmethod
    def get_data(cls, srcpath: str = None, depth: int = 1) -> list:
        """

        Args:
            srcpath: 源文件夹路径
            depth:文件夹深度 1/2

        Returns:
            datalist
        """
        assert srcpath is not None, "请输入路径参数"
        if depth == 2:
            return glob(osp.join(srcpath, "*/*"))
        else:
            return glob(osp.join(srcpath, "*"))

    @classmethod
    def get_info(cls, srcfile: str = None, depth: int = 1, datalist: list = None, channel: bool = True,
                 suffix: bool = True, channel_list: bool = False, suffix_list: bool = False) -> dict:
        """

        Args:
            srcfile:str 输入数据路径
            depth:int 输入深度
            datalist:list 或者直接输入datalist
            channel:bool 是否需要统计通道
            suffix:bool 是否需要统计后缀
            channel_list:bool 是否需要通道的文件信息
            suffix_list:bool 是否需要后缀的文件信息

        Returns:
            data_info 包含数目 通道 后缀 信息的字典
        """

        if srcfile is not None:
            datalist = cls.get_data(srcfile, depth)
        assert datalist is not None, "未输入数据"
        data_info = defaultdict()
        number = len(datalist)
        suffix_dict = defaultdict()
        channal_dict = defaultdict()

        for data in tqdm(datalist):
            _, filename = osp.split(data)
            if suffix:
                sfx = cls.get_suffix(filename)
                if sfx not in suffix_dict.keys():
                    suffix_dict[sfx] = [data] if suffix_list else 1
                elif isinstance(suffix_dict[sfx], int):
                    suffix_dict[sfx] += 1
                else:
                    suffix_dict[sfx].append(data)
            if channel:
                im = Image.open(data)
                c_num = str(len(im.split()))
                if c_num not in channal_dict.keys():
                    channal_dict[c_num] = [data] if channel_list else 1
                elif isinstance(channal_dict[c_num], int):
                    channal_dict[c_num] += 1
                else:
                    channal_dict[c_num].append(data)
        data_info['capacity'] = number
        data_info['suffix'] = suffix_dict if suffix else None
        data_info['channel'] = channal_dict if channel else None
        return data_info

    @classmethod
    def get_suffix(cls, filename: str = None) -> str:
        assert filename is not None, "请输入文件名"
        return osp.splitext(filename)[1][1:]
        # for i,x in enumerate(filename[::-1]):
        #     if x=='.':
        #         return filename[-i:]
        #   jpg

    @classmethod
    def get_filename(cls, filepath: str = None) -> str:
        assert filepath is not None, "请输入文件名"
        return osp.splitext(osp.basename(filepath))[0]

    @classmethod
    def converter(cls, srcfile: str = None, depth: int = 1, datalist: list = None, dstpath: str = None, to_jpg=False,
                  to_png=False, is_groundtruth: bool = False,
                  quick_gt_name:list=None,process_rules: Callable = None) -> None:
        """
        数据转换模块，流程：输入datalist，按照list查找数据，自定义数据转换，保存
        示例：
        Args:
            datalist: 输入
            dstpath: 保存路径
            to_jpg: 转换模式.xxx to .jpg
            to_png: 转换模式.xxx to .png
            channel_compress: 开启to_png时,是否将mask压缩到8位深度的单通道GT
            mask_process_rules: 自定义转换函数 输入ndarray 输出ndarray 因为cv2的read自动转换为ndarray
            quick_gt_name:list=None: 传入image list 自动通过image的编号+gt

        """
        if srcfile is not None:
            datalist = cls.get_data(srcfile, depth)
        assert datalist is not None, "未输入数据"
        assert to_png or to_jpg, "未选择模式"
        if dstpath is None:
            key = input("警告：保存路径未输入，即将在原始文件夹进行数据覆盖，确认请输入:yes")
            if key == 'yes':
                dstpath = srcfile
        assert dstpath is not None, "未输入保存路径"
        if to_jpg:
            print("———————————————————jpg转换开始——————————————————")
            for srcfile in tqdm(datalist):
                im_name = da.get_filename(srcfile)
                image = cv2.imread(srcfile)
                if process_rules:
                    image = process_rules(np.array(image))
                if not osp.exists(dstpath):
                    os.mkdir(dstpath)
                cv2.imwrite(osp.join(dstpath, im_name + '.jpg'), image)
        if to_png:
            print("———————————————————png转换开始——————————————————")
            if is_groundtruth:
                print("***********检测到groundtruth将加载自定义mask转换以及通道压缩***********")
            for index,srcfile in tqdm(enumerate(datalist),total=len(datalist)):
                im_name = da.get_filename(srcfile)
                if quick_gt_name is not None:
                    im_name=da.get_filename(quick_gt_name[index])
                if is_groundtruth:
                    im = cv2.imread(srcfile, flags=0)  # Gray
                else:
                    im = cv2.imread(srcfile)  # original bgr

                if process_rules is not None:
                    im = process_rules(np.array(im))

                if not osp.exists(dstpath):
                    os.mkdir(dstpath)
                cv2.imwrite(osp.join(dstpath, im_name + '.png'), im)

    @classmethod
    def rename(cls, srcfile: str = None, depth: int = 1, datalist: list = None, dstpath: str = None,
               rename_rules: Callable = None, repeate_name_dst: str = None) -> None:
        if srcfile is not None:
            datalist = cls.get_data(srcfile, depth)
        assert datalist is not None, "未输入数据"
        assert rename_rules is not None, "未输入改名规则"
        if dstpath is None:
            key = input("警告：保存路径未输入，即将在原始文件夹进行数据覆盖，确认请输入:yes")
            if key != 'yes':
                assert dstpath is not None, "未输入保存路径"
            print("————————————目标路径未输入，开始本地修改——————————————")
        else:
            print(f"————————————文件改名后存储到{dstpath}——————————————")
        if repeate_name_dst is not None:
            print(f'重复文件输出已启用 自定义保存路径：{repeate_name_dst}')
        count = 0
        for data in tqdm(datalist):
            filename, suffix = osp.splitext(osp.basename(data))
            if rename_rules:
                newname = rename_rules(filename)
            if dstpath is not None:
                if not osp.exists(dstpath):
                    os.mkdir(dstpath)
                name = osp.join(dstpath, newname + suffix)
                if osp.exists(name):
                    print(f'{name} already exists')
                    count += 1
                    if not osp.exists(repeate_name_dst):
                        os.mkdir(repeate_name_dst)
                    shutil.copy(data, osp.join(repeate_name_dst, filename + suffix))
                else:
                    shutil.copy(data, osp.join(dstpath, newname + suffix))
            else:
                name = osp.join(osp.dirname(data), newname + suffix)
                if osp.exists(name):
                    print(f'{name} already exists')
                    count += 1
                    if not osp.exists(repeate_name_dst):
                        os.mkdir(repeate_name_dst)
                    shutil.copy(data, osp.join(repeate_name_dst, filename + suffix))
                else:
                    os.rename(data, name)
        if count != 0:
            print(f'按照此命名规则，有{count} 个重复文件名，已储存至{repeate_name_dst}')
            print("注意repeate的一个很大的可能是rename_rules没写好导致了重复,比如:索引错位，应仔细检查")

    @classmethod
    def del_repeate_data(cls, srcfile: str = None, depth: int = 1, datalist: list = None) -> None:
        pass
        # 当初实验怎么搞出重名文件的不记得了，linux 感觉同名会直接覆盖 所有多了一倍肯定不是同名的
        # 具体实现os.remove(名字)应该只删除一张的 可以用字典计数来控制
        # 感觉这个函数没用 不设置了
        if srcfile is not None:
            datalist = cls.get_data(srcfile, depth)
        assert datalist is not None, "未输入数据"
        checklist = defaultdict()
        for data in datalist:
            filename = cls.get_filename(data)
            checklist[filename] += 1
            # if checklist[filename]!=1:

    @classmethod
    def remove_data(cls, datalist: list = None) -> None:
        print("-------------正在删除所给列表中的文件-----------------------")
        for address in datalist:
            os.remove(address)

    @classmethod
    def ready_for_train(cls, image_path: str = None, img_depth: int = 1, annotation_path: str = None,
                        ano_depth: int = 1,
                        image_list: list = None, annotation_list: list = None,
                        img_suffix: str = 'jpg', ano_suffix: str = 'png',
                        return_correct_list: bool = False,
                        return_error_list: bool = False):
        """

        Args:
            image_path: (可选)图像路径
            img_depth:  （可选
            annotation_path:
            ano_depth:
            image_list:
            annotation_list:
            img_suffix:
            ano_suffix:
            return_correct_list:
            return_error_list:

        Returns:

        """
        point = True
        if image_path is not None:
            image_list = cls.get_data(image_path, img_depth)
        if annotation_path is not None:
            annotation_list = cls.get_data(annotation_path, ano_depth)
        assert image_list and annotation_list, "图像或者mask有一个少传了"
        img_info, ano_info = da.get_info(datalist=image_list, channel=False), da.get_info(datalist=annotation_list,
                                                                                          channel=False)
        num1, num2, suffix1, suffix2 = img_info["capacity"], ano_info["capacity"], list(img_info['suffix']), list(
            ano_info['suffix'])
        if num1 != num2:
            point = False
            print(f'NUM_ERROR :image_list:{num1} , while annotation_list:{num2}')
        else:
            print(f'matched training number:{num1}')
        if (len(suffix1) > 1) or suffix1[0] != img_suffix:
            point = False
            print(f'FORMAT_ERROR image_suffix is {suffix1} while the standard suffix is {img_suffix}')
        else:
            print(f'image_suffix : {img_suffix}')
        if (len(suffix2) > 1) or suffix2[0] != ano_suffix:
            point = False
            print(f'FORMAT_ERROR mask_suffix is {suffix2} while the standard suffix is {ano_suffix}')
        else:
            print(f'annotation_suffix :{ano_suffix}')
        key = input("是否继续？0：基于image_list查找mask，1：基于annotation_list查找image ，其他：退出检查")
        try:
            key not in ['1', '0']
        except KeyboardInterrupt:
            print('finish')
        key2 = input(f'是否需要更改匹配规则，目前{img_suffix} -> {ano_suffix} 0表示不更改 1表示更改')
        try:
            key not in ['1', '0']
        except TypeError:
            print('输入错误，必须是0和1')
        if key2 == '1':
            img_suffix = input('请输入图像后缀 注意包括"." ：')
            ano_suffix = input('请输入标注后缀 注意包括"." ：')

        print('---------------further checking--------------------')
        if key == '0':
            maskname = [osp.basename(x) for x in annotation_list]
            mask_list = [] if return_correct_list else None
            error_mask_list = [] if return_error_list else None
            for image in tqdm(image_list):
                prob_name = osp.basename(image).replace(img_suffix, ano_suffix)
                sta = True
                for id, mask in enumerate(maskname):
                    if prob_name == mask:
                        msk = annotation_list[id]
                        img_shape = cv2.imread(image).shape[:2]
                        mask_shape = cv2.imread(msk).shape[:2]
                        if img_shape != mask_shape:
                            point = False
                            print(
                                f'image from {image} with shape {img_shape} \n mask from {msk} with shape {mask_shape}')
                            if return_error_list:
                                error_mask_list.append(msk)
                        elif return_correct_list:
                            mask_list.append(msk)
                        sta = False
                if sta:
                    point = False
                    print(f'image from {image} cannot mathch any mask')
            if point:
                print('dataset is ready for training ')
                return
            return mask_list, error_mask_list
        else:
            imagename = [osp.basename(x) for x in image_list]
            graph_list = [] if return_correct_list else None
            error_graph_list = [] if return_error_list else None
            for mask in tqdm(annotation_list):
                prob_name = osp.basename(mask).replace(ano_suffix, img_suffix)
                sta = True
                for id, im in enumerate(imagename):
                    if prob_name == im:
                        img = image_list[id]
                        img_shape = cv2.imread(img).shape[:2]
                        mask_shape = cv2.imread(mask).shape[:2]
                        if img_shape != mask_shape:
                            point = False
                            print(
                                f'image from {img} with shape {img_shape} \n mask from {mask} with shape {mask_shape}')
                            if return_error_list:
                                error_graph_list.append(img)
                        elif return_correct_list:
                            graph_list.append(img)
                        sta = False
                if sta:
                    point = False
                    print(f'mask from {mask} cannot mathch any image')
            if point:
                print('dataset is ready for training ')
                return
            return graph_list, error_graph_list

    @classmethod
    def iccp(cls, pngfile: str = None, depth: int = 1, datalist: list = None) -> None:
        if pngfile is not None:
            datalist = cls.get_data(srcpath=pngfile, depth=depth)
        assert datalist is not None, "数据未输入"
        for mask in datalist:
            image = io.imread(mask)
            image = cv2.cvtColor(image, cv2.COLOR_RGBA2BGRA)
            cv2.imencode('.png', image)[1].tofile(mask)
        print('finish')


da = data_assistant()


def nist_process(msk: np.ndarray = None) -> np.ndarray:
    msk[msk != 255] = 200
    msk[msk == 255] = 0
    msk[msk == 200] = 255
    return msk


def Columbia_process(image: np.ndarray = None) -> np.ndarray:
    msk = image[:, :, 1]
    msk[msk <= 120] = 0
    msk[msk > 0] = 255
    return msk

def authentic_mask_generate_rules(image:np.ndarray = None) -> np.ndarray:
    msk = image[:,:]
    msk[msk!=0]=0
    return msk


# 以下是写代码过程中使用函数时顺便留下的8个Demo 帮助使用者理解。
"""

#Demo1 check the suffix tif jpg and channels in caisa dataset and can get special suffix text_list
da=data_assistant()
src='/data/Forgery/train/CASIA 2.0/TP/original_image/'
datalist=da.get_data(src,depth=1)
print(datalist)
info=da.get_info(datalist=datalist,channel=True,suffix=True,channel_list=False,suffix_list=True)
print(info['suffix']['tif'])

#Demo2 check the 3files gt_image of nist
da=data_assistant()
src="/data/Forgery/test/NIST16/3_types_image/"
info=da.get_info(srcfile=src,depth=2,channel=True,suffix=True,channel_list=False,suffix_list=False)
print(info["capacity"])
print(info['suffix'])
print(info['channel'])


#Demo3 process nist reverse gt_image
da=data_assistant()
src='/data/hszhu/dataset/Mydataset/NIST16/mask/'
dstpath='/data/hszhu/dataset/Mydataset/NIST16/mask/GT/'
# datalist=da.get_data(src,depth=2)
da.converter(src, depth=2, dstpath=dstpath, to_png=True, is_groundtruth=True,mask_process_rules=nist_process)
# check whether it is right
info=da.get_info('/data/hszhu/dataset/Mydataset/NIST16/mask/GT/',depth=1)
for k in info.keys():
    print(info[k])


#demo4 process the Columbia gt_image 注：这个数据集tpGT里有一张图是损坏的确实信息，会造成分辨率不同的问题，直接跳过处理
#流程提取特定的图片 获得datalist
#依据datalist进行处理
da=data_assistant()
src='/data/Forgery/test/Columbia/4cam_splc/4cam_splc/edgemask/'
dstpath="/data/Forgery/test/Columbia/4cam_splc/4cam_splc/gt/"
datalist=da.get_data(src,depth=1)
newlist=[]
for data in datalist:
    name,_=osp.splitext(data)
    if name[-1]=='k':
        newlist.append(data)
# ImageFile.LOAD_TRUNCATED_IMAGES=True #如果读入文件出现损坏可以用这句话跳过损坏文件 可以把这句话加到converter里面
da.converter(datalist=newlist,dstpath=dstpath,to_png=True,process_rules=Columbia_process)
#check
info=da.get_info(srcfile=dstpath,depth=1)
for k in info.keys():
    print(info[k])



#Demo5 文件改名 如后缀_mask.png,_gt.png去除语义后缀
#流程查找 pair 检查
#改名与保存
#示例casia一代测试集GT 整合 改名 并压缩通道
src='/data/Forgery/test/CASIA/TP/GT/'
dst="/data/Forgery/test/CASIA/TP/rename_concatGT/"
da=data_assistant()
# datalist=da.get_data(dst,1)


# da.rename(datalist=datalist,dstpath=dst,rename_rules=(lambda x:x[:-3]))
# da.converter(srcfile=dst,to_png=True,is_groundtruth=True)

info=da.get_info(srcfile=dst,depth=1)
for k in info.keys():
    print(info[k])
# for x in info['suffix'].keys():
#     print(x)




#Demo 5 检查哥伦比亚数据集是否ready 因为这里的数据集是真的用的data 不建议跑这个demo
image_path="/data/Forgery/test/Columbia/TP/image/"
mask_path="/data/Forgery/test/Columbia/TP/GT/"
da=data_assistant()

#处理后缀_edgemask
# da.rename(srcfile="/data/Forgery/test/Columbia/TP/GT/",depth=1,rename_rules=lambda x:x[:-9]) #只能改一次不然第二次就要删除多余字符了
# da.ready_for_train(image_path=image_path,annotation_path=mask_path,return_error_list=True)
# info=da.get_info(image_path)
# print(list(info['suffix']))



#Demo 7 检查各种数据集 是否ready
image_path="/data/Forgery/test/COVERAGE/TP/image/"
mask_path="/data/Forgery/test/COVERAGE/TP/GT/"
# _,error_list=da.ready_for_train(image_path=image_path,annotation_path=mask_path,return_error_list=True)
# print(error_list)


#Demo 8 剔除COVERAGE的异常数据
mask_error_list=['/data/Forgery/test/COVERAGE/TP/GT/59.png', '/data/Forgery/test/COVERAGE/TP/GT/48.png', '/data/Forgery/test/COVERAGE/TP/GT/61.png', '/data/Forgery/test/COVERAGE/TP/GT/57.png', '/data/Forgery/test/COVERAGE/TP/GT/95.png', '/data/Forgery/test/COVERAGE/TP/GT/55.png', '/data/Forgery/test/COVERAGE/TP/GT/56.png', '/data/Forgery/test/COVERAGE/TP/GT/58.png', '/data/Forgery/test/COVERAGE/TP/GT/41.png']
data_error_list=['/data/Forgery/test/COVERAGE/TP/image/58.jpg', '/data/Forgery/test/COVERAGE/TP/image/59.jpg', '/data/Forgery/test/COVERAGE/TP/image/95.jpg', '/data/Forgery/test/COVERAGE/TP/image/56.jpg', '/data/Forgery/test/COVERAGE/TP/image/61.jpg', '/data/Forgery/test/COVERAGE/TP/image/57.jpg', '/data/Forgery/test/COVERAGE/TP/image/48.jpg', '/data/Forgery/test/COVERAGE/TP/image/41.jpg', '/data/Forgery/test/COVERAGE/TP/image/55.jpg']
print(len(data_error_list))
print(len(mask_error_list))
for address in mask_error_list:
    os.remove(address)
for address in data_error_list:
    os.remove(address)
da.ready_for_train(image_path=image_path,annotation_path=mask_path,return_error_list=True)
#再补一个直接把list里的东西remove掉的类函数吧 见da.remove_data()


#del_repeate_data待完善
"""

# Demo comp_Raise image 改名以和mask匹配
def  rename_rules(filename):
    i = 1 if filename[:3] =='100' else 0
    j = 1 if filename[25+i:28+i] =='100' else 0
    k = 1 if filename[54+i+j:57+i+j] =='100' else 0
    l = 1 if filename[-3:] == '100' else 0
    return filename[:15+i]+filename[27+i+j:44+i+j]+filename[56+i+j+k:-(12+l)]
# da=data_assistant()
# src="/data/ipad/Forgery/tampCOCO/bcmc_images/"
# dst="/data/ipad/Forgery/tampCOCO/compress_image/"
# da.rename(srcfile=src,depth=1,dstpath=dst,rename_rules=rename_rules)


# da.ready_for_train(image_path="/data/ipad/Forgery/tampCOCO/compress_image/",annotation_path="/data/ipad/Forgery/train/tampRAISE/TP/GT")
# info=da.get_info(srcfile="/data/ipad/Forgery/train/tampRAISE/TP/compress_image/",suffix=False,channel=False)
# for k in info.keys():
#     print(info[k])
# print("success")
# da=data_assistant()

def glide_image_rename(filename):
    filename2=filename[25:-3]
    return filename2
def glide_mask_name(filename):
    for j in range(len(filename)):
        if filename[j]=='_':
            return filename[j+1:-5]

# # da.rename(srcfile=fake_src,dstpath="/data/ipad/Forgery/test/CocoGlide/TP/IMAGE/",rename_rules=glide_image_rename,repeate_name_dst="/data/ipad/Forgery/test/CocoGlide/TP/repe/")
# # da.rename(srcfile="/data/ipad/Forgery/test/CocoGlide/TP/mask/",dstpath="/data/ipad/Forgery/test/CocoGlide/TP/GT/",rename_rules=glide_mask_name,repeate_name_dst="/data/ipad/Forgery/test/CocoGlide/TP/repeate/")
# da.ready_for_train(image_path="/data/ipad/Forgery/test/CocoGlide/TP/image/",annotation_path="/data/ipad/Forgery/test/CocoGlide/TP/GT/",)

# 利用Nist原始数据根据txt读取AU GT
da=data_assistant()
prefix1="/data/ipad/Forgery/test/OpenForensics_Dev_part/TP/image/"
prefix2="/data/ipad/Forgery/test/OpenForensics_Dev_part/TP/GT/"
# prefix3="/data/ipad/Forgery/test/Nist16_subset_160/TP/GT/"
# 打开文件
file_path ="/data/ipad/Forgery/test/OpenForensics_Dev_part/TP/OpenForensics_fake.txt"
with open(file_path, 'r') as file:
    # 读取文件内容到一个字符串列表
    lines = file.readlines()

# 创建不同的列表来存储内容
tplist = []
gtlist = []
aulist=[]
tp_error_list=[
    '/data/ipad/Forgery/test/OpenForensics_Dev_part/AU/image/04612c2c50.jpg',
    '/data/ipad/Forgery/test/OpenForensics_Dev_part/AU/image/ed5bd4a795.jpg',
    '/data/ipad/Forgery/test/OpenForensics_Dev_part/AU/image/a19629a8fe.jpg',
    '/data/ipad/Forgery/test/OpenForensics_Dev_part/AU/image/3034c43716.jpg',
    '/data/ipad/Forgery/test/OpenForensics_Dev_part/AU/image/961b538565.jpg',
]

gt_error_list=[
    '/data/ipad/Forgery/test/OpenForensics_Dev_part/AU/GT/04612c2c50.png',
    '/data/ipad/Forgery/test/OpenForensics_Dev_part/AU/GT/ed5bd4a795.png',
    '/data/ipad/Forgery/test/OpenForensics_Dev_part/AU/GT/a19629a8fe.png',
    '/data/ipad/Forgery/test/OpenForensics_Dev_part/AU/GT/3034c43716.png',
    '/data/ipad/Forgery/test/OpenForensics_Dev_part/AU/GT/961b538565.png'
]
# da.ready_for_train(image_path='/data/ipad/Forgery/test/OpenForensics_test/TP/image/',
#                    annotation_path='/data/ipad/Forgery/test/OpenForensics_test/TP/GT/')
# # rmlist=[]
# # 根据需要将内容分配到不同的列表中
# for line in lines:
#     # 这里假设每一行包含两个值，用空格分隔
#     parts = line.strip().split(',')
#     if len(parts) == 2:
#         value1, value2 = parts
#         tpimage=osp.join(prefix1,da.get_filename(value1))+"."+da.get_suffix(value1)
#         groundtruth=osp.join(prefix2,da.get_filename(value2))+"."+da.get_suffix(value2)
#         if tpimage not in tp_error_list:
#             tplist.append(tpimage)
#         if groundtruth not in gt_error_list:
#             gtlist.append(groundtruth)
#         # rmlist.append(osp.join(prefix3,value2))
#     else:
#         value=parts[0]
#         aulist.append(osp.join(prefix1,value))


# 打印结果
# print("List 1:", tplist[0])
# print("List 2:", gtlist[1])

# tp save
# print(len(tplist))


# da.remove_data(rmlist)
# da.converter(datalist=tp_error_list,dstpath="/data/ipad/Forgery/test/OpenForensics_test/TP/image/",to_jpg=True,)
# da.converter(datalist=gt_error_list,dstpath="/data/ipad/Forgery/test/OpenForensics_test/TP/GT/",to_png=True,is_groundtruth=True)
#
# _,err=da.ready_for_train(image_path="/data/ipad/Forgery/test/DSO-1/TP/image",
#                    annotation_path="/data/ipad/Forgery/test/DSO-1/TP/GT",
#                    img_suffix="jpg",ano_suffix='png')



# da=data_assistant()
# prefix="/data/ipad/Forgery/test/NIST4share/"
# # 打开文件
# file_path ="/data/ipad/Forgery/test/NIST4share/NIST16_real.txt"
# with open(file_path, 'r') as file:
#     # 读取文件内容到一个字符串列表
#     lines = file.readlines()
#
# # 创建不同的列表来存储内容
# reallist = []
#
# # 根据需要将内容分配到不同的列表中
# for line in lines:
#     part=line.strip()
#     # 这里假设每一行包含两个值，用空格分隔
#     reallist.append(osp.join(prefix,part))
# # print(reallist[0])
# da.converter(datalist=reallist,dstpath="/data/ipad/Forgery/test/Nist16_subset_160/AU/image/",to_jpg=True)
# da.converter(datalist=reallist,dstpath="/data/ipad/Forgery/test/Nist16_subset_160/AU/GT/",to_png=True,is_groundtruth=True,
#              process_rules=authentic_mask_generate_rules)
da=data_assistant()
def rename_cover0(name):
    return name[:-1]
def rename_cover1(name):
    return name[:-6]
import cv2
import numpy as np

# prefix_gt = '/data/ipad/Forgery/pristine/COVERAGE/TP/GT/'
# # 创建一个空列表来存储调整大小后的groundtruth图像
# error_gt=['/data/ipad/Forgery/pristine/COVERAGE/TP/GT/56.tif',
#        '/data/ipad/Forgery/pristine/COVERAGE/TP/GT/58.tif',
#        '/data/ipad/Forgery/pristine/COVERAGE/TP/GT/57.tif',
#        '/data/ipad/Forgery/pristine/COVERAGE/TP/GT/59.tif',
#        '/data/ipad/Forgery/pristine/COVERAGE/TP/GT/48.tif',
#        '/data/ipad/Forgery/pristine/COVERAGE/TP/GT/55.tif',
#        '/data/ipad/Forgery/pristine/COVERAGE/TP/GT/95.tif',
#        '/data/ipad/Forgery/pristine/COVERAGE/TP/GT/61.tif',
#        '/data/ipad/Forgery/pristine/COVERAGE/TP/GT/41.tif']
# # error = [x.replace('GT','image') for x in gt]
# true_gt_path = "/data/ipad/Forgery/test/COVERAGE/TP/GT/"
# for gt in error_gt:
#     name = osp.basename(gt).replace('.tif','.png')
#     path = osp.join(true_gt_path,name)
#     gt_i = cv2.imread(path)
#     save_path = osp.join(prefix_gt,name.replace('.png','.tif'))
#     print(save_path)
#     cv2.imwrite(filename=save_path,img=gt_i)
#
# image_paths = error
# groundtruth_paths = gt
# prefix="/data/ipad/Forgery/pristine/COVERAGE/pro"
# # 遍历图像和groundtruth路径
# for image_path, groundtruth_path in zip(sorted(image_paths), sorted(groundtruth_paths)):
#     # 读取图像和groundtruth
#     name=da.get_filename(image_path)
#     image = cv2.imread(image_path,flags=1)
#     gt_image = cv2.imread(groundtruth_path)
#
#     # 确保图像和groundtruth具有相同的大小
#     # if image.shape[:2] != gt_image.shape[:2]:
#     #     # 将groundtruth调整为与图像相同的大小
#     #     gt_image = cv2.resize(gt_image, (image.shape[1], image.shape[0]))
#     # 获取图像和groundtruth的尺寸
#     image_height, image_width, _ = image.shape
#     gt_height, gt_width, _ = gt_image.shape
#
#     crop_height = min(image_height, gt_height)
#     crop_width = min(image_width, gt_width)
#     # print(gt_width - crop_width)
#     top_left = gt_image[:crop_height, :crop_width]
#     top_right = gt_image[:crop_height, gt_width - crop_width:]
#     bottom_left = gt_image[gt_height - crop_height:, :crop_width]
#     bottom_right = gt_image[gt_height - crop_height:, gt_width - crop_width:]
#
#
#
#     #
#     # # 将图像和groundtruth叠加在一起
#     # cv2.imwrite(osp.join(prefix, name) +   '.png', gt_image)
#     # # stacked_image = np.hstack((image, gt_image))
#     # result = cv2.addWeighted(image, 1, gt_image, 1, 0)
#     # # cv2.imwrite(osp.join(prefix, name) +   'stack.png', stacked_image)
#     # cv2.imwrite(osp.join(prefix, name) +  'result.png', result)
#
#     for index,mask in enumerate([top_left,top_right,bottom_right,bottom_left]):
#         # 将图像和groundtruth叠加在一起
#         cv2.imwrite(osp.join(prefix,name)+f'({str(index)})'+'.tif',mask)
#         stacked_image = np.hstack((image, mask))
#         result = cv2.addWeighted(image, 1, mask, 1, 0)
#         # cv2.imwrite(osp.join(prefix,name)+f'({str(index)})'+'stack.png',stacked_image)
#         cv2.imwrite(osp.join(prefix, name) +f'({str(index)})'+ 'result.tif', result)





# da=data_assistant()

# da.converter(srcfile="/data/ipad/Forgery/dd/image/",dstpath="/data/ipad/Forgery/dd/GT/",
#              to_png=True,is_groundtruth=True,process_rules=authentic_mask_generate_rules)


"""
图像gt压缩表示--不丢失空间一般性
"""
#比较不同方式resize的GT图对比
#avg pool





#resize blinear
#
# import cv2
# import numpy as np
# prefix="/data/ipad/Forgery/test/NIST16/exp/downsam/"
# # 读取原始图像 #256/16=16
# image_path = "/data/ipad/Forgery/test/NIST16/GT/NC2016_0048.png"# 替换为你的真实图像路径
# # original_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
# original_image = cv2.imread(image_path)
# original_image = cv2.resize(original_image, (512, 512))
#
#
# # 定义patch大小
# patch_size = 16
#
# # 复制原始图像以便在上面绘制边界框
# image_with_patches = original_image.copy()
#
# # 循环遍历图像并绘制边界框
# for y in range(0, original_image.shape[0], patch_size):
#     for x in range(0, original_image.shape[1], patch_size):
#         cv2.rectangle(image_with_patches, (x, y), (x + patch_size, y + patch_size), (0, 255, 0), 2)  # 在每个patch上绘制绿色边界框
# cv2.imwrite(osp.join(prefix,'gt_withpatch.png'), image_with_patches)
#
#
#
#
#
#
#
# original_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
# original_image = cv2.resize(original_image, (512, 512))
# # # 使用双线性插值下采样（16倍）
# downsampled_bilinear = cv2.resize(original_image, (original_image.shape[1] // 16, original_image.shape[0] // 16), interpolation=cv2.INTER_LINEAR)
#
# # 使用平均池化下采样（16倍）
# kernel_size = 16
# downsampled_average_pooling = np.zeros((original_image.shape[0] // kernel_size, original_image.shape[1] // kernel_size), dtype=np.uint8)
#
# for i in range(0, original_image.shape[0], kernel_size):
#     for j in range(0, original_image.shape[1], kernel_size):
#         block = original_image[i:i+kernel_size, j:j+kernel_size]
#         average_value = np.mean(block)
#         downsampled_average_pooling[i//kernel_size, j//kernel_size] = average_value
# #
# # # 保存结果图像
# cv2.imwrite(osp.join(prefix,'downsampled_bilinear.png'), downsampled_bilinear)
# cv2.imwrite(osp.join(prefix,'downsampled_average_pooling.png'), downsampled_average_pooling)
# #
# # import cv2
# # import numpy as np



# # 读取原始图像
# original_image_path= "/data/ipad/Forgery/test/NIST16/exp/downsam/NC2016_0048.png" # 替换为你的真实图像路径
#
# import cv2
# import numpy as np
#
# original_image_path="/data/ipad/Forgery/test/NIST16/exp/resizegt/NC2016_0048.jpg"
# original_image = cv2.imread(original_image_path)
# original_image = cv2.resize(original_image, (512, 512))
# cv2.imwrite("/data/ipad/Forgery/test/NIST16/exp/resizegt/image/NC2016_0048.jpg",original_image)
# # 读取线性下采样图像
# # linear_downsampled_path = "/data/ipad/Forgery/test/CocoGlide/TP/downsample/downsampled_bilinear.png"  # 替换为线性下采样的图像路径
# linear_downsampled_path="/data/ipad/Forgery/test/NIST16/exp/downsam/downsampled_bilinear.png"
# linear_downsampled = cv2.imread(linear_downsampled_path, cv2.IMREAD_GRAYSCALE)


# # 读取平均池化下采样图像
# # average_pooling_downsampled_path = "/data/ipad/Forgery/test/CocoGlide/TP/downsample/downsampled_average_pooling.png"  # 替换为平均池化下采样的图像路径
# average_pooling_downsampled_path ="/data/ipad/Forgery/test/NIST16/exp/downsam/downsampled_average_pooling.png"
# average_pooling_downsampled = cv2.imread(average_pooling_downsampled_path, cv2.IMREAD_GRAYSCALE)
#
# # 使用双线性插值上采样回原始尺寸
# upsampled_linear = cv2.resize(linear_downsampled, (original_image.shape[1], original_image.shape[0]), interpolation=cv2.INTER_LINEAR)
# upsampled_average_pooling = cv2.resize(average_pooling_downsampled, (original_image.shape[1], original_image.shape[0]), interpolation=cv2.INTER_LINEAR)
#
# # 计算上采样图像与原始图像之间的差异
# difference_linear = cv2.absdiff(original_image, upsampled_linear)
# difference_average_pooling = cv2.absdiff(original_image, upsampled_average_pooling)
# # prefix="/data/ipad/Forgery/test/CocoGlide/TP/re_upsample/"
# prefix="/data/ipad/Forgery/test/NIST16/exp/reup/"
# cv2.imwrite(osp.join(prefix,'upsampled_linear.png'),upsampled_linear)
# cv2.imwrite(osp.join(prefix,'upsampled_average_pooling.png'),upsampled_average_pooling)

# print('hello')
da=data_assistant()#COV
image_path="/data/ipad/Forgery/pristine/COVERAGE/image/"
mask_path="/data/ipad/Forgery/pristine/COVERAGE/mask/"
fake_txt = "/data/ipad/Forgery/pristine/COVERAGE/coverage_fake.txt"
au_txt = "/data/ipad/Forgery/pristine/COVERAGE/coverage_real.txt"
data_root='/data/ipad/Forgery/pristine/COVERAGE/'
prefix = '/data/ipad/Forgery/pristine/COVERAGE/AU/'


# with open(fake_txt, 'r') as file:
#     # 读取文件内容到一个字符串列表
#     lines = file.readlines()
# for line in lines:
#     # 这里假设每一行包含两个值，用空格分隔
#     parts = line.strip().split(',')
#     if len(parts) == 2:
#         value1, value2 = parts
#         image_path = osp.join(data_root,value1)
#         gt_path = osp.join(data_root,value2)
#         # print(image_path)
#         # print(gt_path)
#         # break
#         i_dst = osp.join(prefix,'image',osp.basename(image_path).replace('t.','.'))
#         g_dst = osp.join(prefix,'GT',osp.basename(gt_path).replace('forged.','.'))
#         # print(g_dst)
#         # # break
#         shutil.copyfile(image_path,i_dst)
#         shutil.copyfile(gt_path,g_dst)

# with open(au_txt, 'r') as file:
#     # 读取文件内容到一个字符串列表
#     lines = file.readlines()
# for line in lines:
#     # 这里假设每一行包含两个值，用空格分隔
#     parts = line.strip().split(',')
#     if len(parts) == 1:
#         value1 = parts[0]
#         # print(value1)
#         image_path = osp.join(data_root,value1)
#         # print(image_path)
#         # print(gt_path)
#         # break
#         i_dst = osp.join(prefix,'image',osp.basename(image_path).replace('t.','.'))
#
#         print(i_dst)
#         # break
#         shutil.copyfile(image_path,i_dst)
# data = da.get_data("/data/ipad/Forgery/pristine/COVERAGE/AU/image/")
# da.converter(datalist=data,dstpath="/data/ipad/Forgery/pristine/COVERAGE/AU/GT/",
#              to_png=True,is_groundtruth=True,process_rules=authentic_mask_generate_rules)

#COV有9张gt image 尺寸不匹配的，要纠错或者剔除
# _,error_list=da.ready_for_train(image_path="/data/ipad/Forgery/pristine/COVERAGE/TP/image/",
#                    annotation_path="/data/ipad/Forgery/pristine/COVERAGE/TP/GT/",
#                    img_suffix='tif',ano_suffix='tif',return_error_list=True)
# print(error_list)
# error=['/data/ipad/Forgery/pristine/COVERAGE/TP/GT/56.tif',
#        '/data/ipad/Forgery/pristine/COVERAGE/TP/GT/58.tif',
#        '/data/ipad/Forgery/pristine/COVERAGE/TP/GT/57.tif',
#        '/data/ipad/Forgery/pristine/COVERAGE/TP/GT/59.tif',
#        '/data/ipad/Forgery/pristine/COVERAGE/TP/GT/48.tif',
#        '/data/ipad/Forgery/pristine/COVERAGE/TP/GT/55.tif',
#        '/data/ipad/Forgery/pristine/COVERAGE/TP/GT/95.tif',
#        '/data/ipad/Forgery/pristine/COVERAGE/TP/GT/61.tif',
#        '/data/ipad/Forgery/pristine/COVERAGE/TP/GT/41.tif']
#
# _,error = da.ready_for_train(image_path="/data/ipad/Forgery/pristine/COVERAGE/TP/image/",
#                    annotation_path="/data/ipad/Forgery/pristine/COVERAGE/TP/GT/",
#                    img_suffix='tif',
#                    ano_suffix='tif',
#                    return_error_list=True)
