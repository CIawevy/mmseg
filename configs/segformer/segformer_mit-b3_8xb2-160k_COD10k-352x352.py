_base_ = ['./segformer_mit-b0_8xb2-160k_COD10k-352x352.py']
checkpoint ="/home/ipad_ind/hszhu/pretrained/mit_b3.pth"  #hszhu
crop_size = (512,512)
data_preprocessor = dict(size=crop_size)
# model settings
model = dict(
    data_preprocessor=data_preprocessor,
    pretrained=checkpoint,
    backbone=dict(
        embed_dims=64, num_heads=[1, 2, 5, 8], num_layers=[3, 4, 18, 3]),
    decode_head=dict(in_channels=[64, 128, 320, 512],num_classes=2))
