_base_ = [
   './Tamp_Sam_large.py'
]
checkpoint="/home/ipad_ind/hszhu/pretrained/vit-huge-p16_SAM-pre_3rdparty_sa1b-1024px_20230411-3f13c653.pth"
crop_size = (512, 512)
data_preprocessor = dict(size=crop_size)
train_dataloader = dict(batch_size=4, num_workers=4)

model = dict(
    data_preprocessor=data_preprocessor,
    backbone=dict(
        arch='huge',
        init_cfg=dict(type='Pretrained',checkpoint=checkpoint),
        out_indices=-1,#v4 v3
        # out_indices=[7,15,23,31], #v3.5
        # v3通过网络定义global输出，已经缺省最后一层输出 v3
        frozen_stages=32,# froze when adapting
    ),
    decode_head=dict(num_classes=2,
                     in_channels=[1280],#v4初始化实则无效参数
                     # in_channels/in_index影响的仅有decoder之前准备数据的那个函数，
                     # 另外decoder在初始化时需要检查 input 与这两个参数的一致性
                     input_channel=1280, #v4 pyramid输入参数
                     in_index=[0], #large被更新，这里再次覆写v4
                     # in_channels=[320,320,320,320], #v3.5
                     # in_index=[0, 1, 2, 3] ,       #v3.5
                     # in_channels=[1280,1280,1280,1280,256], #v3
                     # in_index=[0,1,2,3,4]               #v3
                     ))

# default_hooks = dict(
#     checkpoint=dict(interval=16000,max_keep_ckpts=16),
# )
default_hooks = dict(  #调参使用
    checkpoint=dict(interval=2000,save_best='mFscore',rule='greater'),
)

train_cfg = dict(
    type='IterBasedTrainLoop', max_iters=192000, val_interval=2000)

optim_wrapper = dict(
    _delete_=True,
    type='OptimWrapper',
    optimizer=dict(
        type='AdamW', lr=0.0002, betas=(0.9, 0.999), weight_decay=0.01), #2E-4
    paramwise_cfg=dict(
        custom_keys={
            'pos_block': dict(decay_mult=0.),
            'norm': dict(decay_mult=0.),
            'head': dict(lr_mult=10.)
        }))
#
param_scheduler = [
    dict(
        type='LinearLR', start_factor=1e-6, by_epoch=False, begin=0, end=1500),
    dict(
        type='PolyLR',
        eta_min=0.0,
        power=1.0,
        begin=1500,
        end=192000,
        by_epoch=False,
    )
]
