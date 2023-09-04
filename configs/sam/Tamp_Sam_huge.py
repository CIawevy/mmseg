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
        # out_indices=[0,7,15,23,31],
        frozen_stages=32,# froze when adapting
    ),
    decode_head=dict(num_classes=2,
                     in_channels=[1280, 1280, 1280, 1280, 256],
                     in_index=[0, 1, 2, 3, 4],
                     ))

default_hooks = dict(
    checkpoint=dict(interval=16000,max_keep_ckpts=12),
)

train_cfg = dict(
    type='IterBasedTrainLoop', max_iters=160000, val_interval=16000)

optim_wrapper = dict(
    _delete_=True,
    type='OptimWrapper',
    optimizer=dict(
        type='AdamW', lr=0.00006, betas=(0.9, 0.999), weight_decay=0.01),
    paramwise_cfg=dict(
        custom_keys={
            'pos_block': dict(decay_mult=0.),
            'norm': dict(decay_mult=0.),
            'head': dict(lr_mult=10.)
        }))

param_scheduler = [
    dict(
        type='LinearLR', start_factor=1e-6, by_epoch=False, begin=0, end=1500),
    dict(
        type='PolyLR',
        eta_min=0.0,
        power=1.0,
        begin=1500,
        end=160000,
        by_epoch=False,
    )
]
