_base_ = [
    '../_base_/models/SAM.py', '../_base_/datasets/forensics.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_160k.py'
]
crop_size = (512, 512)
data_preprocessor = dict(size=crop_size)
train_dataloader = dict(batch_size=8, num_workers=4)

model = dict(
    data_preprocessor=data_preprocessor,
    backbone=dict(
        type='SAM',
        arch='base',
        # frozen_stages=12,
    ),
    decode_head=dict(num_classes=2))

default_hooks = dict(
    checkpoint=dict(interval=8000,max_keep_ckpts=5),
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
