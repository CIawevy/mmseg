_base_ = [
    '../_base_/models/SAM_A_v4.py', '../_base_/datasets/forensics.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_160k.py'
]
crop_size = (1024, 1024)
data_preprocessor = dict(size=crop_size)
train_dataloader = dict(batch_size=2, num_workers=4)

model = dict(
    data_preprocessor=data_preprocessor,
    decode_head=dict(num_classes=2))

default_hooks = dict(
    checkpoint=dict(interval=8000),
)

train_cfg = dict(
    type='IterBasedTrainLoop', max_iters=200000, val_interval=20000)

optim_wrapper = dict(
    _delete_=True,
    type='OptimWrapper',
    optimizer=dict(
        type='AdamW', lr=0.0001 ,betas=(0.9, 0.999), weight_decay=0.01),
    paramwise_cfg=dict(
        custom_keys={
            'pos_block': dict(decay_mult=0.),
            'norm': dict(decay_mult=0.),
            'head': dict(lr_mult=10.)
        }))

# param_scheduler = [
#     dict(
#         type='LinearLR', start_factor=1e-6, by_epoch=False, begin=0, end=1500),
#     dict(
#         type='PolyLR',
#         eta_min=0.0,
#         power=1.0,
#         begin=1500,
#         end=160000,
#         by_epoch=False,
#     )
# ]
param_scheduler = [
    dict(
        type='LinearLR', start_factor=1e-6, by_epoch=False, begin=0, end=1500),
    dict(
        type='CosineAnnealingLR',
        T_max=0,
        eta_min=0.0,
        begin=1500,
        end=200000,
        by_epoch=False,
    )
]
