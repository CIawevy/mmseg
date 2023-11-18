# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved.

_base_ = [
    '../_base_/datasets/forensics.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_160k.py'
]
val_evaluator=dict(type='MyFmeasure', iou_metrics=['F1_th']) #调参
test_evaluator=dict(type='MyFmeasure',
                    iou_metrics=['F1_th' ,'F1_best'] ,
                    # output_dir = out_dir,
                    need_noise=False,
                    need_conf=False)
# model settings
crop_size = (512, 512)
norm_cfg = dict(type='SyncBN', requires_grad=True)
checkpoint="/home/ipad_ind/hszhu/pretrained/clawer_hrvit_b1.pth"
train_dataloader = dict(batch_size=2, num_workers=4)
data_preprocessor = dict(
    type='SegDataPreProcessor',
    size=crop_size,
    bgr_to_rgb=True,
    pad_val=0,
    seg_pad_val=255)
model = dict(
    type='EncoderDecoder',
    backbone=dict(
        type='hrvit_b1',
        drop_path_rate=0.1,
        with_cp=False,
        norm_cfg=norm_cfg,
        init_cfg=dict(type='Pretrained', checkpoint=checkpoint),
    ),
    decode_head=dict(
        type="SegformerHead",
        in_channels=[32, 64, 128, 256],
        in_index=[0, 1, 2, 3],
        channels=256,
        dropout_ratio=0.1,
        num_classes=2,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=[
            dict(type='CrossEntropyLoss', loss_name='loss_ce', loss_weight=3.0, class_weight=[0.5, 2.5]),
            dict(type='DiceLoss', loss_name='loss_dice', loss_weight=7.0)
        ]),
    train_cfg=dict(),
    test_cfg=dict(mode='whole'))

default_hooks = dict(
    checkpoint=dict(interval=16000),
)

train_cfg = dict(
    type='IterBasedTrainLoop', max_iters=160000, val_interval=16000)

# AdamW optimizer, no weight decay for position embedding & layer norm in backbone
optimizer = dict(_delete_=True, type='AdamW', lr=0.00006, betas=(0.9, 0.999), weight_decay=0.01,
                 paramwise_cfg=dict(custom_keys={'absolute_pos_embed': dict(decay_mult=0.),
                                                 'relative_position_bias_table': dict(decay_mult=0.),
                                                 'norm': dict(decay_mult=0.),
                                                 'head': dict(lr_mult=10.)}))

lr_config = dict(_delete_=True, policy='poly',
                 warmup='linear',
                 warmup_iters=1500,
                 warmup_ratio=1e-6,
                 power=1.0, min_lr=0.0, by_epoch=False)

# data=dict(samples_per_gpu=2)
# evaluation = dict(interval=16000, metric='mIoU')