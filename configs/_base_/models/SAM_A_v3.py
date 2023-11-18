# model settings
norm_cfg = dict(type='SyncBN', requires_grad=True)
# checkpoint="/data/hszhu/code/mmseg_project/pretrained/vit-base-p16_SAM-pre_3rdparty_sa1b-1024px_20230411-2320f9cc.pth"
checkpoint="/home/ipad_ind/hszhu/pretrained/vit-base-p16_SAM-pre_3rdparty_sa1b-1024px_20230411-2320f9cc.pth"
data_preprocessor = dict(
    type='SegDataPreProcessor',
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    bgr_to_rgb=True,
    pad_val=0,
    seg_pad_val=255)
model = dict(
    type='EncoderDecoder',
    data_preprocessor=data_preprocessor,
    backbone=dict(
        type='ViTSAMv3',
        arch='base',
        img_size=1024,
        patch_size=16,
        out_channels=256,
        use_abs_pos=True,
        use_rel_pos=True,
        window_size=14,
        init_cfg=dict(type='Pretrained', checkpoint=checkpoint),
        frozen_stages=12
    ),
    decode_head=dict(
        type='SegformerHead',
        in_channels=[768, 768, 768, 768, 256],
        in_index=[0, 1, 2, 3, 4],
        channels=512,
        dropout_ratio=0.1,
        num_classes=19,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=[
            dict(type='CrossEntropyLoss', loss_name='loss_ce', loss_weight=3.0),
            dict(type='DiceLoss', loss_name='loss_dice', loss_weight=7.0)
        ]),
    # model training and testing settings
    train_cfg=dict(),
    test_cfg=dict(mode='whole'))
