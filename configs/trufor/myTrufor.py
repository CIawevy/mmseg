_base_ = [
    '../_base_/datasets/forensics.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_160k.py'
]
out_dir="/data/ipad/Forgery/hszhu/out/columbia/test/"
val_evaluator=dict(type='MyFmeasure', iou_metrics=['F1_th']) #调参
test_evaluator=dict(type='MyFmeasure',
                    iou_metrics=['F1_th' ,'F1_best'] ,
                    # output_dir = out_dir,
                    need_noise=True,
                    need_conf=False)

# model settings
crop_size = (512, 512)
norm_cfg = dict(type='SyncBN', requires_grad=True)
checkpoint="/home/ipad_ind/hszhu/pretrained/weights/RGBX_mit_b2.pth"
train_dataloader = dict(batch_size=8, num_workers=4)
data_preprocessor = dict(
    type='MySegDataPreProcessor', #/256
    size=crop_size,
    bgr_to_rgb=True,
    pad_val=0,
    seg_pad_val=255)
model = dict(
    type='MyEncoderDecoder',
    data_preprocessor=data_preprocessor,
    backbone=dict(
        type='RGBXTransformer', #mit-b2 参数
        img_size=512,
        patch_size=4,
        embed_dims=[64, 128, 320, 512],
        num_heads=[1, 2, 5, 8],
        qkv_bias=True,
        drop_rate=0.0,
        drop_path_rate=0.1,
        init_cfg=dict(type='Pretrained', checkpoint=checkpoint),
        stage_cfg=dict(stage='segment'),
        raw=True,
    ),
    decode_head=dict(
        type='DualDecoderHead', #
        in_channels=[64, 128, 320, 512],
        in_index=[0, 1, 2, 3],
        channels=512,
        dropout_ratio=0.1,
        num_classes=2,
        norm_cfg=norm_cfg,
        conf_cfg=False,  #whether use CONF
        detect_head_cfg=False,  # whether predict image level label
        align_corners=False,
        init_cfg=dict(type='Kaiming', mode='fan_in', nonlinearity='relu'),
        loss_decode=[
            dict(type='CrossEntropyLoss', loss_name='loss_ce', loss_weight=3.0,class_weight=[0.5,2.5]),
            dict(type='DiceLoss', loss_name='loss_dice', loss_weight=7.0)
        ]),
    train_cfg=dict(),
    test_cfg=dict(mode='whole'))

default_hooks = dict(
    checkpoint=dict(interval=10000),
)

train_cfg = dict(
    type='IterBasedTrainLoop', max_iters=100000, val_interval=10000)

optim_wrapper = dict(
    _delete_=True,
    type='OptimWrapper',
    optimizer=dict(
        type='SGD', lr=0.005, momentum=0.9, weight_decay=0.0001),#原文設置
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
        end=100000,
        by_epoch=False,
    )
]
