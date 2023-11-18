_base_ = [
    '../_base_/datasets/forensics_detection.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_160k.py'
]
out_dir="/data/ipad/Forgery/hszhu/out/detection/columbia/"
val_evaluator=dict(type='MyFmeasure', iou_metrics=['F1_th','ACC' ])
test_evaluator=dict(type='MyFmeasure',
                    # iou_metrics=['F1_th' ,'F1_best'] ,
                    iou_metrics=[ 'ACC' ,'AUC'] ,
                    # output_dir = out_dir,
                    need_noise=True,
                    need_conf = True,)

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
        stage_cfg=dict(stage='detection'),
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
        conf_cfg=True,  #whether use CONF
        detect_head_cfg=True,  # whether predict image level label
        align_corners=False,
        init_cfg=dict(type='Kaiming', mode='fan_in', nonlinearity='relu'),
        #不用担心loss_decode参数不传报错，basedecode里自动设置为CEloss
        det_loss_decode=[ #自定义初始化在decoder时
            dict(type='ConfLoss', loss_name='loss_conf', loss_weight=1.0),
            dict(type='DetectionLoss', loss_name='loss_det', loss_weight=0.5,)], #如果要在这一阶段加入其他损失需要自定义loss_sta标签为else
        ),
    train_cfg=dict(),
    test_cfg=dict(mode='whole'))

default_hooks = dict(
    checkpoint=dict(interval=16000),
)

train_cfg = dict(
    type='IterBasedTrainLoop', max_iters=160000, val_interval=16000)

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
        end=160000,
        by_epoch=False,
    )
]
