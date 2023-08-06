# dataset settings

dataset_type = 'ForgeryDataset'
data_root = '/data/ipad/Forgery/'
crop_size = (512, 512)
#crop_size = (352, 352)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', binary=True),
    #dict(type='Resize', scale=crop_size),
    dict(
        type='RandomResize',
        scale=crop_size,
        ratio_range=(0.5, 2.0),
        keep_ratio=True),
    dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PhotoMetricDistortion'),
    dict(type='PackSegInputs')
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    #dict(type='Resize', scale=crop_size),
    # add loading annotation after ``Resize`` because ground truth
    # does not need to do resize data transform
    dict(type='LoadAnnotations', binary=True),
    dict(type='PackSegInputs')
]

dataset_CASIA_train1 = dict(
    type='RepeatDataset',
    times=1,
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        global_class=1,
        data_prefix=dict(
            img_path='train/CASIA 2.0/TP/image', seg_map_path='train/CASIA 2.0/TP/GT/'),
        pipeline=train_pipeline
    )
)
dataset_CASIA_train0 = dict(
    type='RepeatDataset',
    times=1,
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        global_class=0,
        data_prefix=dict(
            img_path='train/CASIA 2.0/AU/image', seg_map_path='train/CASIA 2.0/AU/GT/'),
        pipeline=train_pipeline
    )
)


dataset_IMD2020_train1 = dict(
    type='RepeatDataset',
    times=1,
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        global_class=1,
        data_prefix=dict(
            img_path='train/IMD2020/TP/image', seg_map_path='train/IMD2020/TP/GT'),
        pipeline=train_pipeline
    )
)
dataset_IMD2020_train0 = dict(
    type='RepeatDataset',
    times=1,
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        global_class=0,
        data_prefix=dict(
            img_path='train/IMD2020/AU/image', seg_map_path='train/IMD2020/AU/GT'),
        pipeline=train_pipeline
    )
)

dataset_tampCOCO_train1 = dict(
    type='RepeatDataset',
    times=1,
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        global_class=1,
        data_prefix=dict(
            img_path='train/tampCOCO/image', seg_map_path='train/tampCOCO/GT'),
        pipeline=train_pipeline
    )
)

dataset_tampRAISE_train1 = dict(
    type='RepeatDataset',
    times=1,
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        global_class=1,
        data_prefix=dict(
            img_path='train/tampRAISE/TP/image', seg_map_path='train/tampRAISE/TP/GT'),
        pipeline=train_pipeline
    )
)
dataset_tampRAISE_train0 = dict(
    type='RepeatDataset',
    times=1,
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        global_class=0,
        data_prefix=dict(
            img_path='train/tampRAISE/AU/image', seg_map_path='train/tampRAISE/AU/GT'),
        pipeline=train_pipeline
    )
)

dataset_OpenForensics_train1 = dict(
    type='RepeatDataset',
    times=1,
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        global_class=1,
        data_prefix=dict(
            img_path='test/OpenForensics/TP/image', seg_map_path='test/OpenForensics/TP/GT'),
        pipeline=train_pipeline
    )
)

concatenate_dataset = dict(
    type='ConcatDataset',
    #datasets=[dataset_CASIA_train1, dataset_CASIA_train0, dataset_IMD2020_train1, dataset_IMD2020_train0, dataset_tampCOCO_train1, dataset_tampRAISE_train1, dataset_tampRAISE_train0]
    datasets=[dataset_CASIA_train1, dataset_CASIA_train0, dataset_IMD2020_train1, dataset_IMD2020_train0, dataset_tampCOCO_train1, dataset_tampRAISE_train0]
)

train_dataloader = dict(
    batch_size=4,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='InfiniteSampler', shuffle=True),
    dataset=concatenate_dataset
)


# Test_dataset = dict(
#     type='ConcatDataset',
#     datasets =
# )


val_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(
            img_path='test/Columbia/TP/image/',
            seg_map_path='test/Columbia/TP/GT/'),
        pipeline=test_pipeline)
   # dataset=dict(
   #     type='ConcatDataset',
   #     datasets=[CHAMELEON_Test_dataset,
   #     CAMO_Test_dataset,
   #     COD10K_Test_dataset,
   #     NC4K_Test_dataset])
)
test_dataloader = val_dataloader

val_evaluator = dict(type='IoUMetric', iou_metrics=['mIoU'])
test_evaluator = val_evaluator


