# dataset settings
dataset_type = 'ForensicsDataset'
data_root = '/data/ipad/Forgery/'
crop_size = (512, 512)
# crop_size = (352, 352)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', binary=True),
    dict(type='SelfRandomResize', ratio_range=(0.5, 1.5)),
    # dict(type='Resize', scale=crop_size),
    # dict(
    #     type='RandomResize',
    #     scale=crop_size,
    #     ratio_range=(0.5, 2.0),
    #     keep_ratio=True),
    dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
    # dict(type='RandomFlip', prob=0.5),
    # dict(type='PhotoMetricDistortion'),
    dict(type='JPEGcompression', quality_factor=(30, 100)),
    dict(type='PackSegInputs')
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    # dict(type='Resize', scale=(512,512)),#trufor 测试不需要resize
    # dict(type='JPEGcompression', quality_factor=(70,71)),
    # dict(type='SelfRandomResize', ratio_range=(1.0, 1.5)),
    # add loading annotation after ``Resize`` because ground truth
    # does not need to do resize data transform
    dict(type='LoadAnnotations', binary=True),
    dict(type='PackSegInputs')
]

CASIAv2_tamp_train_dataset = dict(
    type='RepeatDataset',
    times=34,
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        global_class=1,
        data_prefix=dict(
            img_path='train/CASIA 2.0/TP/image', seg_map_path='train/CASIA 2.0/TP/GT/'),
        pipeline=train_pipeline
    )
)
CASIAv2_authentic_train_dataset = dict(
    type='RepeatDataset',
    times=34,
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        global_class=0,
        data_prefix=dict(
            img_path='train/CASIA 2.0/AU/image', seg_map_path='train/CASIA 2.0/AU/GT/'),
        pipeline=train_pipeline
    )
)

IMD2020_tamp_train_dataset = dict(
    type='RepeatDataset',
    times=160,
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        global_class=1,
        data_prefix=dict(
            img_path='train/IMD2020/TP/image', seg_map_path='train/IMD2020/TP/GT'),
        pipeline=train_pipeline
    )
)
IMD2020_authentic_train_dataset = dict(
    type='RepeatDataset',
    times=160,
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        global_class=0,
        data_prefix=dict(
            img_path='train/IMD2020/AU/image', seg_map_path='train/IMD2020/AU/GT'),
        pipeline=train_pipeline
    )
)

tampCOCO_train_dataset = dict(
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

tampRAISE_tamp_train_dataset = dict(
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
tampRAISE_compress_tamp_train_dataset = dict(
    type='RepeatDataset',
    times=1,
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        global_class=1,
        data_prefix=dict(
            img_path='train/tampRAISE/TP/compress_image', seg_map_path='train/tampRAISE/TP/GT'),
        pipeline=train_pipeline
    )
)
tampRAISE_authentic_train_dataset = dict(
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

concatenate_dataset = dict(
    type='ConcatDataset',
    # datasets=[CASIAv2_tamp_train_dataset, dataset_CASIA_train0, IMD2020_tamp_train_dataset, dataset_IMD2020_train0, tampCOCO_train_dataset, tampRAISE_tamp_train_dataset, dataset_tampRAISE_train0]
    datasets=[CASIAv2_tamp_train_dataset,
              CASIAv2_authentic_train_dataset,
              IMD2020_tamp_train_dataset,
              IMD2020_authentic_train_dataset,
              tampCOCO_train_dataset,
              tampRAISE_compress_tamp_train_dataset,
              tampRAISE_authentic_train_dataset]
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
# CASIAv1_plus

"""

test_dataset

"""
CASIAv1_plus_authentic_test_dataset = dict(
    type=dataset_type,
    data_root="/data/ipad/Forgery/test/_TEST_CASIAv1+/AU/",
    global_class=0,
    data_prefix=dict(
        img_path="image",
        seg_map_path="GT"),
    pipeline=test_pipeline)

CASIAv1_plus_tamp_test_dataset = dict(
    type=dataset_type,
    data_root="/data/ipad/Forgery/test/_TEST_CASIAv1+/Tp/",
    global_class=1,
    data_prefix=dict(
        img_path="image",
        seg_map_path="GT"),
    pipeline=test_pipeline)

CASIAv1_plus_all_test_dataset = dict(
    type='ConcatDataset',
    datasets=[CASIAv1_plus_authentic_test_dataset, CASIAv1_plus_tamp_test_dataset])

# Columbia


Columbia_authentic_test_dataset = dict(
    type=dataset_type,
    data_root="/data/ipad/Forgery/pristine/Columbia/AU/",
    img_suffix = '.tif',
    seg_map_suffix='.png',
    global_class=0,
    data_prefix=dict(
        img_path="image",
        seg_map_path="GT"),
    pipeline=test_pipeline)

Columbia_tamp_test_dataset = dict(
    type=dataset_type,
    data_root="/data/ipad/Forgery/pristine/Columbia/TP/",
    img_suffix = '.tif',
    seg_map_suffix='.png',
    global_class=1,
    data_prefix=dict(
        img_path="image",
        seg_map_path="GT"),
    pipeline=test_pipeline)

Columbia_all_test_dataset = dict(
    type='ConcatDataset',
    datasets=[Columbia_authentic_test_dataset, Columbia_tamp_test_dataset])

# COVERAGE

COVERAGE_authentic_test_dataset = dict(
    type=dataset_type,
    data_root="/data/ipad/Forgery/pristine/COVERAGE/AU",
    img_suffix = '.tif',
    seg_map_suffix='.png',
    global_class=0,
    data_prefix=dict(
        img_path="image",
        seg_map_path="GT"),
    pipeline=test_pipeline)

COVERAGE_tamp_test_dataset = dict(
    type=dataset_type,
    data_root="/data/ipad/Forgery/pristine/COVERAGE/TP",
    img_suffix = '.tif',
    seg_map_suffix='.tif',
    global_class=1,
    data_prefix=dict(
        img_path="image",
        seg_map_path="GT"),
    pipeline=test_pipeline)

COVERAGE_all_test_dataset = dict(
    type='ConcatDataset',
    datasets=[COVERAGE_authentic_test_dataset, COVERAGE_tamp_test_dataset])

# NIST16
NIST16_564_tamp_test_dataset = dict(
    type=dataset_type,
    data_root="/data/ipad/Forgery/test/NIST16/",
    global_class=1,
    # indices=30,
    data_prefix=dict(
        img_path="image",
        seg_map_path="GT"),
    pipeline=test_pipeline)

NIST16_160_tamp_test_dataset = dict(
    type=dataset_type,
    data_root="/data/ipad/Forgery/test/Nist16_subset_160/TP/",
    global_class=1,
    # indices=30,
    data_prefix=dict(
        img_path="image",
        seg_map_path="GT"),
    pipeline=test_pipeline)

NIST16_160_auth_test_dataset = dict(
    type=dataset_type,
    data_root="/data/ipad/Forgery/test/Nist16_subset_160/AU/",
    global_class=0,
    # indices=30,
    data_prefix=dict(
        img_path="image",
        seg_map_path="GT"),
    pipeline=test_pipeline)

Nist16_160_test_all_dataset = dict(
    type='ConcatDataset',
    datasets=[NIST16_160_auth_test_dataset, NIST16_160_tamp_test_dataset])

NIST16_tampSP_test_dataset = dict(
    type=dataset_type,
    data_root="/data/ipad/Forgery/test/NIST16/",
    global_class=1,
    # indices=30,
    data_prefix=dict(
        img_path="splice",
        seg_map_path="GT"),
    pipeline=test_pipeline)
# OpenForensics
# open_au_test_indices=np.random.randint(0, 6670,2000)
# open_tp_test_indices=np.random.randint(0,9988,2000)
# config import 会报错 因为lazy import这里的配置文件无法使用import东西的功能

OpenForensics_tamp_test_dataset = dict(
    type=dataset_type,
    data_root="/data/ipad/Forgery/test/OpenForensics_test/TP",
    global_class=1,
    data_prefix=dict(
        img_path="image",
        seg_map_path="GT"),
    pipeline=test_pipeline)

OpenForensics_authentic_test_dataset = dict(
    type=dataset_type,
    data_root="/data/ipad/Forgery/test/OpenForensics_test/AU",
    global_class=0,
    data_prefix=dict(
        img_path="image",
        seg_map_path="GT"),
    pipeline=test_pipeline)

OpenForensics_all_test_dataset = dict(
    type='ConcatDataset',
    datasets=[OpenForensics_tamp_test_dataset, OpenForensics_authentic_test_dataset])

# CocoGlide

CocoGlide_tamp_test_dataset = dict(
    type=dataset_type,
    data_root="/data/ipad/Forgery/pristine/CocoGlide/TP/",
    img_suffix = '.png',
    seg_map_suffix='.png',
    global_class=1,
    data_prefix=dict(
        img_path='image',
        seg_map_path='GT'),
    # indices=5,
    pipeline=test_pipeline)

CocoGlide_authentic_test_dataset = dict(
    type=dataset_type,
    data_root="/data/ipad/Forgery/pristine/CocoGlide/AU",
    img_suffix='.png',
    seg_map_suffix='.png',
    global_class=0,
    data_prefix=dict(
        img_path='image',
        seg_map_path='GT',
    ),
    pipeline=test_pipeline,
)
CocoGlide_all_test_dataset = dict(
    type='ConcatDataset',
    datasets=[CocoGlide_tamp_test_dataset, CocoGlide_authentic_test_dataset])

# DSO-1
DSO_1_tamp_test_dataset = dict(
    type=dataset_type,
    data_root="/data/ipad/Forgery/pristine/DSO-1/TP/",
    img_suffix = '.png',
    seg_map_suffix='.png',
    global_class=1,
    data_prefix=dict(
        img_path='image',
        seg_map_path='GT'),
    # indices=5,
    pipeline=test_pipeline)


DSO_1_authentic_test_dataset = dict(
    type=dataset_type,
    data_root="/data/ipad/Forgery/pristine/DSO-1/AU/",
    img_suffix='.png',
    seg_map_suffix='.png',
    global_class=0,
    data_prefix=dict(
        img_path='image',
        seg_map_path='GT',
    ),
    pipeline=test_pipeline,
)
DSO_1_all_test_dataset = dict(
    type='ConcatDataset',
    datasets=[DSO_1_tamp_test_dataset, DSO_1_authentic_test_dataset])

val_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='ConcatDataset',
        datasets=[
            # NIST16_160_tamp_test_dataset,
            Columbia_all_test_dataset,
            # COVERAGE_tamp_test_dataset,
            # CASIAv1_plus_tamp_test_dataset,
        ])

)
# IMD2020
IMD2020_tamp_test_dataset = dict(
    type=dataset_type,
    data_root="/data/ipad/Forgery/train/IMD2020/TP/",
    global_class=1,
    data_prefix=dict(
        img_path='image',
        seg_map_path='GT'),
    # indices=5,
    pipeline=test_pipeline)

my_test_dataset = dict(
    type=dataset_type,
    data_root='/data/ipad/Forgery/test/NIST16/exp/resizegt/',
    global_class=1,
    data_prefix=dict(
        img_path='image',
        seg_map_path='GT'),
    # indices=5,
    pipeline=test_pipeline)

test_dataloader = dict( #det
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='ConcatDataset',
        datasets=[

            # CASIAv1_plus_all_test_dataset,

            # COVERAGE_all_test_dataset

            # Columbia_all_test_dataset,

            DSO_1_all_test_dataset,

            # Nist16_160_test_all_dataset

            # CocoGlide_all_test_dataset

            # my_test_dataset
        ])

)


# val_evaluator=dict(type='IoUMetric', iou_metrics=['mIoU'])
val_evaluator = dict(type='MyFmeasure', iou_metrics=['mFscore'])  # 调参
test_evaluator = dict(type='MyFmeasure', iou_metrics=['mFscore'])
# test_evaluator=dict(type='MyIoUMetric', iou_metrics=['myFscore'])


