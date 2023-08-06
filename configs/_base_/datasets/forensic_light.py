# dataset settings
dataset_type = 'ForensicsDataset'
data_root = "/data/ipad/Forgery/"
crop_size = (512, 512)
# crop_size = (352, 352)
# crop_size = (1024, 1024) #SAM


train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations',),#label_map={0:0,255:1}
    dict(type='Resize', scale=crop_size),
    dict(
        type='RandomResize',
        scale=(1024,1024),
        ratio_range=(0.5, 1.5),
        keep_ratio=False),
    dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.5),
    # dict(type='PhotoMetricDistortion'),
    dict(type='PackSegInputs')
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=crop_size),
    # add loading annotation after ``Resize`` because ground truth
    # does not need to do resize data transform
    dict(type='LoadAnnotations'),
    dict(type='PackSegInputs')
]

#CASIAv2
CASIAv2_tamp_train_dataset=dict(
    type=dataset_type,
    data_root='/data/ipad/Forgery/train/CASIA 2.0/TP/',
    data_prefix=dict(
        img_path="image",
        seg_map_path="GT"),
    pipeline=train_pipeline)

CASIAv2_authentic_train_dataset=dict(
    type=dataset_type,
    data_root='/data/ipad/Forgery/train/CASIA 2.0/AU/',
    data_prefix=dict(
        img_path="image",
        seg_map_path="GT"),
    is_real=True,
    pipeline=train_pipeline)

CASIAv2_all_train_dataset=dict(
    type='ConcatDataset',
    datasets=[CASIAv2_tamp_train_dataset,CASIAv2_authentic_train_dataset]
)
CASIAv2_repeate_train_dataset=dict(
    type='RepeatDataset',
    times=5,
    dataset=CASIAv2_all_train_dataset

)

#IMD2020
IMD2020_tamp_train_dataset=dict(
    type=dataset_type,
    data_root="/data/ipad/Forgery/train/IMD2020/TP/",
    data_prefix=dict(
        img_path="image",
        seg_map_path="GT"),
    pipeline=train_pipeline)

IMD2020_authentic_train_dataset=dict(
    type=dataset_type,
    data_root="/data/ipad/Forgery/train/IMD2020/AU/",
    data_prefix=dict(
        img_path="image",
        seg_map_path="GT"),
    is_real=True,
    pipeline=train_pipeline)

IMD2020_all_train_dataset=dict(
    type='ConcatDataset',
    datasets=[IMD2020_authentic_train_dataset,IMD2020_tamp_train_dataset]
)

IMD2020_repeate_train_dataset=dict(
    type='RepeatDataset',
    times=3,
    dataset=IMD2020_all_train_dataset
)

#tampCOCO
tampCOCO_train_dataset=dict(
    type=dataset_type,
    data_root="/data/ipad/Forgery/train/tampCOCO/",
    data_prefix=dict(
        img_path="image",
        seg_map_path="GT"),
    indices=50000,
    pipeline=train_pipeline)

#tampRAISE
tampRAISE_tamp_train_dataset=dict(
    type=dataset_type,
    data_root="/data/ipad/Forgery/train/tampRAISE/TP/",
    data_prefix=dict(
        img_path="image",
        seg_map_path="GT"),
    indices=30000,
    pipeline=train_pipeline)

tampRAISE_compress_tamp_train_dataset=dict(
    type=dataset_type,
    data_root="/data/ipad/Forgery/train/tampRAISE/TP/",
    data_prefix=dict(
        img_path="compress_image",
        seg_map_path="GT"),
    # indices=30000,
    pipeline=train_pipeline)

tampRAISE_authentic_train_dataset=dict(
    type=dataset_type,
    data_root="/data/ipad/Forgery/train/tampRAISE/AU/",
    data_prefix=dict(
        img_path="image",
        seg_map_path="GT"),
    pipeline=train_pipeline)


train_dataloader = dict(
    batch_size=8,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='InfiniteSampler', shuffle=False),
    dataset=dict(
        type='ConcatDataset',
        datasets=[

CASIAv2_all_train_dataset,IMD2020_repeate_train_dataset,

                 ]))
# tampRAISE_compress_tamp_train_dataset,
# tampRAISE_tamp_train_dataset,
# tampRAISE_authentic_train_dataset,

# tampCOCO_train_dataset,


# CASIAv2_tamp_train_dataset,
# CASIAv2_authentic_train_dataset,

# IMD2020_tamp_train_dataset,
# IMD2020_authentic_train_dataset,


# Test_dataset = dict(
#     type='ConcatDataset',
#     datasets =
# )

#CASIAv1_plus

CASIAv1_plus_authentic_test_dataset=dict(
    type=dataset_type,
    data_root="/data/ipad/Forgery/test/_TEST_CASIAv1+/AU/",
    data_prefix=dict(
        img_path="image",
        seg_map_path="GT"),
    indices=50,
    pipeline=test_pipeline)

CASIAv1_plus_tamp_test_dataset=dict(
    type=dataset_type,
    data_root="/data/ipad/Forgery/test/_TEST_CASIAv1+/Tp/",
    data_prefix=dict(
        img_path="image",
        seg_map_path="GT"),
    indices=50,
    pipeline=test_pipeline)

CASIAv1_plus_all_test_dataset=dict(
    type='ConcatDataset',
    datasets=[CASIAv1_plus_authentic_test_dataset,CASIAv1_plus_tamp_test_dataset])


#Columbia



Columbia_authentic_test_dataset=dict(
    type=dataset_type,
    data_root="/data/ipad/Forgery/test/Columbia/AU/",
    data_prefix=dict(
        img_path="image",
        seg_map_path="GT"),
    pipeline=test_pipeline)

Columbia_tamp_test_dataset=dict(
    type=dataset_type,
    data_root="/data/ipad/Forgery/test/Columbia/TP/",
    data_prefix=dict(
        img_path="image",
        seg_map_path="GT"),
    # indices=5,
    pipeline=test_pipeline)

Columbia_all_test_dataset=dict(
    type='ConcatDataset',
    datasets=[Columbia_authentic_test_dataset,Columbia_tamp_test_dataset])

#COVERAGE

COVERAGE_authentic_test_dataset=dict(
    type=dataset_type,
    data_root="/data/ipad/Forgery/test/COVERAGE/AU/",
    data_prefix=dict(
        img_path="image",
        seg_map_path="GT"),
    pipeline=test_pipeline)

COVERAGE_tamp_test_dataset=dict(
    type=dataset_type,
    data_root="/data/ipad/Forgery/test/COVERAGE/TP/",
    data_prefix=dict(
        img_path="image",
        seg_map_path="GT"),
    pipeline=test_pipeline)

COVERAGE_all_test_dataset=dict(
    type='ConcatDataset',
    datasets=[COVERAGE_authentic_test_dataset,COVERAGE_tamp_test_dataset])

#NIST16
NIST16_tamp_test_dataset=dict(
    type=dataset_type,
    data_root="/data/ipad/Forgery/test/NIST16/",
    data_prefix=dict(
        img_path="image",
        seg_map_path="GT"),
    indices=10,
    pipeline=test_pipeline)

#OpenForensics
OpenForensics_tamp_test_dataset=dict(
    type=dataset_type,
    data_root="/data/ipad/Forgery/test/OpenForensics/TP/",
    data_prefix=dict(
        img_path="image",
        seg_map_path="GT"),
    indices=30,
    pipeline=test_pipeline)

OpenForensics_authentic_test_dataset=dict(
    type=dataset_type,
    data_root="/data/ipad/Forgery/test/OpenForensics/AU/",
    data_prefix=dict(
        img_path="image",
        seg_map_path="GT"),
    indices=30,
    pipeline=test_pipeline)

OpenForensics_all_test_dataset=dict(
    type='ConcatDataset',
    datasets=[OpenForensics_tamp_test_dataset, OpenForensics_authentic_test_dataset])


tampIC13_test_dataset=dict(
    type=dataset_type,
    data_root="/data/ipad/Forgery/test/Tampered-IC13/",
    data_prefix=dict(
        img_path="test_img",
        seg_map_path="test_mask",),
    indices=5,
    pipeline=test_pipeline)


val_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='ConcatDataset',
        datasets=[CASIAv1_plus_all_test_dataset])

)

test_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='ConcatDataset',
        datasets=[COVERAGE_tamp_test_dataset
            ])

)
# Columbia_all_test_dataset,NIST16_tamp_test_dataset,CASIAv1_plus_all_test_dataset,COVERAGE_all_test_dataset
# test_dataloader = dict(
#     batch_size=1,
#     num_workers=4,
#     persistent_workers=True,
#     sampler=dict(type='DefaultSampler', shuffle=False),
#     dataset=dict(
#         type='ConcatDataset',
#         datasets=[CASIAv1_plus_all_test_dataset,
#                   Columbia_all_test_dataset,
#                   COVERAGE_all_test_dataset,
#                   NIST16_tamp_test_dataset,
#                   OpenForensics_all_test_dataset])
#
# )



# val_evaluator = dict(type='SodMetric', sod_metrics=['MAE'])
val_evaluator=dict(type='IoUMetric', iou_metrics=['mIoU'])
test_evaluator=dict(type='IoUMetric', iou_metrics=['mIoU'])
# test_evaluator = dict(type='SodMetric', sod_metrics=['MAE','E-measure','S-measure','wF-measure'])
# test_evaluator=val_evaluator
