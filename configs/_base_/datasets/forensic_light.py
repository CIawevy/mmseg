# dataset settings

dataset_type = 'ForensicsDataset'
data_root = '/data/ipad/Forgery/'
crop_size = (1024, 1024)
#crop_size = (352, 352)
# NIST OPENFOR 数据集图像太大 直接放进去会导致attention步骤超内存 90多个G显然是不可以的

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=crop_size),
    # add loading annotation after ``Resize`` because ground truth
    # does not need to do resize data transform
    dict(type='LoadAnnotations', binary=True),
    dict(type='PackSegInputs')
]


# Test_dataset = dict(
#     type='ConcatDataset',
#     datasets =
# )
#CASIAv1_plus

"""

test_dataset

"""
CASIAv1_plus_authentic_test_dataset=dict(
    type=dataset_type,
    data_root="/data/ipad/Forgery/test/_TEST_CASIAv1+/AU/",
    global_class=0,
    data_prefix=dict(
        img_path="image",
        seg_map_path="GT"),
    pipeline=test_pipeline)

CASIAv1_plus_tamp_test_dataset=dict(
    type=dataset_type,
    data_root="/data/ipad/Forgery/test/_TEST_CASIAv1+/Tp/",
    global_class=1,
    data_prefix=dict(
        img_path="image",
        seg_map_path="GT"),
    indices=2,
    pipeline=test_pipeline)

CASIAv1_plus_all_test_dataset=dict(
    type='ConcatDataset',
    datasets=[CASIAv1_plus_authentic_test_dataset,CASIAv1_plus_tamp_test_dataset])


#Columbia



Columbia_authentic_test_dataset=dict(
    type=dataset_type,
    data_root="/data/ipad/Forgery/test/Columbia/AU/",
    global_class=0,
    data_prefix=dict(
        img_path="image",
        seg_map_path="GT"),
    pipeline=test_pipeline)

Columbia_tamp_test_dataset=dict(
    type=dataset_type,
    data_root="/data/ipad/Forgery/test/Columbia/TP/",
    global_class=1,
    data_prefix=dict(
        img_path="image",
        seg_map_path="GT"),
    indices=5,
    pipeline=test_pipeline)

Columbia_all_test_dataset=dict(
    type='ConcatDataset',
    datasets=[Columbia_authentic_test_dataset,Columbia_tamp_test_dataset])

#COVERAGE

COVERAGE_authentic_test_dataset=dict(
    type=dataset_type,
    data_root="/data/ipad/Forgery/test/COVERAGE/AU/",
    global_class=0,
    data_prefix=dict(
        img_path="image",
        seg_map_path="GT"),
    pipeline=test_pipeline)

COVERAGE_tamp_test_dataset=dict(
    type=dataset_type,
    data_root="/data/ipad/Forgery/test/COVERAGE/TP/",
    global_class=1,
    data_prefix=dict(
        img_path="image",
        seg_map_path="GT"),
    indices=5,
    pipeline=test_pipeline)

COVERAGE_all_test_dataset=dict(
    type='ConcatDataset',
    datasets=[COVERAGE_authentic_test_dataset,COVERAGE_tamp_test_dataset])

#NIST16
NIST16_tamp_test_dataset=dict(
    type=dataset_type,
    data_root="/data/ipad/Forgery/test/NIST16/",
    global_class=1,
    data_prefix=dict(
        img_path="image",
        seg_map_path="GT"),
    indices=1,
    pipeline=test_pipeline)

#OpenForensics
OpenForensics_tamp_test_dataset=dict(
    type=dataset_type,
    data_root="/data/ipad/Forgery/test/OpenForensics/TP/",
    global_class=1,
    data_prefix=dict(
        img_path="image",
        seg_map_path="GT"),
    indices=2,
    pipeline=test_pipeline)

OpenForensics_authentic_test_dataset=dict(
    type=dataset_type,
    data_root="/data/ipad/Forgery/test/OpenForensics/AU/",
    global_class=0,
    data_prefix=dict(
        img_path="image",
        seg_map_path="GT"),
    indices=6,
    pipeline=test_pipeline)

OpenForensics_all_test_dataset=dict(
    type='ConcatDataset',
    datasets=[OpenForensics_tamp_test_dataset, OpenForensics_authentic_test_dataset])

tampIC13_tamp_test_dataset=dict(
    type=dataset_type,
    data_root="/data/ipad/Forgery/test/Tampered-IC13/",
    global_class=1,
    data_prefix=dict(
        img_path="test_img",
        seg_map_path="test_mask"),
    indices=[2,3],
    pipeline=test_pipeline)

test_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='ConcatDataset',
        datasets=[
            # OpenForensics_tamp_test_dataset,
            # Columbia_tamp_test_dataset
            # CASIAv1_plus_tamp_test_dataset,
            COVERAGE_tamp_test_dataset
            # NIST16_tamp_test_dataset,
            # tampIC13_tamp_test_dataset
        ])

)




test_evaluator=dict(type='IoUMetric', iou_metrics=['mIoU'])


