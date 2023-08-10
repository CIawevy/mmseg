# dataset settings

dataset_type = 'ForensicsDataset'
data_root = '/data/ipad/Forgery/'
crop_size = (512, 512)
#crop_size = (352, 352)


test_pipeline = [
    dict(type='LoadImageFromFile'),
    #dict(type='Resize', scale=crop_size),
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
    indices=5,
    pipeline=test_pipeline)

#OpenForensics
OpenForensics_tamp_test_dataset=dict(
    type=dataset_type,
    data_root="/data/ipad/Forgery/test/OpenForensics/TP/",
    global_class=1,
    data_prefix=dict(
        img_path="image",
        seg_map_path="GT"),
    indices=1400,
    pipeline=test_pipeline)

OpenForensics_authentic_test_dataset=dict(
    type=dataset_type,
    data_root="/data/ipad/Forgery/test/OpenForensics/AU/",
    global_class=0,
    data_prefix=dict(
        img_path="image",
        seg_map_path="GT"),
    indices=600,
    pipeline=test_pipeline)

OpenForensics_all_test_dataset=dict(
    type='ConcatDataset',
    datasets=[OpenForensics_tamp_test_dataset, OpenForensics_authentic_test_dataset])



test_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='ConcatDataset',
        datasets=[
            # OpenForensics_all_test_dataset,
            Columbia_tamp_test_dataset
            ])

)




test_evaluator=dict(type='IoUMetric', iou_metrics=['mIoU'])


