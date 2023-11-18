default_scope = 'mmseg'
env_cfg = dict(
    cudnn_benchmark=True,
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    dist_cfg=dict(backend='nccl'),
)
# vis_backends = [dict(type='LocalVisBackend')]
vis_backends = [dict(type='LocalVisBackend'),
                dict(type='TensorboardVisBackend')] #添加Tensorboard可视化
#参考
#https://mmsegmentation.readthedocs.io/zh_CN/main/user_guides/visualization.html#:~:text=%E5%9C%A8%E9%85%8D%E7%BD%AE%E6%96%87%E4%BB%B6%20default_runtime.py%20%E7%9A%84%20vis_backend%20%E4%B8%AD%E6%B7%BB%E5%8A%A0%20TensorboardVisBackend%20%E3%80%82%20vis_backends,%3D%20%5Bdict%28type%3D%27LocalVisBackend%27%29%2C%20dict%28type%3D%27TensorboardVisBackend%27%29%5D%20visualizer%20%3D%20dict%28type%3D%27SegLocalVisualizer%27%2C%20vis_backends%3Dvis_backends%2C%20name%3D%27visualizer%27%29
visualizer = dict(

    type='SegLocalVisualizer', vis_backends=vis_backends, name='visualizer')
log_processor = dict(by_epoch=False)
log_level = 'INFO'
load_from = None
resume = False

tta_model = dict(type='SegTTAModel')
