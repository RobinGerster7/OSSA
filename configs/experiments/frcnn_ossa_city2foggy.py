default_scope = 'mmdet'

_base_ = [
    '../datasets/cityscapes2foggy.py',
    '../models/faster-rcnn_r50_fpn.py',
    '../schedules/schedule.py'
]

custom_hooks = [
    dict(type='OSSAPrototypeHook', priority='NORMAL')
]

model = dict(
    backbone=dict(
        type='ResNetOSSA',
        aug_stages = [0,1],
        aug_prob = 0.5,
        aug_intensity = 0.75,
        make_prototypes = False
    ),
)

