default_scope = 'mmdet'

_base_ = [
    '../datasets/cityscapes.py',
    '../models/faster-rcnn_r50_fpn.py',
    '../schedules/schedule.py'
]

custom_hooks = [
    dict(type='OSSAPrototypeHook', priority='NORMAL')
]

train_cfg = dict(type='IterBasedTrainLoop', max_iters=1, val_interval=99999)

model = dict(
    backbone=dict(
        type='ResNetOSSA',
        frozen_stages=4,
        make_prototypes = True,
    ),
)


