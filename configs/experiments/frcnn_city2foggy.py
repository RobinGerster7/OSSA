default_scope = 'mmdet'

_base_ = [
    '../datasets/cityscapes2foggy.py',
    '../models/faster-rcnn_r50_fpn.py',
    '../schedules/schedule.py'
]
