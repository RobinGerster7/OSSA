default_scope = 'mmdet'

_base_ = [
    '../datasets/sim2city.py',
    '../models/faster-rcnn_r50_fpn.py',
    '../schedules/schedule.py'
]

model = dict(
    roi_head=dict(
        bbox_head=dict(
            num_classes=1,
        )
    )
)
