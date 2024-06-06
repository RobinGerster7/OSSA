train_cfg = dict(type='IterBasedTrainLoop', max_iters=70000, val_interval=5000)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

# learning rate
param_scheduler = [
    dict(
        type='MultiStepLR',
        begin=0,
        end=70000,
        by_epoch=False,
        milestones=[50000],
        gamma=0.1)
]

# optimizer
optim_wrapper = dict(
    type='AmpOptimWrapper',
    optimizer=dict(type='SGD', lr=0.001, momentum=0.9, weight_decay=0.0001))


auto_scale_lr = dict(enable=False, base_batch_size=1)

default_hooks = dict(
    checkpoint=dict(type='CheckpointHook', save_last=True),
    logger=dict(type='LoggerHook', interval=500))


vis_backends = [dict(type='LocalVisBackend')]
visualizer = dict(
    type='DetLocalVisualizer', vis_backends=vis_backends, name='visualizer')
