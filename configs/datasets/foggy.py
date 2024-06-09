# dataset settings
dataset_type = 'CocoDataset'
data_root = './datasets/foggy_cityscapes/'

batch_size = 1
num_workers = 2
metainfo = {
    'classes': ('person', 'rider', 'car', 'truck', 'bus', 'train',
                'motorcycle', 'bicycle'),
    'palette': [(220, 20, 60), (255, 0, 0), (0, 0, 142), (0, 0, 70),
                (0, 60, 100), (0, 80, 100), (0, 0, 230), (119, 11, 32)]
}
backend_args = None

train_pipeline = [
    dict(type='LoadImageFromFile', backend_args=backend_args),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', scale=(1200, 600), keep_ratio=True),
    dict(type='PackDetInputs')
]
test_pipeline = [
    dict(type='LoadImageFromFile', backend_args=backend_args),
    dict(type='Resize', scale=(1200, 600), keep_ratio=True),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor'))
]

train_dataloader = dict(
    batch_size=batch_size,
    num_workers=num_workers,
    persistent_workers=True,
    sampler=dict(type='InfiniteSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        metainfo=metainfo,
        ann_file='annotations/instancesonly_filtered_gtFine_train.json',
        data_prefix=dict(img='leftImg8bit/train/'),
        filter_cfg=dict(filter_empty_gt=True, min_size=32),
        pipeline=train_pipeline,
        backend_args=backend_args))

val_dataloader = dict(
    batch_size=batch_size,
    num_workers=num_workers,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        metainfo=metainfo,
        ann_file='annotations/instancesonly_filtered_gtFine_val.json',
        data_prefix=dict(img='leftImg8bit/val/'),
        test_mode=True,
        pipeline=test_pipeline,
        backend_args=backend_args))

val_evaluator = dict(
    type='CocoMetric',
    ann_file=data_root + 'annotations/instancesonly_filtered_gtFine_val.json',
    metric='bbox',
    format_only=False,
    backend_args=backend_args)

test_dataloader = val_dataloader
test_evaluator = val_evaluator