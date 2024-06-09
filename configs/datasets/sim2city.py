# dataset settings
dataset_type = 'CocoDataset'
data_root = './datasets/sim10k/'

batch_size = 1
num_workers = 2
metainfo = {
    'classes': ('car'),
    'palette': [(0, 0, 255)]
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
        ann_file='annotations.coco.json',
        data_prefix=dict(img='JPEGImages/'),
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
        ann_file='../cityscapes/annotations/instancesonly_filtered_gtFine_val.json',
        data_prefix=dict(img='../cityscapes/leftImg8bit/val/'),
        test_mode=True,
        pipeline=test_pipeline,
        backend_args=backend_args))


shot_dataloader = dict(
    batch_size=batch_size,
    num_workers=num_workers,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        metainfo=metainfo,
        ann_file='../cityscapes/annotations/instancesonly_filtered_gtFine_train.json',
        data_prefix=dict(img='../cityscapes/leftImg8bit/train/'),
        test_mode=True,
        pipeline=test_pipeline,
        backend_args=backend_args))


val_evaluator = dict(
    type='CocoMetric',
    ann_file=data_root + '../cityscapes/annotations/instancesonly_filtered_gtFine_val.json',
    metric='bbox',
    format_only=False,
    backend_args=backend_args)

test_dataloader = dict(
    batch_size=batch_size,
    num_workers=num_workers,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        metainfo=metainfo,
        ann_file='../cityscapes/annotations/instancesonly_filtered_gtFine_val.json',
        data_prefix=dict(img='../cityscapes/leftImg8bit/val'),
        test_mode=True,
        pipeline=test_pipeline,
        backend_args=backend_args))

test_evaluator = dict(
    type='CocoMetric',
    ann_file=data_root + '../cityscapes/annotations/instancesonly_filtered_gtFine_val.json',
    metric='bbox',
    format_only=False,
    backend_args=backend_args)

