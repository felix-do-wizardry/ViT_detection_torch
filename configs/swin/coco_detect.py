# dataset settings
args = {
    'dataset_type': 'CocoDataset',
    'data_root': '/home/hai/data/coco2017/',
    'bs': 2,
    'workers': 8,
    'image_size': 224,
}
dataset_type = 'CocoDataset'

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    
    # dict(type='Resize', img_scale=(1333, 800), keep_ratio=True),
    dict(type='Resize', img_scale=(args['image_size'], args['image_size']), keep_ratio=False),
    
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        
        # img_scale=(1333, 800),
        img_scale=(args['image_size'], args['image_size']),
        
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]
data = dict(
    samples_per_gpu=args['bs'],
    workers_per_gpu=args['workers'],
    train=dict(
        type=args['dataset_type'],
        ann_file=args['data_root'] + 'annotations/instances_train2017.json',
        img_prefix=args['data_root'] + 'train2017/',
        pipeline=train_pipeline),
    val=dict(
        type=args['dataset_type'],
        ann_file=args['data_root'] + 'annotations/instances_val2017.json',
        img_prefix=args['data_root'] + 'val2017/',
        pipeline=test_pipeline),
    test=dict(
        type=args['dataset_type'],
        ann_file=args['data_root'] + 'annotations/instances_val2017.json',
        img_prefix=args['data_root'] + 'val2017/',
        pipeline=test_pipeline))

evaluation = dict(interval=1, metric='bbox')
