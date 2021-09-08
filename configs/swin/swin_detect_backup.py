
# Currently HARDCODED
args = dict(
    image_size=224,
    # image_size=384,
    bs=4,
    workers=6,
    classes=['person'],
    dataset='coco',
    dataset_type='CocoDataset',
    task='detection',
    target='_person',
    # data_root='/host/ubuntu/torch/coco/',
    data_root='/home/hai/data/coco2017',
    swin_name='swin_large_patch4_window7_224_22k',
    swin_channels=1536,
    swin_num_out=4,
    pretrained=True,
    lr=0.003,
    optimizer='SGD',
)

args['num_class'] = len(args['classes'])

model = dict(
    train_cfg=dict(
        rpn=dict(
            assigner=dict(
                type='MaxIoUAssigner',
                pos_iou_thr=0.7,
                neg_iou_thr=0.3,
                min_pos_iou=0.3,
                match_low_quality=True,
                ignore_iof_thr=-1),
            sampler=dict(
                type='RandomSampler',
                num=256,
                pos_fraction=0.5,
                neg_pos_ub=-1,
                add_gt_as_proposals=False),
            allowed_border=-1,
            pos_weight=-1,
            debug=False),
        rpn_proposal=dict(
            nms_pre=2000,
            max_per_img=1000,
            nms=dict(type='nms', iou_threshold=0.7),
            min_bbox_size=0),
        rcnn=dict(
            assigner=dict(
                type='MaxIoUAssigner',
                pos_iou_thr=0.5,
                neg_iou_thr=0.5,
                min_pos_iou=0.5,
                match_low_quality=False,
                ignore_iof_thr=-1),
            sampler=dict(
                type='RandomSampler',
                num=512,
                pos_fraction=0.25,
                neg_pos_ub=-1,
                add_gt_as_proposals=True),
            pos_weight=-1,
            debug=False)),
    test_cfg=dict(
        rpn=dict(
            nms_pre=1000,
            max_per_img=1000,
            nms=dict(type='nms', iou_threshold=0.7),
            min_bbox_size=0),
        rcnn=dict(
            score_thr=0.05,
            nms=dict(type='nms', iou_threshold=0.5),
            max_per_img=100)),
    type='FasterRCNN',
    backbone=dict(
        # type='SwinTransformerFPNWrap',
        # name=args['swin_name'],
        type='SwinTransformerFPN',
        model_name=args['swin_name'],
        device='cuda',
        # device='cpu',
    ),
    neck=dict(
        type='FPN',
        in_channels=[
            int(args['swin_channels'] / (2 ** i))
            for i in range(args['swin_num_out'])
        ][::-1],
        out_channels=256,
        num_outs=args['swin_num_out']),
    rpn_head=dict(
        type='RPNHead',
        in_channels=256,
        feat_channels=256,
        anchor_generator=dict(
            type='AnchorGenerator',
            scales=[8],
            ratios=[0.5, 1.0, 2.0],
            strides=[4, 8, 16, 32]),
        bbox_coder=dict(
            type='DeltaXYWHBBoxCoder',
            target_means=[0.0, 0.0, 0.0, 0.0],
            target_stds=[1.0, 1.0, 1.0, 1.0]),
        loss_cls=dict(
            type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0),
        loss_bbox=dict(type='L1Loss', loss_weight=1.0)),
    roi_head=dict(
        type='StandardRoIHead',
        bbox_roi_extractor=dict(
            type='SingleRoIExtractor',
            roi_layer=dict(
                type='RoIAlign',
                output_size=7,
                sampling_ratio=0,
                use_torchvision=True),
            out_channels=256,
            featmap_strides=[4],
            finest_scale=56),
        bbox_head=dict(
            type='Shared2FCBBoxHead',
            in_channels=256,
            fc_out_channels=1024,
            roi_feat_size=7,
            num_classes=args['num_class'],
            bbox_coder=dict(
                type='DeltaXYWHBBoxCoder',
                target_means=[0.0, 0.0, 0.0, 0.0],
                target_stds=[0.1, 0.1, 0.2, 0.2]),
            reg_class_agnostic=False,
            loss_cls=dict(
                type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
            loss_bbox=dict(type='L1Loss', loss_weight=1.0))))

optimizer = dict(
    type=args['optimizer'],
    lr=args['lr'],
    momentum=0.9,
    weight_decay=0.0001,
)
optimizer_config = dict(grad_clip=None)
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.001,
    step=[8, 11],
)

runner = dict(type='EpochBasedRunner', max_epochs=12)
checkpoint_config = dict(interval=1)
log_config = dict(interval=50, hooks=[dict(type='TextLoggerHook')])
custom_hooks = [dict(type='NumClassCheckHook')]
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)]

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    to_rgb=True,
)

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', img_scale=(args['image_size'], args['image_size']), keep_ratio=False),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(
        type='Normalize',
        **img_norm_cfg,
    ),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'])
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(args['image_size'], args['image_size']),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=False),
            dict(type='RandomFlip'),
            dict(
                type='Normalize',
                **img_norm_cfg,
            ),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img'])
        ])
]

data = dict(
    samples_per_gpu=args['bs'],
    workers_per_gpu=args['workers'],
    train=dict(
        type=args['dataset_type'],
        ann_file=args['data_root'] + '/annotations/instances_train2017.json',
        img_prefix=args['data_root'] + '/train2017/',
        pipeline=train_pipeline,
        classes=args['classes']),
    val=dict(
        type=args['dataset_type'],
        ann_file=args['data_root'] + '/annotations/instances_val2017.json',
        img_prefix=args['data_root'] + '/val2017/',
        pipeline=test_pipeline,
        classes=args['classes']),
    test=dict(
        type=args['dataset_type'],
        ann_file=args['data_root'] + '/annotations/instances_val2017.json',
        img_prefix=args['data_root'] + '/val2017/',
        pipeline=test_pipeline,
        classes=args['classes']))
evaluation = dict(interval=1, metric='bbox')

work_dir = '/home/hai/mmdet'
# gpu_ids = range(0, 1)
gpu_ids = [0]
# gpu_ids = [1]
