_base_ = [
    '../_base_/models/cascade_mask_rcnn_swin_fpn.py',
    '../_base_/datasets/coco_instance.py',
    # '../_base_/schedules/schedule_1x.py',
    # '../_base_/default_runtime.py',
]

_args = {
    'classes': ['person'],
    'num_classes': None,
    'bs': 1,
    'workers': 4,
}
_args['num_classes'] = len(_args['classes'])


model = dict(
    backbone=dict(
        embed_dim=96,
        depths=[2, 2, 6, 2],
        num_heads=[3, 6, 12, 24],
        window_size=7,
        ape=False,
        drop_path_rate=0.0,
        patch_norm=True,
        use_checkpoint=False
    ),
    neck=dict(in_channels=[96, 192, 384, 768]),
    roi_head=dict(
        bbox_head=[
            dict(
                type='ConvFCBBoxHead',
                num_shared_convs=4,
                num_shared_fcs=1,
                in_channels=256,
                conv_out_channels=256,
                fc_out_channels=1024,
                roi_feat_size=7,
                num_classes=_args['num_classes'],
                bbox_coder=dict(
                    type='DeltaXYWHBBoxCoder',
                    target_means=[0., 0., 0., 0.],
                    target_stds=[0.1, 0.1, 0.2, 0.2]),
                reg_class_agnostic=False,
                reg_decoded_bbox=True,
                norm_cfg=dict(type='BN', requires_grad=True),
                loss_cls=dict(
                    type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
                loss_bbox=dict(type='GIoULoss', loss_weight=10.0)),
            dict(
                type='ConvFCBBoxHead',
                num_shared_convs=4,
                num_shared_fcs=1,
                in_channels=256,
                conv_out_channels=256,
                fc_out_channels=1024,
                roi_feat_size=7,
                num_classes=_args['num_classes'],
                bbox_coder=dict(
                    type='DeltaXYWHBBoxCoder',
                    target_means=[0., 0., 0., 0.],
                    target_stds=[0.05, 0.05, 0.1, 0.1]),
                reg_class_agnostic=False,
                reg_decoded_bbox=True,
                norm_cfg=dict(type='BN', requires_grad=True),
                loss_cls=dict(
                    type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
                loss_bbox=dict(type='GIoULoss', loss_weight=10.0)),
            dict(
                type='ConvFCBBoxHead',
                num_shared_convs=4,
                num_shared_fcs=1,
                in_channels=256,
                conv_out_channels=256,
                fc_out_channels=1024,
                roi_feat_size=7,
                num_classes=_args['num_classes'],
                bbox_coder=dict(
                    type='DeltaXYWHBBoxCoder',
                    target_means=[0., 0., 0., 0.],
                    target_stds=[0.033, 0.033, 0.067, 0.067]),
                reg_class_agnostic=False,
                reg_decoded_bbox=True,
                norm_cfg=dict(type='BN', requires_grad=True),
                loss_cls=dict(
                    type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
                loss_bbox=dict(type='GIoULoss', loss_weight=10.0))
        ]))


# model = dict(
#     type='CascadeRCNN',
#     pretrained=None,
#     backbone=dict(
#         type='SwinTransformer',
#         embed_dim=128,
#         depths=[2, 2, 18, 2],
#         num_heads=[4, 8, 16, 32],
#         window_size=7,
#         mlp_ratio=4.,
#         qkv_bias=True,
#         qk_scale=None,
#         drop_rate=0.,
#         attn_drop_rate=0.,
#         drop_path_rate=0.3,
#         ape=False,
#         patch_norm=True,
#         out_indices=(0, 1, 2, 3),
#         use_checkpoint=False
#     ),
#     neck=dict(
#         type='FPN',
#         # in_channels=[96, 192, 384, 768],
#         in_channels=[128, 256, 512, 1024],
#         out_channels=256,
#         num_outs=5),
#     rpn_head=dict(
#         type='RPNHead',
#         in_channels=256,
#         feat_channels=256,
#         anchor_generator=dict(
#             type='AnchorGenerator',
#             scales=[8],
#             ratios=[0.5, 1.0, 2.0],
#             strides=[4, 8, 16, 32, 64]),
#         bbox_coder=dict(
#             type='DeltaXYWHBBoxCoder',
#             target_means=[.0, .0, .0, .0],
#             target_stds=[1.0, 1.0, 1.0, 1.0]),
#         loss_cls=dict(
#             type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0),
#         loss_bbox=dict(type='SmoothL1Loss', beta=1.0 / 9.0, loss_weight=1.0)),
#     roi_head=dict(
#         type='CascadeRoIHead',
#         num_stages=3,
#         stage_loss_weights=[1, 0.5, 0.25],
#         bbox_roi_extractor=dict(
#             type='SingleRoIExtractor',
#             roi_layer=dict(type='RoIAlign', output_size=7, sampling_ratio=0),
#             out_channels=256,
#             featmap_strides=[4, 8, 16, 32]),
#         bbox_head=[
#             dict(
#                 type='ConvFCBBoxHead',
#                 num_shared_convs=4,
#                 num_shared_fcs=1,
#                 in_channels=256,
#                 conv_out_channels=256,
#                 fc_out_channels=1024,
#                 roi_feat_size=7,
#                 num_classes=_args['num_classes'],
#                 bbox_coder=dict(
#                     type='DeltaXYWHBBoxCoder',
#                     target_means=[0., 0., 0., 0.],
#                     target_stds=[0.1, 0.1, 0.2, 0.2]),
#                 reg_class_agnostic=False,
#                 reg_decoded_bbox=True,
#                 norm_cfg=dict(type='BN', requires_grad=True),
#                 loss_cls=dict(
#                     type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
#                 loss_bbox=dict(type='GIoULoss', loss_weight=10.0)),
#             dict(
#                 type='ConvFCBBoxHead',
#                 num_shared_convs=4,
#                 num_shared_fcs=1,
#                 in_channels=256,
#                 conv_out_channels=256,
#                 fc_out_channels=1024,
#                 roi_feat_size=7,
#                 num_classes=_args['num_classes'],
#                 bbox_coder=dict(
#                     type='DeltaXYWHBBoxCoder',
#                     target_means=[0., 0., 0., 0.],
#                     target_stds=[0.05, 0.05, 0.1, 0.1]),
#                 reg_class_agnostic=False,
#                 reg_decoded_bbox=True,
#                 norm_cfg=dict(type='BN', requires_grad=True),
#                 loss_cls=dict(
#                     type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
#                 loss_bbox=dict(type='GIoULoss', loss_weight=10.0)),
#             dict(
#                 type='ConvFCBBoxHead',
#                 num_shared_convs=4,
#                 num_shared_fcs=1,
#                 in_channels=256,
#                 conv_out_channels=256,
#                 fc_out_channels=1024,
#                 roi_feat_size=7,
#                 num_classes=_args['num_classes'],
#                 bbox_coder=dict(
#                     type='DeltaXYWHBBoxCoder',
#                     target_means=[0., 0., 0., 0.],
#                     target_stds=[0.033, 0.033, 0.067, 0.067]),
#                 reg_class_agnostic=False,
#                 reg_decoded_bbox=True,
#                 norm_cfg=dict(type='BN', requires_grad=True),
#                 loss_cls=dict(
#                     type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
#                 loss_bbox=dict(type='GIoULoss', loss_weight=10.0))
#         ],
#         mask_roi_extractor=dict(
#             type='SingleRoIExtractor',
#             roi_layer=dict(type='RoIAlign', output_size=14, sampling_ratio=0),
#             out_channels=256,
#             featmap_strides=[4, 8, 16, 32]),
#         mask_head=dict(
#             type='FCNMaskHead',
#             num_convs=4,
#             in_channels=256,
#             conv_out_channels=256,
#             num_classes=_args['num_classes'],
#             loss_mask=dict(
#                 type='CrossEntropyLoss', use_mask=True, loss_weight=1.0))),
#     # model training and testing settings
#     train_cfg = dict(
#         rpn=dict(
#             assigner=dict(
#                 type='MaxIoUAssigner',
#                 pos_iou_thr=0.7,
#                 neg_iou_thr=0.3,
#                 min_pos_iou=0.3,
#                 match_low_quality=True,
#                 ignore_iof_thr=-1),
#             sampler=dict(
#                 type='RandomSampler',
#                 num=256,
#                 pos_fraction=0.5,
#                 neg_pos_ub=-1,
#                 add_gt_as_proposals=False),
#             allowed_border=0,
#             pos_weight=-1,
#             debug=False),
#         rpn_proposal=dict(
#             nms_across_levels=False,
#             nms_pre=2000,
#             nms_post=2000,
#             max_per_img=2000,
#             nms=dict(type='nms', iou_threshold=0.7),
#             min_bbox_size=0),
#         rcnn=[
#             dict(
#                 assigner=dict(
#                     type='MaxIoUAssigner',
#                     pos_iou_thr=0.5,
#                     neg_iou_thr=0.5,
#                     min_pos_iou=0.5,
#                     match_low_quality=False,
#                     ignore_iof_thr=-1),
#                 sampler=dict(
#                     type='RandomSampler',
#                     num=512,
#                     pos_fraction=0.25,
#                     neg_pos_ub=-1,
#                     add_gt_as_proposals=True),
#                 mask_size=28,
#                 pos_weight=-1,
#                 debug=False),
#             dict(
#                 assigner=dict(
#                     type='MaxIoUAssigner',
#                     pos_iou_thr=0.6,
#                     neg_iou_thr=0.6,
#                     min_pos_iou=0.6,
#                     match_low_quality=False,
#                     ignore_iof_thr=-1),
#                 sampler=dict(
#                     type='RandomSampler',
#                     num=512,
#                     pos_fraction=0.25,
#                     neg_pos_ub=-1,
#                     add_gt_as_proposals=True),
#                 mask_size=28,
#                 pos_weight=-1,
#                 debug=False),
#             dict(
#                 assigner=dict(
#                     type='MaxIoUAssigner',
#                     pos_iou_thr=0.7,
#                     neg_iou_thr=0.7,
#                     min_pos_iou=0.7,
#                     match_low_quality=False,
#                     ignore_iof_thr=-1),
#                 sampler=dict(
#                     type='RandomSampler',
#                     num=512,
#                     pos_fraction=0.25,
#                     neg_pos_ub=-1,
#                     add_gt_as_proposals=True),
#                 mask_size=28,
#                 pos_weight=-1,
#                 debug=False)
#         ]),
#     test_cfg = dict(
#         rpn=dict(
#             nms_across_levels=False,
#             nms_pre=1000,
#             nms_post=1000,
#             max_per_img=1000,
#             nms=dict(type='nms', iou_threshold=0.7),
#             min_bbox_size=0),
#         rcnn=dict(
#             score_thr=0.05,
#             nms=dict(type='nms', iou_threshold=0.5),
#             max_per_img=100,
#             mask_thr_binary=0.5)))

# %% MODEL
# model = dict(
#     backbone=dict(
#         embed_dim=128,
#         depths=[2, 2, 18, 2],
#         num_heads=[4, 8, 16, 32],
#         window_size=7,
#         ape=False,
#         drop_path_rate=0.3,
#         patch_norm=True,
#         use_checkpoint=False
#     ),
#     neck=dict(in_channels=[128, 256, 512, 1024]),
#     roi_head=dict(
#         bbox_head=[
#             dict(
#                 type='ConvFCBBoxHead',
#                 num_shared_convs=4,
#                 num_shared_fcs=1,
#                 in_channels=256,
#                 conv_out_channels=256,
#                 fc_out_channels=1024,
#                 roi_feat_size=7,
#                 num_classes=80,
#                 bbox_coder=dict(
#                     type='DeltaXYWHBBoxCoder',
#                     target_means=[0., 0., 0., 0.],
#                     target_stds=[0.1, 0.1, 0.2, 0.2]),
#                 reg_class_agnostic=False,
#                 reg_decoded_bbox=True,
#                 norm_cfg=dict(type='BN', requires_grad=True),
#                 loss_cls=dict(
#                     type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
#                 loss_bbox=dict(type='GIoULoss', loss_weight=10.0)),
#             dict(
#                 type='ConvFCBBoxHead',
#                 num_shared_convs=4,
#                 num_shared_fcs=1,
#                 in_channels=256,
#                 conv_out_channels=256,
#                 fc_out_channels=1024,
#                 roi_feat_size=7,
#                 num_classes=80,
#                 bbox_coder=dict(
#                     type='DeltaXYWHBBoxCoder',
#                     target_means=[0., 0., 0., 0.],
#                     target_stds=[0.05, 0.05, 0.1, 0.1]),
#                 reg_class_agnostic=False,
#                 reg_decoded_bbox=True,
#                 norm_cfg=dict(type='BN', requires_grad=True),
#                 loss_cls=dict(
#                     type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
#                 loss_bbox=dict(type='GIoULoss', loss_weight=10.0)),
#             dict(
#                 type='ConvFCBBoxHead',
#                 num_shared_convs=4,
#                 num_shared_fcs=1,
#                 in_channels=256,
#                 conv_out_channels=256,
#                 fc_out_channels=1024,
#                 roi_feat_size=7,
#                 num_classes=80,
#                 bbox_coder=dict(
#                     type='DeltaXYWHBBoxCoder',
#                     target_means=[0., 0., 0., 0.],
#                     target_stds=[0.033, 0.033, 0.067, 0.067]),
#                 reg_class_agnostic=False,
#                 reg_decoded_bbox=True,
#                 norm_cfg=dict(type='BN', requires_grad=True),
#                 loss_cls=dict(
#                     type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
#                 loss_bbox=dict(type='GIoULoss', loss_weight=10.0))
#         ]))

# %% DATASET
# merged
dataset_type = 'CocoDataset'
data_root = '/host/ubuntu/torch/coco/'

# img_norm_cfg = dict(
#     mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

# augmentation strategy originates from DETR / Sparse RCNN
# train_pipeline = [
#     dict(type='LoadImageFromFile'),
#     dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
#     dict(type='RandomFlip', flip_ratio=0.5),
#     dict(type='AutoAugment',
#          policies=[
#              [
#                  dict(type='Resize',
#                       img_scale=[(480, 1333), (512, 1333), (544, 1333), (576, 1333),
#                                  (608, 1333), (640, 1333), (672, 1333), (704, 1333),
#                                  (736, 1333), (768, 1333), (800, 1333)],
#                       multiscale_mode='value',
#                       keep_ratio=True)
#              ],
#              [
#                  dict(type='Resize',
#                       img_scale=[(400, 1333), (500, 1333), (600, 1333)],
#                       multiscale_mode='value',
#                       keep_ratio=True),
#                  dict(type='RandomCrop',
#                       crop_type='absolute_range',
#                       crop_size=(384, 600),
#                       allow_negative_crop=True),
#                  dict(type='Resize',
#                       img_scale=[(480, 1333), (512, 1333), (544, 1333),
#                                  (576, 1333), (608, 1333), (640, 1333),
#                                  (672, 1333), (704, 1333), (736, 1333),
#                                  (768, 1333), (800, 1333)],
#                       multiscale_mode='value',
#                       override=True,
#                       keep_ratio=True)
#              ]
#          ]),
#     dict(type='Normalize', **img_norm_cfg),
#     dict(type='Pad', size_divisor=32),
#     dict(type='DefaultFormatBundle'),
#     dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels', 'gt_masks']),
# ]

# test_pipeline = [
#     dict(type='LoadImageFromFile'),
#     dict(
#         type='MultiScaleFlipAug',
#         img_scale=(1333, 800),
#         flip=False,
#         transforms=[
#             dict(type='Resize', keep_ratio=True),
#             dict(type='RandomFlip'),
#             dict(type='Normalize', **img_norm_cfg),
#             dict(type='Pad', size_divisor=32),
#             dict(type='ImageToTensor', keys=['img']),
#             dict(type='Collect', keys=['img']),
#         ])
# ]
# evaluation = dict(metric=['bbox', 'segm'])

# _data_configs = {
#     k: {
#         'type': 'CocoDataset',
#         'ann_file': data_root + 'annotations/instances_{}2017.json'.format(data_name),
#         'img_prefix': data_root + '{}2017/'.format(data_name),
#         'pipeline': _pipeline,
#         'classes': _args['classes'],
#     }
#     for k, data_name, _pipeline in zip(
#         ['train', 'val', 'test'],
#         ['train', 'val', 'val'],
#         [train_pipeline, test_pipeline, test_pipeline],
#     )
# }

# data = dict(
#     samples_per_gpu=_args['bs'],
#     workers_per_gpu=_args['workers'],
#     **_data_configs,
#     # train=dict(
#     #     type=dataset_type,
#     #     ann_file=data_root + 'annotations/instances_train2017.json',
#     #     img_prefix=data_root + 'train2017/',
#     #     pipeline=train_pipeline),
#     # val=dict(
#     #     type=dataset_type,
#     #     ann_file=data_root + 'annotations/instances_val2017.json',
#     #     img_prefix=data_root + 'val2017/',
#     #     pipeline=test_pipeline),
#     # test=dict(
#     #     type=dataset_type,
#     #     ann_file=data_root + 'annotations/instances_val2017.json',
#     #     img_prefix=data_root + 'val2017/',
#     #     pipeline=test_pipeline)
# )

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

# augmentation strategy originates from DETR / Sparse RCNN
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='AutoAugment',
         policies=[
             [
                 dict(type='Resize',
                      img_scale=[(480, 1333), (512, 1333), (544, 1333), (576, 1333),
                                 (608, 1333), (640, 1333), (672, 1333), (704, 1333),
                                 (736, 1333), (768, 1333), (800, 1333)],
                      multiscale_mode='value',
                      keep_ratio=True)
             ],
             [
                 dict(type='Resize',
                      img_scale=[(400, 1333), (500, 1333), (600, 1333)],
                      multiscale_mode='value',
                      keep_ratio=True),
                 dict(type='RandomCrop',
                      crop_type='absolute_range',
                      crop_size=(384, 600),
                      allow_negative_crop=True),
                 dict(type='Resize',
                      img_scale=[(480, 1333), (512, 1333), (544, 1333),
                                 (576, 1333), (608, 1333), (640, 1333),
                                 (672, 1333), (704, 1333), (736, 1333),
                                 (768, 1333), (800, 1333)],
                      multiscale_mode='value',
                      override=True,
                      keep_ratio=True)
             ]
         ]),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels', 'gt_masks']),
]

data = dict(
    train=dict(
        pipeline=train_pipeline,
        classes=_args['classes'],
    ),
    val=dict(
        classes=_args['classes'],
    ),
    test=dict(
        classes=_args['classes'],
    ),
)


# %% SCHEDULE

# schedule_1x copied over
# optimizer
# optimizer = dict(type='SGD', lr=0.02, momentum=0.9, weight_decay=0.0001)
# optimizer_config = dict(grad_clip=None)
# learning policy
# lr_config = dict(
#     policy='step',
#     warmup='linear',
#     warmup_iters=500,
#     warmup_ratio=0.001,
#     step=[8, 11])
# runner = dict(type='EpochBasedRunner', max_epochs=12)

# this schedule
optimizer = dict(
    # _delete_=True,
    type='AdamW',
    # momentum=0.9,
    lr=0.0001,
    betas=(0.9, 0.999),
    weight_decay=0.05,
    paramwise_cfg=dict(custom_keys={'absolute_pos_embed': dict(decay_mult=0.),
                                    'relative_position_bias_table': dict(decay_mult=0.),
                                    'norm': dict(decay_mult=0.)}),
)
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.001,
    step=[27, 33],
)
runner = dict(type='EpochBasedRunnerAmp', max_epochs=36)

# do not use mmdet version fp16
fp16 = None
optimizer_config = dict(
    type="DistOptimizerHook",
    update_interval=1,
    grad_clip=None,
    coalesce=True,
    bucket_size_mb=-1,
    use_fp16=True,
)


# %% RUNTIME

# default_runtime
checkpoint_config = dict(interval=1)
# yapf:disable
log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        # dict(type='TensorboardLoggerHook')
    ])
# yapf:enable
custom_hooks = [dict(type='NumClassCheckHook')]
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)]


