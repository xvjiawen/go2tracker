_base_ = ('yolov10_s_syncbn_fast_8xb16-500e_coco.py')
custom_imports = dict(imports=['yolo_world'], allow_failed_imports=False)

# hyper-parameters
num_classes = 1203
num_training_classes = 80
max_epochs = 100  # Maximum training epochs
close_mosaic_epochs = 2
save_epoch_intervals = 1
text_channels = 512
neck_embed_channels = [128, 256, _base_.last_stage_out_channels // 2]
neck_num_heads = [4, 8, _base_.last_stage_out_channels // 2 // 32]
base_lr = 5e-4
weight_decay = 0.05 / 2
train_batch_size_per_gpu = 16

# model settings
infer_type = "one2one"

model = dict(
    type='YOLOWorldDetector',
    mm_neck=False,
    num_train_classes=num_training_classes,
    num_test_classes=num_classes,
    data_preprocessor=dict(type='YOLOWDetDataPreprocessor'),
    backbone=dict(
        _delete_=True,
        type='MultiModalYOLOBackbone',
        image_model={{_base_.model.backbone}},
        text_model=dict(
            type='HuggingCLIPLanguageBackbone',
            model_name='models/clip-vit-base-patch32',
            training_use_lora=True),
    ),
    bbox_head=dict(type='YOLOv10WorldHead',
                   infer_type=infer_type,
                   head_module=dict(type='YOLOv10WorldHeadModule',
                                    use_bn_head=True,
                                    embed_dims=text_channels,
                                    num_classes=num_training_classes)),
    train_cfg=dict(
        one2many_assigner=dict(num_classes=num_training_classes),
        one2one_assigner=dict(num_classes=num_training_classes),
    ))

test_pipeline = [
    dict(type='LoadImageFromNDArray', backend_args=None),
    dict(type='YOLOv5KeepRatioResize', scale=(640, 640)),
    dict(
        type='LetterResize',
        scale=(640, 640),
        allow_scale_up=False,
        pad_val=dict(img=114)),
    dict(type='LoadAnnotations', with_bbox=True, _scope_='mmdet'),
    # dict(type='LoadText'),
    dict(type='mmdet.PackDetInputs',
         meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                    'scale_factor', 'pad_param',
                    # 'texts'
                    ))
]

coco_val_dataset = dict(
    _delete_=True,
    type='MultiModalDataset',
    dataset=dict(type='YOLOv5LVISV1Dataset',
                 data_root='configs/',
                 test_mode=True,
                 ann_file='lvis_v1_minival_inserted_image_name.json',
                 data_prefix=dict(img=''),
                 batch_shapes_cfg=None),
    class_text_path='configs/lvis_v1_class_texts.json',
    pipeline=test_pipeline)
val_dataloader = dict(dataset=coco_val_dataset)
test_dataloader = val_dataloader



