def test_SegDETRHead():
    from src.seg_head.seg_detr_head import SegDETRHead
    train_cfg = {'assigner': {'type': 'HungarianAssigner', 'cls_cost': {'type': 'FocalLossCost', 'weight': 2.0}, 'reg_cost': {'type': 'BBoxL1Cost', 'weight': 5.0, 'box_format': 'xywh'}, 'iou_cost': {'type': 'IoUCost', 'iou_mode': 'giou', 'weight': 2.0}}, 'assigner_with_mask': {'type': 'HungarianAssigner_multi_info', 'cls_cost': {'type': 'FocalLossCost', 'weight': 2.0}, 'reg_cost': {'type': 'BBoxL1Cost', 'weight': 5.0, 'box_format': 'xywh'}, 'iou_cost': {'type': 'IoUCost', 'iou_mode': 'giou', 'weight': 2.0}, 'mask_cost': {'type': 'DiceCost', 'weight': 2.0}}, 'sampler': {'type': 'PseudoSampler'}, 'sampler_with_mask': {'type': 'PseudoSampler_segformer'}}
    transformer = {'type': 'SegDeformableTransformer', 'encoder': {'type': 'DetrTransformerEncoder', 'num_layers': 6, 'transformerlayers': {'type': 'BaseTransformerLayer', 'attn_cfgs': {'type': 'MultiScaleDeformableAttention', 'embed_dims': 256, 'num_levels': 4}, 'feedforward_channels': 512, 'ffn_dropout': 0.1, 'operation_order': ('self_attn', 'norm', 'ffn', 'norm')}}, 'decoder': {'type': 'DeformableDetrTransformerDecoder', 'num_layers': 6, 'return_intermediate': True, 'transformerlayers': {'type': 'DetrTransformerDecoderLayer', 'attn_cfgs': [{'type': 'MultiheadAttention', 'embed_dims': 256, 'num_heads': 8, 'dropout': 0.1}, {'type': 'MultiScaleDeformableAttention', 'embed_dims': 256, 'num_levels': 4}], 'feedforward_channels': 512, 'ffn_dropout': 0.1, 'operation_order': ('self_attn', 'norm', 'cross_attn', 'norm', 'ffn', 'norm')}}}
    kwargs = {'num_query': 300, 'num_classes': 4, 'num_things_classes': 3, 'num_stuff_classes': 1, 'in_channels': 2048, 'sync_cls_avg_factor': True, 'positional_encoding': {'type': 'SinePositionalEncoding', 'num_feats': 128, 'normalize': True, 'offset': -0.5}, 'loss_cls': {'type': 'FocalLoss', 'use_sigmoid': True, 'gamma': 2.0, 'alpha': 0.25, 'loss_weight': 2.0}, 'loss_bbox': {'type': 'L1Loss', 'loss_weight': 5.0}, 'loss_iou': {'type': 'GIoULoss', 'loss_weight': 2.0}}
    head = SegDETRHead(train_cfg=train_cfg,
                transformer=transformer,
                **kwargs)
    

def test_PansegformerHead():
    args = ()
    bev_h = 200
    bev_w = 200
    canvas_size = (200,200)
    pc_range = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]
    with_box_refine = True
    as_two_stage = False
    transformer = {'type': 'SegDeformableTransformer', 'encoder': {'type': 'DetrTransformerEncoder', 'num_layers': 6, 'transformerlayers': {'type': 'BaseTransformerLayer', 'attn_cfgs': {'type': 'MultiScaleDeformableAttention', 'embed_dims': 256, 'num_levels': 4}, 'feedforward_channels': 512, 'ffn_dropout': 0.1, 'operation_order': ('self_attn', 'norm', 'ffn', 'norm')}}, 'decoder': {'type': 'DeformableDetrTransformerDecoder', 'num_layers': 6, 'return_intermediate': True, 'transformerlayers': {'type': 'DetrTransformerDecoderLayer', 'attn_cfgs': [{'type': 'MultiheadAttention', 'embed_dims': 256, 'num_heads': 8, 'dropout': 0.1}, {'type': 'MultiScaleDeformableAttention', 'embed_dims': 256, 'num_levels': 4}], 'feedforward_channels': 512, 'ffn_dropout': 0.1, 'operation_order': ('self_attn', 'norm', 'cross_attn', 'norm', 'ffn', 'norm')}}}
    thing_transformer_head = {'type': 'SegMaskHead', 'd_model': 256, 'nhead': 8, 'num_decoder_layers': 4}
    stuff_transformer_head = {'type': 'SegMaskHead', 'd_model': 256, 'nhead': 8, 'num_decoder_layers': 6, 'self_attn': True}
    loss_mask = {'type': 'DiceLoss', 'loss_weight': 2.0}
    train_cfg = {'assigner': {'type': 'HungarianAssigner', 'cls_cost': {'type': 'FocalLossCost', 'weight': 2.0}, 'reg_cost': {'type': 'BBoxL1Cost', 'weight': 5.0, 'box_format': 'xywh'}, 'iou_cost': {'type': 'IoUCost', 'iou_mode': 'giou', 'weight': 2.0}}, 'assigner_with_mask': {'type': 'HungarianAssigner_multi_info', 'cls_cost': {'type': 'FocalLossCost', 'weight': 2.0}, 'reg_cost': {'type': 'BBoxL1Cost', 'weight': 5.0, 'box_format': 'xywh'}, 'iou_cost': {'type': 'IoUCost', 'iou_mode': 'giou', 'weight': 2.0}, 'mask_cost': {'type': 'DiceCost', 'weight': 2.0}}, 'sampler': {'type': 'PseudoSampler'}, 'sampler_with_mask': {'type': 'PseudoSampler_segformer'}}
    kwargs = {'num_query': 300, 'num_classes': 4, 'num_things_classes': 3, 'num_stuff_classes': 1, 'in_channels': 2048, 'sync_cls_avg_factor': True, 'positional_encoding': {'type': 'SinePositionalEncoding', 'num_feats': 128, 'normalize': True, 'offset': -0.5}, 'loss_cls': {'type': 'FocalLoss', 'use_sigmoid': True, 'gamma': 2.0, 'alpha': 0.25, 'loss_weight': 2.0}, 'loss_bbox': {'type': 'L1Loss', 'loss_weight': 5.0}, 'loss_iou': {'type': 'GIoULoss', 'loss_weight': 2.0}}
    from src.seg_head.panseg_head import PansegformerHead
    head = PansegformerHead(
        args,
        bev_h=bev_h,
        bev_w=bev_w,
        canvas_size=canvas_size,
        pc_range=pc_range,
        with_box_refine=with_box_refine, 
        as_two_stage=as_two_stage,
        transformer=transformer,
        thing_transformer_head=thing_transformer_head,
        stuff_transformer_head=stuff_transformer_head,
        loss_mask=loss_mask,
        train_cfg=train_cfg,
        kwargs=kwargs
    )
    
def test_BEVFormerEncoder():
    from src.bevformer.custom_encoder import BEVFormerEncoder
    pc_range = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]
    encoder = BEVFormerEncoder(pc_range=pc_range,num_points_in_pillar=4)
    return encoder