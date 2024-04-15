def test_SegDETRHead():
    from src.seg_head.seg_detr_head import SegDETRHead
    train_cfg = {'assigner': {'type': 'HungarianAssigner', 'cls_cost': {'type': 'FocalLossCost', 'weight': 2.0}, 'reg_cost': {'type': 'BBoxL1Cost', 'weight': 5.0, 'box_format': 'xywh'}, 'iou_cost': {'type': 'IoUCost', 'iou_mode': 'giou', 'weight': 2.0}}, 'assigner_with_mask': {'type': 'HungarianAssigner_multi_info', 'cls_cost': {'type': 'FocalLossCost', 'weight': 2.0}, 'reg_cost': {'type': 'BBoxL1Cost', 'weight': 5.0, 'box_format': 'xywh'}, 'iou_cost': {'type': 'IoUCost', 'iou_mode': 'giou', 'weight': 2.0}, 'mask_cost': {'type': 'DiceCost', 'weight': 2.0}}, 'sampler': {'type': 'PseudoSampler'}, 'sampler_with_mask': {'type': 'PseudoSampler_segformer'}}
    transformer = {'type': 'SegDeformableTransformer', 'encoder': {'type': 'DetrTransformerEncoder', 'num_layers': 6, 'transformerlayers': {'type': 'BaseTransformerLayer', 'attn_cfgs': {'type': 'MultiScaleDeformableAttention', 'embed_dims': 256, 'num_levels': 4}, 'feedforward_channels': 512, 'ffn_dropout': 0.1, 'operation_order': ('self_attn', 'norm', 'ffn', 'norm')}}, 'decoder': {'type': 'DeformableDetrTransformerDecoder', 'num_layers': 6, 'return_intermediate': True, 'transformerlayers': {'type': 'DetrTransformerDecoderLayer', 'attn_cfgs': [{'type': 'MultiheadAttention', 'embed_dims': 256, 'num_heads': 8, 'dropout': 0.1}, {'type': 'MultiScaleDeformableAttention', 'embed_dims': 256, 'num_levels': 4}], 'feedforward_channels': 512, 'ffn_dropout': 0.1, 'operation_order': ('self_attn', 'norm', 'cross_attn', 'norm', 'ffn', 'norm')}}}
    kwargs = {'num_query': 300, 'num_classes': 4, 'num_things_classes': 3, 'num_stuff_classes': 1, 'in_channels': 2048, 'sync_cls_avg_factor': True, 'positional_encoding': {'type': 'SinePositionalEncoding', 'num_feats': 128, 'normalize': True, 'offset': -0.5}, 'loss_cls': {'type': 'FocalLoss', 'use_sigmoid': True, 'gamma': 2.0, 'alpha': 0.25, 'loss_weight': 2.0}, 'loss_bbox': {'type': 'L1Loss', 'loss_weight': 5.0}, 'loss_iou': {'type': 'GIoULoss', 'loss_weight': 2.0}}
    head = SegDETRHead(train_cfg=train_cfg,
                transformer=transformer,
                **kwargs)
    

def test_PansegformerHead():
    args = None
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
    # head = PansegformerHead(
    #     args,
    #     bev_h=bev_h,
    #     bev_w=bev_w,
    #     canvas_size=canvas_size,
    #     pc_range=pc_range,
    #     with_box_refine=with_box_refine, 
    # )


def test_occ_head():
    from src.occ_head.occ_head import OccHead
    cfg = {'type': 'OccHead', 'grid_conf': {'xbound': [-50.0, 50.0, 0.5], 'ybound': [-50.0, 50.0, 0.5], 'zbound': [-10.0, 10.0, 20.0]}, 'ignore_index': 255, 'bev_proj_dim': 256, 'bev_proj_nlayers': 4, 'attn_mask_thresh': 0.3, 'transformer_decoder': {'type': 'DetrTransformerDecoder', 'return_intermediate': True, 'num_layers': 5, 'transformerlayers': {'type': 'DetrTransformerDecoderLayer', 'attn_cfgs': {'type': 'MultiheadAttention', 'embed_dims': 256, 'num_heads': 8, 'attn_drop': 0.0, 'proj_drop': 0.0, 'dropout_layer': None, 'batch_first': False}, 'ffn_cfgs': {'type': 'FFN', 'embed_dims': 256, 'feedforward_channels': 2048, 'num_fcs': 2, 'act_cfg': {'type': 'ReLU', 'inplace': True}, 'ffn_drop': 0.0, 'dropout_layer': None, 'add_identity': True}, 'feedforward_channels': 2048, 'operation_order': ('self_attn', 'norm', 'cross_attn', 'norm', 'ffn', 'norm')}, 'init_cfg': None}, 'query_dim': 256, 'query_mlp_layers': 3, 'aux_loss_weight': 1.0, 'loss_mask': {'type': 'FieryBinarySegmentationLoss', 'use_top_k': True, 'top_k_ratio': 0.25, 'future_discount': 0.95, 'loss_weight': 5.0, 'ignore_index': 255}, 'loss_dice': {'type': 'DiceLossWithMasks', 'use_sigmoid': True, 'activate': True, 'reduction': 'mean', 'naive_dice': True, 'eps': 1.0, 'ignore_index': 255, 'loss_weight': 1.0}, 'pan_eval': True, 'test_seg_thresh': 0.1, 'test_with_track_score': True}
    cfg.pop('type')
    occhead_instance = OccHead(**cfg)
    print(occhead_instance)


def test_motion_head():
    from src.motion_head.motion_head import MotionHead
    cfg = {'type': 'MotionHead', 'bev_h': 200, 'bev_w': 200, 'num_query': 300, 'num_classes': 10, 'predict_steps': 12, 'predict_modes': 6, 'embed_dims': 256, 'loss_traj': {'type': 'TrajLoss', 'use_variance': True, 'cls_loss_weight': 0.5, 'nll_loss_weight': 0.5, 'loss_weight_minade': 0.0, 'loss_weight_minfde': 0.25}, 'num_cls_fcs': 3, 'pc_range': [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0], 'group_id_list': [[0, 1, 2, 3, 4], [6, 7], [8], [5, 9]], 'num_anchor': 6, 'use_nonlinear_optimizer': True, 'anchor_info_path': 'weights/motion_anchor_infos_mode6.pkl', 'transformerlayers': {'type': 'MotionTransformerDecoder', 'pc_range': [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0], 'embed_dims': 256, 'num_layers': 3, 'transformerlayers': {'type': 'MotionTransformerAttentionLayer', 'batch_first': True, 'attn_cfgs': [{'type': 'MotionDeformableAttention', 'num_steps': 12, 'embed_dims': 256, 'num_levels': 1, 'num_heads': 8, 'num_points': 4, 'sample_index': -1}], 'feedforward_channels': 512, 'ffn_dropout': 0.1, 'operation_order': ('cross_attn', 'norm', 'ffn', 'norm')}}}
    motionhead_instance = MotionHead(**cfg)
    print(motionhead_instance)

def test_bevformer_decoder():
    from src.bevformer.CustomMSDeformableAttention import DetectionTransformerDecoder, CustomMSDeformableAttention
    from src.utils.builder import build_transformer_layer_sequence
    cfg = {'type': 'DetectionTransformerDecoder', 'num_layers': 6, 'return_intermediate': True, 'transformerlayers': {'type': 'DetrTransformerDecoderLayer', 'attn_cfgs': [{'type': 'MultiheadAttention', 'embed_dims': 256, 'num_heads': 8, 'dropout': 0.1}, {'type': 'CustomMSDeformableAttention', 'embed_dims': 256, 'num_levels': 1}], 'feedforward_channels': 512, 'ffn_dropout': 0.1, 'operation_order': ('self_attn', 'norm', 'cross_attn', 'norm', 'ffn', 'norm')}}
    decoder = build_transformer_layer_sequence(cfg)
    print(decoder)
            

test_bevformer_decoder()
test_occ_head()
test_motion_head()